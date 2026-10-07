"""Stage 2: Transcript refinement via programmatic edits.

- RECOVER (arXiv 2603.16411): the LLM never regenerates the transcript.
  Phase 1 proposes [{"original", "correction"}]; Phase 2 (pure Python)
  validates and applies them with word-boundary regexes. No content drift.
- Safety filters: edits touching digits, number words, negation, commitment
  words (will/should/might...) or person names are rejected, as are edits
  whose span is not found verbatim or whose correction does not sound like
  the original (blocks synonym swaps / meaning changes).
- DARAG (Ghosh et al., ACL 2025): phonetic + fuzzy retrieval against an
  optional glossary; only retrieved candidates are shown to the LLM.
- Returns a diff log (applied + rejected) for the UI.

Note: Whisper already emits punctuation/capitalisation, so this stage only
fixes misrecognised terms. prompts/refine_prompt.txt is no longer used.
"""
import difflib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq
from pipeline.stt import PipelineError

load_dotenv()

REFINE_MODEL = "openai/gpt-oss-20b"
CHUNK_WORDS = 1200
MIN_WORDS = 10
DEFAULT_GLOSSARY_PATH = os.getenv("GLOSSARY_PATH", "glossary.json")

MAX_CANDIDATES = 12          # glossary candidates fed to the LLM per chunk
PHONETIC_THRESHOLD = 0.85    # retrieval: phonetic-key similarity
RAW_THRESHOLD = 0.82         # retrieval: raw string similarity
SOUNDS_LIKE_MIN = 0.5        # validation: original vs correction similarity
MAX_SPAN_WORDS = 6

# --------------------------------------------------------------------------
# Word lists for safety filters
# --------------------------------------------------------------------------
_NEG = {"not", "never", "no", "nor", "none", "neither", "cannot", "without",
        "nothing", "nobody", "nowhere", "hardly", "barely", "cant", "wont", "dont"}
_COMMIT = {"will", "shall", "should", "might", "must", "would", "could",
           "may", "can", "gonna"}
_NUMWORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "percent", "half", "dozen"}
_STOP = {
    "the", "a", "an", "to", "of", "and", "or", "in", "on", "at", "is", "it",
    "for", "that", "this", "with", "as", "be", "are", "was", "we", "i", "you",
    "he", "she", "they", "there", "who", "so", "if", "but", "by", "from",
    "everyone", "someone", "have", "has", "had", "do", "does", "did"}

_LETTERS = {
    "a": "a", "ay": "a", "b": "b", "bee": "b", "c": "c", "see": "c",
    "sea": "c", "cee": "c", "d": "d", "dee": "d", "e": "e", "ee": "e",
    "f": "f", "ef": "f", "eff": "f", "g": "g", "gee": "g", "h": "h",
    "aitch": "h", "i": "i", "eye": "i", "j": "j", "jay": "j", "k": "k",
    "kay": "k", "l": "l", "el": "l", "ell": "l", "m": "m", "em": "m",
    "n": "n", "en": "n", "o": "o", "oh": "o", "p": "p", "pee": "p",
    "q": "q", "queue": "q", "cue": "q", "r": "r", "are": "r", "ar": "r",
    "s": "s", "es": "s", "ess": "s", "t": "t", "tee": "t", "tea": "t",
    "u": "u", "you": "u", "v": "v", "vee": "v", "w": "w", "x": "x",
    "ex": "x", "y": "y", "why": "y", "z": "z", "zee": "z", "zed": "z"}

_WORD_RE = r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*"


def _words(text):
    return re.findall(_WORD_RE, text.lower())


def _is_neg(w):
    return w in _NEG or w.endswith("n't") or w.endswith("n’t")


# --------------------------------------------------------------------------
# Phonetic key (light-weight, dependency-free)
# --------------------------------------------------------------------------
_PHON_SUBS = [("ph", "f"), ("ck", "k"), ("kn", "n"), ("wr", "r"),
              ("gh", "g"), ("sch", "sk"), ("c", "k"), ("q", "k"),
              ("x", "ks"), ("z", "s"), ("v", "f")]


def _phon(text):
    """Consonant skeleton: 'kuber natties' and 'Kubernetes' -> 'kbrnts'."""
    s = re.sub(r"[^a-z]", "", text.lower())
    for a, b in _PHON_SUBS:
        s = s.replace(a, b)
    if not s:
        return ""
    out = [s[0]]
    for ch in s[1:]:
        if ch in "aeiouyh":
            continue
        if ch != out[-1]:
            out.append(ch)
    return "".join(out)


def _flat(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _ratio(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def _spelled_letters(text):
    """'see eye see dee' -> 'cicd' (only if every token is a letter name)."""
    toks = re.findall(r"[a-z]+", text.lower())
    if len(toks) < 2 or not all(t in _LETTERS for t in toks):
        return None
    return "".join(_LETTERS[t] for t in toks)


# --------------------------------------------------------------------------
# Glossary (DARAG-style retrieval)
# --------------------------------------------------------------------------
@dataclass
class Glossary:
    terms: List[str] = field(default_factory=list)
    aliases: Dict[str, str] = field(
        default_factory=dict)   # flat alias -> term
    names: set = field(default_factory=set)                 # protected names


def _ingest(data, terms, aliases, names):
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                terms.append(item)
            elif isinstance(item, dict) and isinstance(item.get("term"), str):
                terms.append(item["term"])
                for a in item.get("aliases", []) or []:
                    aliases[_flat(str(a))] = item["term"]
    elif isinstance(data, dict):
        if "terms" in data or "names" in data:
            _ingest(data.get("terms", []), terms, aliases, names)
            names.update(str(n).lower() for n in data.get("names", []) or [])
        else:  # {"Kubernetes": ["kuber natties", ...]}
            for term, al in data.items():
                terms.append(term)
                if isinstance(al, list):
                    for a in al:
                        aliases[_flat(str(a))] = term


def load_glossary(path=None, extra_text="") -> Glossary:
    """Merge optional glossary.json with terms typed in the UI (comma/newline)."""
    terms, aliases, names = [], {}, set()
    path = path or DEFAULT_GLOSSARY_PATH
    if path and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                _ingest(json.load(f), terms, aliases, names)
        except (OSError, ValueError):
            pass  # malformed glossary must never break the pipeline
    for t in re.split(r"[,\n;]+", extra_text or ""):
        t = t.strip()
        if t:
            terms.append(t)
    seen, uniq = set(), []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            uniq.append(t)
    return Glossary(uniq, aliases, names)


def retrieve_candidates(chunk, glossary, k=MAX_CANDIDATES):
    """Phonetic/fuzzy match n-grams of the chunk against glossary terms.

    Returns [{"term", "span", "score"}] sorted by score. Spans already equal
    to the term are skipped (nothing to fix).
    """
    if not glossary.terms:
        return []
    info = [(t, _flat(t), _phon(t)) for t in glossary.terms]
    by_key: Dict[str, list] = {}
    for t, f, key in info:
        if len(key) >= 3:
            by_key.setdefault(key, []).append((t, f))

    toks = re.findall(_WORD_RE, chunk)
    best: Dict[tuple, float] = {}

    for n in (1, 2, 3, 4):
        for i in range(len(toks) - n + 1):
            span_toks = toks[i:i + n]
            if span_toks[0].lower() in _STOP or span_toks[-1].lower() in _STOP:
                continue
            span = " ".join(span_toks)
            sf = _flat(span)
            if not sf:
                continue
            if sf in glossary.aliases:
                best[(glossary.aliases[sf], span)] = 1.0
                continue
            key = _phon(span)
            if len(key) < 3:
                continue
            for t, tf in by_key.get(key, []):
                if sf != tf:
                    best[(t, span)] = max(best.get((t, span), 0), 1.0)
            for t, tf, tkey in info:
                if sf == tf or len(tkey) < 3 or (t, span) in best:
                    continue
                score = 0.0
                if key[0] == tkey[0] and abs(len(key) - len(tkey)) <= 1:
                    r = _ratio(key, tkey)
                    if r >= PHONETIC_THRESHOLD:
                        score = r
                if abs(len(sf) - len(tf)) <= 3:
                    r = _ratio(sf, tf)
                    if r >= RAW_THRESHOLD:
                        score = max(score, r)
                if score:
                    best[(t, span)] = score

    ranked = sorted(best.items(), key=lambda kv: -kv[1])[:k]
    return [{"term": t, "span": s, "score": round(sc, 2)}
            for (t, s), sc in ranked]


# --------------------------------------------------------------------------
# Person-name protection (heuristic; no NER dependency)
# --------------------------------------------------------------------------
_NAME_CUES = (r"(?i:thanks|thank you|hi|hey|hello|ask|tell|ping|cc|"
              r"assigned to|assign to|owner is|over to|hear from)")
_SPEECH_VERBS = (r"(?:will|said|says|mentioned|agreed|suggested|asked|owns|"
                 r"takes|volunteered|confirmed)")
_CAP = r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"


def find_protected_names(text, glossary) -> set:
    found = set()
    patterns = [
        # "thanks Priya", "ask John Smith"
        rf"\b{_NAME_CUES}\s+{_CAP}",
        rf"\b{_CAP}\s+{_SPEECH_VERBS}\b",      # "Priya will ..."
        rf"(?m)^\s*{_CAP}\s*:",                # "Priya: ..." speaker labels
    ]
    for p in patterns:
        for m in re.finditer(p, text):
            for tok in m.group(1).lower().split():
                if tok not in _STOP:
                    found.add(tok)
    term_tokens = {t for term in glossary.terms for t in _words(term)}
    found -= term_tokens                       # glossary terms are not names
    found |= {n for n in glossary.names}       # explicitly protected names
    return found


# --------------------------------------------------------------------------
# Phase 1: LLM proposes edits (never rewrites text)
# --------------------------------------------------------------------------
_SYSTEM = (
    "You are a speech-recognition error detector for a work-meeting "
    "transcript. Find ONLY misrecognised technical terms, product names, "
    "acronyms and common meeting words (deadline, timeline, owner, milestone, "
    "budget). Return JSON: {\"errors\": [{\"original\": \"...\", "
    "\"correction\": \"...\"}]}.\n"
    "Rules:\n"
    "1. \"original\" must be copied EXACTLY from the transcript and be as "
    "short as possible (only the wrong words).\n"
    "2. Never touch numbers, dates, amounts, units; negation or commitment "
    "words (not, never, won't, will, should, might); or person names. If "
    "unsure, do not list it.\n"
    "3. Do not list synonym swaps, grammar, punctuation or capitalisation.\n"
    "4. Misheard technical terms often become ordinary English words (e.g. "
    "'Arboretum cache' for 'Redis cache'). If an ordinary word makes no "
    "sense in a technical phrase, propose the technical term it most likely "
    "sounds like. Use the same correction everywhere it appears.\n"
    "5. If glossary candidates are given, use one only when the transcript "
    "clearly contains a misrecognition of that term.\n"
    "6. If there are no errors return {\"errors\": []}. JSON only."
)


def _identify_errors(client, chunk, candidates):
    user = ""
    if candidates:
        user += ("Glossary candidates (possible matches, verify in context):\n"
                 + "\n".join(f"- heard \"{c['span']}\" -> may be \"{c['term']}\""
                             for c in candidates) + "\n\n")
    user += "Transcript:\n" + chunk
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=REFINE_MODEL,
                messages=[{"role": "system", "content": _SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0,
                response_format={"type": "json_object"},
                max_completion_tokens=2000,
                extra_body={"reasoning_effort": "low"},
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            errors = data.get("errors", [])
            return errors if isinstance(errors, list) else []
        except (ValueError, TypeError):
            continue
        except Exception:
            break
    return []   # on failure: leave the chunk untouched (fail safe)


# --------------------------------------------------------------------------
# Phase 2: deterministic validation + application
# --------------------------------------------------------------------------
def _clean_span(x):
    return x.strip(" \t\r\n\"'`.,;:!?()[]") if isinstance(x, str) else ""


def _norm_key(s):
    return " ".join(s.split()).lower()


def _pattern(original):
    body = r"\s+".join(re.escape(t) for t in original.split())
    return rf"(?<!\w){body}(?!\w)"


def _sounds_like(orig, corr, glossary, retrieved_spans):
    if _flat(corr) in {_flat(t) for t in glossary.terms} and \
            _flat(orig) in retrieved_spans:
        return True
    spelled = _spelled_letters(orig)
    if spelled and spelled == re.sub(r"[^a-z]", "", corr.lower()):
        return True
    return _ratio(_phon(orig), _phon(corr)) >= SOUNDS_LIKE_MIN


def validate_edit(orig, corr, glossary, names, retrieved_spans) -> Optional[str]:
    """Return a rejection reason, or None if the edit is safe."""
    if not orig or not corr:
        return "empty_edit"
    if orig == corr:
        return "no_change"
    if re.search(r"\d", orig + corr):
        return "touches_digits"
    ow, cw = _words(orig), _words(corr)
    both = set(ow) | set(cw)
    if both & _NUMWORDS:
        return "touches_number_words"
    if any(_is_neg(w) for w in both):
        return "touches_negation"
    if both & _COMMIT:
        return "touches_commitment_word"
    if len(ow) > MAX_SPAN_WORDS or len(cw) > MAX_SPAN_WORDS:
        return "span_too_long"
    if len(ow) == 1 and (len(ow[0]) <= 2 or ow[0] in _STOP):
        return "ordinary_word"
    if both & names:
        return "touches_person_name"
    if not _sounds_like(orig, corr, glossary, retrieved_spans):
        return "not_a_sound_alike"
    return None


def _apply_edits(text, edits: Dict[str, tuple]):
    """Single-pass regex substitution (no chaining). Returns (text, stats)."""
    if not edits:
        return text, {}
    order = sorted(edits, key=len, reverse=True)
    rx = re.compile("|".join(f"(?:{_pattern(edits[k][0])})" for k in order),
                    re.IGNORECASE)
    stats: Dict[str, dict] = {}

    def repl(m):
        key = _norm_key(m.group(0))
        if key not in edits:
            return m.group(0)
        s = stats.setdefault(key, {"count": 0, "context": ""})
        s["count"] += 1
        if not s["context"]:
            a, b = max(0, m.start() - 45), min(len(text), m.end() + 45)
            ctx = (text[a:m.start()] + "«" +
                   m.group(0) + "»" + text[m.end():b])
            s["context"] = " ".join(ctx.split())
        return edits[key][1]

    return rx.sub(repl, text), stats


def _signature(text):
    """Numbers and negations must be identical before/after refinement."""
    nums = re.findall(r"\d+(?:[.,:/]\d+)*", text)
    negs = [w for w in _words(text) if _is_neg(w)]
    return nums, negs


def _chunks(text, size=CHUNK_WORDS):
    cur, n = [], 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        w = len(sent.split())
        if cur and n + w > size:
            yield " ".join(cur)
            cur, n = [], 0
        cur.append(sent)
        n += w
    if cur:
        yield " ".join(cur)


# --------------------------------------------------------------------------
# Result + public API
# --------------------------------------------------------------------------
@dataclass
class RefineResult:
    text: str
    applied: List[dict] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)
    candidates: List[dict] = field(default_factory=list)

    @property
    def rows(self):
        """Rows for gr.Dataframe: [original, correction, count, context]."""
        return [[a["original"], a["correction"], a["count"], a["context"]]
                for a in self.applied]

    @property
    def rejected_rows(self):
        return [[r["original"], r["correction"], r["reason"]]
                for r in self.rejected]

    def to_markdown(self):
        def esc(s): return str(s).replace("|", "\\|")
        if not self.applied:
            return "_No corrections were applied._"
        out = ["| Original | Correction | Count | Context |",
               "|---|---|---|---|"]
        out += [f"| {esc(a['original'])} | {esc(a['correction'])} | "
                f"{a['count']} | {esc(a['context'])} |" for a in self.applied]
        return "\n".join(out)


def refine(raw_text, glossary_text="", glossary_path=None) -> RefineResult:
    """Raw transcript -> RefineResult(text, applied, rejected, candidates)."""
    if not raw_text or not raw_text.strip():
        raise PipelineError("There is no transcript to refine.")
    wc = len(raw_text.split())
    if wc < MIN_WORDS:
        raise PipelineError(
            f"The transcript is too short to refine ({wc} words). "
            "Please check your audio file.")

    try:
        client = Groq()
        glossary = load_glossary(glossary_path, glossary_text)
        names = find_protected_names(raw_text, glossary)

        proposals, all_cands = [], []
        for chunk in _chunks(raw_text):
            cands = retrieve_candidates(chunk, glossary)
            all_cands.extend(cands)
            proposals.extend(_identify_errors(client, chunk, cands))
        retrieved_spans = {_flat(c["span"]) for c in all_cands}

        accepted: Dict[str, tuple] = {}
        rejected, seen_rej = [], set()
        for p in proposals:
            if not isinstance(p, dict):
                continue
            orig = _clean_span(p.get("original"))
            corr = _clean_span(p.get("correction"))
            reason = validate_edit(orig, corr, glossary,
                                   names, retrieved_spans)
            if not reason and not re.search(_pattern(orig), raw_text, re.I):
                reason = "not_found_in_transcript"
            key = _norm_key(orig)
            if not reason and key in accepted and accepted[key][1] != corr:
                reason = "conflicting_correction"
            if reason:
                sig = (orig, corr, reason)
                if sig not in seen_rej:
                    seen_rej.add(sig)
                    rejected.append({"original": orig, "correction": corr,
                                     "reason": reason})
                continue
            accepted.setdefault(key, (orig, corr))

        new_text, stats = _apply_edits(raw_text, accepted)

        # Final integrity assertion: numbers and negations must be unchanged.
        if _signature(new_text) != _signature(raw_text):
            return RefineResult(raw_text, [], rejected + [
                {"original": "*", "correction": "*",
                 "reason": "integrity_check_failed_reverted"}], all_cands)

        applied = [{"original": accepted[k][0], "correction": accepted[k][1],
                    "count": s["count"], "context": s["context"]}
                   for k, s in stats.items()]
        return RefineResult(new_text, applied, rejected, all_cands)
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(f"Transcript refinement failed: {e}")


def refine_text(raw_text, glossary_text="") -> str:
    """Convenience wrapper returning only the refined string."""
    return refine(raw_text, glossary_text).text
