"""Stage 3: Meeting documentation + verification.

Pipeline:
  1. FRAME (Kirstein et al., EMNLP 2025): the LLM returns only sentence
     indices + topic names; [TOPIC: ...] markers are inserted in Python, so
     the transcript is never regenerated (no token cost, no word drift).
  2. Draft record (gpt-oss-120b, JSON mode).
  3. Critic pass (Kirstein et al., COLING 2025): a second LLM pass lists
     errors (hallucinated owner, proposal-as-decision, negation flip, ...);
     fixes are applied in Python, and text rewrites are accepted only if
     they pass deterministic checks (numbers + negation count).
  4. Chain-of-Verification (Dhuliawala et al., ACL 2024), factored:
       Step 1 (deterministic): every source_quote is fuzzy-matched against
         the transcript. Missing quotes -> item dropped; weak -> flagged;
         matched quotes are snapped to the exact transcript wording.
       Step 2 (LLM, independent): cheap calls that see ONLY the quote and a
         small context window (never the draft), asking "explicit group
         agreement?" / "is a person explicitly assigned?". Failures demote
         the decision to open_items or reset the owner to "unspecified".
  Every change is written to an audit log for the UI.
"""
import bisect
import difflib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ValidationError
from pipeline.stt import PipelineError
from pipeline.refine import _is_neg, _words

load_dotenv()

DOC_MODEL = "openai/gpt-oss-120b"
PROMPT_PATH = os.path.join(os.path.dirname(
    __file__), "..", "prompts", "document_prompt.txt")

MIN_WORDS = 10               # Xie et al. (2025): inter-stage validation
SEGMENT_THRESHOLD = 300      # only segment transcripts longer than this
MIN_SEGMENT_SENTENCES = 3    # FRAME: ignore tiny segments
QUOTE_PASS = 0.85            # CoVe step 1: quote considered verbatim
QUOTE_DROP = 0.60            # below this the quote is "not in transcript"
CONTEXT_SENTENCES = 2        # CoVe step 2: sentences either side of quote
COVE_WORKERS = 4
_NUM_RE = r"\d+(?:[.,:/]\d+)*"


# ---------------------------------------------------------------------------
# Pydantic models (the exact JSON schema the app exports)
# ---------------------------------------------------------------------------
class MinutesSection(BaseModel):
    topic: str
    points: List[str]


class Decision(BaseModel):
    decision: str
    source_quote: str = ""


class ActionItem(BaseModel):
    task: str
    owner: str = "unspecified"
    deadline: str = "unspecified"
    source_quote: str = ""


class OpenItem(BaseModel):
    item: str
    note: str = ""


class MeetingRecord(BaseModel):
    tldr: str
    summary: str
    minutes: List[MinutesSection]
    decisions: List[Decision]
    action_items: List[ActionItem]
    open_items: List[OpenItem] = []


@dataclass
class DocumentResult:
    """Record + the evidence trail of every automatic change."""
    record: MeetingRecord
    segmented_text: str = ""
    audit: List[dict] = field(default_factory=list)

    @property
    def audit_rows(self):
        """Rows for gr.Dataframe: [stage, section, item, action, detail]."""
        return [[a["stage"], a["section"], a["item"], a["action"], a["detail"]]
                for a in self.audit]

    def audit_markdown(self):
        if not self.audit:
            return "_All items passed verification unchanged._"

        def esc(s): return str(s).replace("|", "\\|")
        out = ["| Stage | Section | Item | Action | Detail |",
               "|---|---|---|---|---|"]
        out += [f"| {esc(a['stage'])} | {esc(a['section'])} | {esc(a['item'])} "
                f"| {esc(a['action'])} | {esc(a['detail'])} |"
                for a in self.audit]
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
_TOK = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*")


def _tokens(text):
    """[(lowercase token, char_start, char_end)] with offsets into `text`."""
    return [(m.group().lower().replace("’", "'"), m.start(), m.end())
            for m in _TOK.finditer(text)]


def _sentence_spans(text):
    return [(m.start(), m.end()) for m in
            re.finditer(r"\S.*?(?:[.!?]+(?=\s|$)|$)", text, re.S)]


def _neg_count(text):
    return sum(1 for w in _words(text) if _is_neg(w))


def _short(s, n=90):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n - 1] + "…"


def _log(audit, stage, section, item, action, detail=""):
    audit.append({"stage": stage, "section": section, "item": _short(item),
                  "action": action, "detail": detail})


def _chat_json(client, system, user, max_tokens, effort="low"):
    """Auxiliary JSON call. Returns a dict, or None on any failure
    (callers treat None as 'check unavailable' and fail safe)."""
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=DOC_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0,
                response_format={"type": "json_object"},
                max_completion_tokens=max_tokens,
                extra_body={"reasoning_effort": effort},
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            if isinstance(data, dict):
                return data
        except (ValueError, TypeError):
            continue
        except Exception:
            break
    return None


def locate_quote(quote, toks) -> Tuple[float, Optional[Tuple[int, int]]]:
    """Fuzzy-find `quote` in the tokenised transcript.

    Returns (similarity 0..1, (char_start, char_end) in the original text).
    Windows are only tried around content-word hits, so it stays fast on
    long transcripts.
    """
    qt = [t for t, _, _ in _tokens(quote or "")]
    if not qt or not toks:
        return 0.0, None
    words = [t for t, _, _ in toks]
    qs, n_q = " ".join(qt), len(qt)
    key = {w for w in qt if len(w) > 3} or set(qt)
    starts = set()
    for h, w in enumerate(words):
        if w in key:
            starts.update(range(max(0, h - 2), h + 1))
    best, span = 0.0, None
    for i in sorted(starts):
        for n in {max(1, n_q - 1), n_q, n_q + 1}:
            j = i + n
            if j > len(words):
                continue
            r = difflib.SequenceMatcher(None, qs, " ".join(words[i:j])).ratio()
            if r > best:
                best, span = r, (toks[i][1], toks[j - 1][2])
                if r == 1.0:
                    return best, span
    if span and best >= QUOTE_PASS:
        # Fuzzy similarity is blind to "will" vs "will not": a quote whose
        # negation count or numbers differ from the matched text can never
        # count as a verbatim match.
        matched_neg = sum(1 for w in qt if _is_neg(w))
        found = [t for t, a, b in toks if a >= span[0] and b <= span[1]]
        if (matched_neg != sum(1 for w in found if _is_neg(w))
                or sorted(re.findall(r"\d+", qs)) != sorted(re.findall(r"\d+", " ".join(found)))):
            best = min(best, QUOTE_PASS - 0.01)
    return best, span


def _context_window(text, sents, span, n=CONTEXT_SENTENCES):
    starts = [a for a, _ in sents]
    first = max(0, bisect.bisect_right(starts, span[0]) - 1)
    last = max(first, bisect.bisect_right(
        starts, max(span[0], span[1] - 1)) - 1)
    lo, hi = max(0, first - n), min(len(sents) - 1, last + n)
    return text[sents[lo][0]:sents[hi][1]]


# ---------------------------------------------------------------------------
# 1. FRAME: topic segmentation by sentence index
# ---------------------------------------------------------------------------
_FRAME_SYSTEM = (
    "You segment a meeting transcript by topic. The transcript is given as "
    "numbered sentences like [12] text. Return JSON: "
    "{\"segments\": [{\"start\": <number of the sentence where a new topic "
    "begins>, \"topic\": \"<2-5 word topic name>\"}]}. "
    "The first segment must start at 0. Use only sentence numbers that "
    "appear. Prefer a few coherent segments (roughly one per 8-15 "
    "sentences). Never return transcript text. JSON only."
)


def _frame_segment(client, text, audit):
    """Insert [TOPIC: ...] markers programmatically (FRAME)."""
    if len(text.split()) < SEGMENT_THRESHOLD:
        return text
    spans = _sentence_spans(text)
    sents = [text[a:b].strip() for a, b in spans]
    if len(sents) < 2 * MIN_SEGMENT_SENTENCES:
        return text

    data = _chat_json(client, _FRAME_SYSTEM,
                      "\n".join(f"[{i}] {s}" for i, s in enumerate(sents)),
                      max_tokens=1500)
    raw = (data or {}).get("segments")
    if not isinstance(raw, list):
        _log(audit, "frame", "-", "-", "skipped",
             "segmentation unavailable; using unsegmented transcript")
        return text

    marks = {}
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        try:
            idx = int(seg.get("start"))
        except (TypeError, ValueError):
            continue
        topic = " ".join(
            re.sub(r"[\[\]:]+", " ", str(seg.get("topic", ""))).split())[:60]
        if 0 <= idx < len(sents) and topic:
            marks.setdefault(idx, topic)
    if not marks:
        return text
    marks.setdefault(0, "Introduction")

    kept, last = {}, None
    for idx in sorted(marks):                     # drop tiny segments
        if last is None or idx - last >= MIN_SEGMENT_SENTENCES:
            kept[idx] = marks[idx]
            last = idx
    if len(kept) < 2:
        return text

    order, blocks, cur = sorted(kept), [], []
    for i, s in enumerate(sents):
        if i in kept:
            if cur:
                blocks.append(" ".join(cur))
            cur = [f"[TOPIC: {kept[i]}]\n{s}"]
        else:
            cur.append(s)
    blocks.append(" ".join(cur))
    _log(audit, "frame", "-", f"{len(order)} topics", "segmented",
         "markers inserted programmatically from sentence indices")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 2. Draft record
# ---------------------------------------------------------------------------
def _clean(record, transcript):
    """Safety net: an owner must at least appear in the transcript."""
    low = transcript.lower()
    for a in record.action_items:
        a.owner = (a.owner or "").strip() or "unspecified"
        a.deadline = (a.deadline or "").strip() or "unspecified"
        if a.owner.lower() != "unspecified":
            names = [t for t in re.split(
                r"\W+", a.owner.lower()) if len(t) > 2]
            if not any(n in low for n in names):
                a.owner = "unspecified"
    return record


def _draft_record(client, system, segmented_text):
    last_err = None
    for _ in range(2):  # one retry on bad JSON
        try:
            resp = client.chat.completions.create(
                model=DOC_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": segmented_text}],
                temperature=0.1,
                response_format={"type": "json_object"},
                max_completion_tokens=8000,
            )
            return MeetingRecord.model_validate(
                json.loads(resp.choices[0].message.content))
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            last_err = e
        except Exception as e:
            raise PipelineError(f"Meeting documentation failed: {e}")
    raise PipelineError(
        f"The model returned an invalid record twice: {last_err}")


# ---------------------------------------------------------------------------
# 3. Critic pass (COLING 2025) with programmatic fixes
# ---------------------------------------------------------------------------
_CRITIC_SYSTEM = (
    "You are a meticulous auditor of meeting records. You receive a "
    "TRANSCRIPT and a DRAFT RECORD (JSON). Find real errors by checking the "
    "draft against the transcript only. Report an issue only if confident.\n"
    "Error types:\n"
    "- hallucinated_owner: action item owner not clearly assigned in the transcript\n"
    "- hallucinated_deadline: deadline not stated in the transcript\n"
    "- proposal_as_decision: a decisions entry that was only a suggestion, idea, "
    "proposal, 'maybe' or 'we could', not clearly agreed\n"
    "- negation_flip: a statement that reverses or drops a negation\n"
    "- number_changed: a number, date, name or unit differs from the transcript\n"
    "- unsupported_fact: a fact, reason or name not in the transcript\n"
    "- status_word_added: timing/urgency/status words that were not said\n"
    "Lists are 0-based. For minutes use index = topic position and "
    "point_index = point position inside that topic. For summary or tldr use index 0.\n"
    "Return JSON: {\"issues\": [{\"section\": "
    "\"decisions|action_items|open_items|minutes|summary|tldr\", \"index\": 0, "
    "\"point_index\": null, \"type\": \"...\", \"evidence\": \"short exact quote "
    "from the transcript, or empty\", \"corrected\": \"corrected statement for "
    "this item, or empty if it should simply be removed or reset\"}]}. "
    "If there are no errors return {\"issues\": []}. JSON only."
)
_TEXT_FIX_TYPES = {"negation_flip", "number_changed",
                   "unsupported_fact", "status_word_added"}
_TEXT_ATTR = {"decisions": "decision",
              "action_items": "task", "open_items": "item"}


def _safe_correction(old, new, evidence, transcript, toks):
    """A critic rewrite is applied only if it keeps numbers and negation
    consistent with the transcript evidence."""
    if not new or " ".join(new.split()) == " ".join(old.split()) or not evidence:
        return False
    if not set(re.findall(_NUM_RE, new)) <= set(re.findall(_NUM_RE, transcript)):
        return False
    score, span = locate_quote(evidence, toks)
    if score < QUOTE_PASS or not span:
        return False
    return _neg_count(new) == _neg_count(transcript[span[0]:span[1]])


def _critic_pass(client, record, transcript, toks, audit):
    user = ("TRANSCRIPT:\n" + transcript + "\n\nDRAFT RECORD (JSON):\n" +
            json.dumps(record.model_dump(), ensure_ascii=False, indent=1))
    data = _chat_json(client, _CRITIC_SYSTEM, user, max_tokens=3000,
                      effort="medium")
    if data is None:
        _log(audit, "critic", "-", "-", "skipped", "critic unavailable")
        return record
    issues = data.get("issues")
    if not isinstance(issues, list):
        return record

    move_dec, drop_open, drop_pts = set(), set(), set()
    for iss in issues:
        if not isinstance(iss, dict):
            continue
        sec, typ = iss.get("section"), str(iss.get("type", ""))
        idx, pidx = iss.get("index"), iss.get("point_index")
        corrected = iss.get("corrected")
        corrected = corrected.strip() if isinstance(corrected, str) else ""
        evidence = iss.get("evidence")
        evidence = evidence.strip() if isinstance(evidence, str) else ""
        tag = f"critic:{typ}"

        if sec in _TEXT_ATTR:
            lst = getattr(record, sec)
            if not isinstance(idx, int) or not 0 <= idx < len(lst):
                continue
            obj, attr = lst[idx], _TEXT_ATTR[sec]
            label = getattr(obj, attr)
            if typ == "hallucinated_owner" and sec == "action_items":
                obj.owner = "unspecified"
                _log(audit, "critic", sec, label, "owner_reset", typ)
            elif typ == "hallucinated_deadline" and sec == "action_items":
                obj.deadline = "unspecified"
                _log(audit, "critic", sec, label, "deadline_reset", typ)
            elif typ == "proposal_as_decision" and sec == "decisions":
                move_dec.add(idx)
                _log(audit, "critic", sec, label, "moved_to_open_items", typ)
            elif typ in _TEXT_FIX_TYPES:
                if _safe_correction(label, corrected, evidence, transcript, toks):
                    setattr(obj, attr, corrected)
                    _log(audit, "critic", sec, label, "rewritten",
                         f"{typ}; rewrite passed number/negation check")
                elif sec in ("decisions", "action_items") and obj.source_quote:
                    setattr(
                        obj, attr, f'Transcript states: "{obj.source_quote}"')
                    _log(audit, "critic", sec, label,
                         "replaced_with_quote", typ)
                elif sec == "open_items":
                    drop_open.add(idx)
                    _log(audit, "critic", sec, label, "removed", typ)
        elif sec == "minutes":
            if not (isinstance(idx, int) and isinstance(pidx, int)
                    and 0 <= idx < len(record.minutes)
                    and 0 <= pidx < len(record.minutes[idx].points)):
                continue
            pts = record.minutes[idx].points
            if typ in _TEXT_FIX_TYPES:
                if _safe_correction(pts[pidx], corrected, evidence, transcript, toks):
                    _log(audit, "critic", sec, pts[pidx], "rewritten", typ)
                    pts[pidx] = corrected
                else:
                    drop_pts.add((idx, pidx))
                    _log(audit, "critic", sec, pts[pidx], "removed", typ)
        elif sec in ("summary", "tldr") and typ in _TEXT_FIX_TYPES:
            old = getattr(record, sec)
            if _safe_correction(old, corrected, evidence, transcript, toks):
                setattr(record, sec, corrected)
                _log(audit, "critic", sec, old, "rewritten", typ)
            else:
                _log(audit, "critic", sec, old, "flagged_unfixed", typ)

    moved = [record.decisions[i] for i in sorted(move_dec)]
    record.decisions = [d for i, d in enumerate(
        record.decisions) if i not in move_dec]
    record.open_items = [o for i, o in enumerate(
        record.open_items) if i not in drop_open]
    record.open_items += [OpenItem(
        item=d.decision,
        note="Proposed or discussed; the transcript does not show explicit group agreement.")
        for d in moved]
    for ti, sec_ in enumerate(record.minutes):
        sec_.points = [p for pi, p in enumerate(
            sec_.points) if (ti, pi) not in drop_pts]
    record.minutes = [m for m in record.minutes if m.points]
    return record


# ---------------------------------------------------------------------------
# 4. Chain-of-Verification
# ---------------------------------------------------------------------------
def _cove_step1(record, transcript, toks, audit):
    """Deterministic: every source_quote must exist in the transcript.

    Returns (decision spans, action-item spans) aligned with the filtered
    lists. Matched quotes are snapped to the exact transcript wording.
    """
    def check(items, section, label_attr):
        kept, spans = [], []
        for it in items:
            label = getattr(it, label_attr)
            score, span = locate_quote(it.source_quote, toks)
            if score < QUOTE_DROP or not span:
                _log(audit, "cove-1", section, label, "dropped",
                     f"source_quote not found in transcript (similarity {score:.2f})")
                continue
            it.source_quote = transcript[span[0]:span[1]]
            if score < QUOTE_PASS:
                _log(audit, "cove-1", section, label, "flagged",
                     f"weak quote match ({score:.2f}); quote snapped to transcript text")
            kept.append(it)
            spans.append(span)
        return kept, spans

    record.decisions, dec_spans = check(
        record.decisions, "decisions", "decision")
    record.action_items, act_spans = check(
        record.action_items, "action_items", "task")

    for a in record.action_items:                 # deadline must be stated
        if a.deadline.lower() != "unspecified":
            score, _ = locate_quote(a.deadline, toks)
            if score < QUOTE_PASS:
                _log(audit, "cove-1", "action_items", a.task, "deadline_reset",
                     f"deadline '{a.deadline}' not found in transcript")
                a.deadline = "unspecified"
    return dec_spans, act_spans


_COVE_DECISION = (
    "You verify one passage from a meeting transcript. Use ONLY the passage. "
    "Question: does the focus quote show explicit group agreement or approval "
    "(e.g. 'we decided', 'we will', 'approved', 'that's final')? Ideas, "
    "suggestions, proposals, 'maybe', 'we could', 'should we' and questions "
    "are NOT agreement. Return JSON: {\"explicit_agreement\": true|false, "
    "\"reason\": \"short\"}."
)
_COVE_OWNER = (
    "You verify one passage from a meeting transcript. Use ONLY the passage. "
    "Question: is a specific person explicitly assigned to do the work in the "
    "focus quote, or does a named person explicitly volunteer for it? "
    "'Someone should', 'we need to', 'everyone', 'the team', or 'I'm not "
    "assigning that yet' mean NO. Return JSON: {\"explicit\": true|false, "
    "\"assigned_person\": \"name exactly as written in the passage, or none\"}."
)


def _ask(client, system, context, quote):
    return _chat_json(client, system,
                      f"Passage:\n{context}\n\nFocus quote:\n{quote}", max_tokens=300)


def _cove_step2(client, record, transcript, sents, dec_spans, act_spans, audit):
    """Independent LLM checks: each call sees only quote + context window."""
    jobs = []
    for d, sp in zip(record.decisions, dec_spans):
        jobs.append(("decision", d, _COVE_DECISION,
                     _context_window(transcript, sents, sp), d.source_quote))
    for a, sp in zip(record.action_items, act_spans):
        if a.owner.lower() != "unspecified":
            jobs.append(("owner", a, _COVE_OWNER,
                         _context_window(transcript, sents, sp), a.source_quote))
    if not jobs:
        return record

    with ThreadPoolExecutor(max_workers=COVE_WORKERS) as pool:
        results = list(pool.map(
            lambda j: _ask(client, j[2], j[3], j[4]), jobs))

    demote = []
    for (kind, obj, _, _, _), res in zip(jobs, results):
        if res is None:
            _log(audit, "cove-2", kind, getattr(obj, "decision", None) or obj.task,
                 "check_unavailable", "LLM check failed; item kept")
            continue
        if kind == "decision":
            if res.get("explicit_agreement") is False:
                demote.append(obj)
                _log(audit, "cove-2", "decisions", obj.decision,
                     "moved_to_open_items", str(res.get("reason", ""))[:160])
        else:
            who = str(res.get("assigned_person", "none")).lower()
            owner_toks = [t for t in re.split(
                r"\W+", obj.owner.lower()) if len(t) > 2]
            ok = (res.get("explicit") is not False and who not in ("", "none")
                  and any(t in who for t in owner_toks))
            if not ok:
                _log(audit, "cove-2", "action_items", obj.task, "owner_reset",
                     f"no explicit assignment of '{obj.owner}' in context")
                obj.owner = "unspecified"

    if demote:
        ids = {id(d) for d in demote}
        record.decisions = [d for d in record.decisions if id(d) not in ids]
        record.open_items += [OpenItem(
            item=d.decision,
            note="Quoted passage does not show explicit group agreement.")
            for d in demote]
    return record


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_record(refined_text) -> DocumentResult:
    """Refined transcript -> DocumentResult(record, segmented_text, audit)."""
    if not refined_text or not refined_text.strip():
        raise PipelineError("There is no transcript to document.")
    word_count = len(refined_text.split())
    if word_count < MIN_WORDS:
        raise PipelineError(
            f"The transcript is too short to generate meeting records ({word_count} words). "
            "Please upload a longer recording with clear speech.")

    with open(PROMPT_PATH, encoding="utf-8") as f:
        system = f.read()

    client = Groq()
    audit: List[dict] = []

    segmented = _frame_segment(client, refined_text, audit)
    record = _clean(_draft_record(client, system, segmented), refined_text)

    if record.tldr.strip().lower().startswith("no meeting content"):
        return DocumentResult(record, segmented, audit)

    toks = _tokens(refined_text)
    sents = _sentence_spans(refined_text)

    try:
        record = _critic_pass(client, record, refined_text, toks, audit)
    except Exception as e:      # verification must never break the pipeline
        _log(audit, "critic", "-", "-", "skipped", f"error: {e}")
    try:
        dec_spans, act_spans = _cove_step1(record, refined_text, toks, audit)
        record = _cove_step2(client, record, refined_text, sents,
                             dec_spans, act_spans, audit)
    except Exception as e:
        _log(audit, "cove", "-", "-", "skipped", f"error: {e}")

    return DocumentResult(_clean(record, refined_text), segmented, audit)


def to_markdown(r):
    """Human-readable version of the record (TL;DR first, then detail)."""
    out = ["# Meeting Record", "", "## TL;DR", r.tldr, "",
           "## Summary", r.summary, "", "## Minutes"]
    for s in r.minutes:
        out.append(f"### {s.topic}")
        out += [f"- {p}" for p in s.points]
        out.append("")
    out.append("## Key Decisions")
    out += [f"- {d.decision}" for d in r.decisions] or ["- None recorded"]
    out += ["", "## Action Items"]
    for a in r.action_items:
        out.append(f"- {a.task} (Owner: {a.owner}; Deadline: {a.deadline})")
    if not r.action_items:
        out.append("- None recorded")
    out += ["", "## Open Items (discussed, not decided)"]
    out += [f"- {o.item}" + (f": {o.note}" if o.note else "")
            for o in r.open_items] or ["- None"]
    return "\n".join(out)
