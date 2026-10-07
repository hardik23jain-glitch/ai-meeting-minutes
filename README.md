# 🎙️ AI-Powered Meeting Assistant

**Inter IIT Bootcamp Phase 2 — ML Problem Statement: "AI-Powered Meeting Assistant"**
IIT Guwahati Tech Board

An end-to-end pipeline that takes a meeting recording and produces a verified transcript, a domain-corrected transcript, structured minutes, key decisions and actionable tasks. The design goal is **hallucination resistance**: the language models are never trusted to rewrite or invent content. They *propose*, and deterministic code or an independent verification pass *disposes*.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture & Model Roles](#pipeline-architecture--model-roles)
3. [Key Technical Features & Academic Grounding](#key-technical-features--academic-grounding)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Running the Application & Usage Guide](#running-the-application--usage-guide)
7. [Testing & Verification](#testing--verification)
8. [Deliverables & Schema](#deliverables--schema)
9. [Limitations](#limitations)
10. [References](#references)

---

## Overview

Meeting-summarisation systems fail in predictable ways: speech-to-text hallucinates text over silence, LLM "clean-up" passes silently change numbers and negations, and summarisers turn tentative proposals into decisions and invent task owners. Each stage of this system is built around one of these failure modes.

```
 Audio (mp3/wav/m4a/mp4/mpeg/webm, up to 500 MB)
   │
   ▼
┌──────────────────────────────┐
│ Stage 1  Speech-to-Text      │  ffmpeg → 16 kHz mono PCM → WebRTC VAD
│ pipeline/stt.py              │  → speech chunks (≤300 s) → Whisper large-v3
│                              │  → hallucination filters
└──────────────┬───────────────┘
               │  raw transcript (+ dropped / flagged segment logs)
               ▼
┌──────────────────────────────┐
│ Stage 2  Refinement          │  Glossary retrieval → LLM *proposes* edits
│ pipeline/refine.py           │  → deterministic validation → regex apply
└──────────────┬───────────────┘
               │  refined transcript + diff log (applied / rejected)
               ▼
┌──────────────────────────────┐
│ Stage 3  Documentation       │  Topic segmentation → structured draft
│ pipeline/document.py         │  → critic pass → Chain-of-Verification
└──────────────┬───────────────┘
               ▼
 Minutes · Decisions · Action items · Open items · Verification audit log
```

`pipeline/run.py` orchestrates the three stages and streams progress; `app.py` is the Gradio front-end.

---

## Pipeline Architecture & Model Roles

| Stage | Model | Served via | Responsibility |
|---|---|---|---|
| 1. Speech-to-Text | `whisper-large-v3` | Groq API | Transcribes speech chunks to text and returns per-segment confidence diagnostics |
| 2. Refinement | `openai/gpt-oss-20b` | Groq API | Detects misrecognised technical terms and *proposes* `{original, correction}` pairs. It never regenerates the transcript. |
| 3. Documentation | `openai/gpt-oss-120b` | Groq API | Topic segmentation, structured minutes, critic pass and verification checks |

The smaller model handles the narrow, constrained detection task in Stage 2. The larger model is used where reasoning quality matters (Stage 3).

### Inter-stage data flow

| Hand-off | Payload | Validation before next stage |
|---|---|---|
| Stage 1 → 2 | Raw transcript string (plus dropped and flagged segment logs and audio statistics) | Rejects unreadable, empty or silent audio, and transcripts under 10 words |
| Stage 2 → 3 | Refined transcript + `RefineResult` (applied, rejected) | Numbers and negations must be identical before and after refinement, otherwise the refinement is reverted |
| Stage 3 → UI | Structured record + audit rows | Pydantic schema validation, with one retry on malformed JSON |

Failures raise a `PipelineError` carrying a user-readable message, which the UI shows as an error banner.

---

## Key Technical Features & Academic Grounding

### Stage 1: Audio → Raw transcript

- **Whisper large-v3** via Groq for strong zero-shot ASR across accents and domains (Radford et al., 2022), with `temperature=0`, the language pinned to English and a meeting-domain prompt.
- **Decode and downsample.** FFmpeg converts any supported container to raw **16 kHz, mono, 16-bit PCM** (32,000 bytes per second). 16 kHz gives an 8 kHz Nyquist limit, enough to preserve speech formants and fricatives, and it is the rate Whisper was trained on.
- **WhisperX-style VAD chunking** (Bain et al., Interspeech 2023). WebRTC voice-activity detection classifies 30 ms frames as speech or non-speech (aggressiveness 2). The pipeline then:
  1. turns consecutive speech frames into regions
  2. bridges pauses of up to 0.6 s
  3. drops blips shorter than 0.25 s
  4. pads each region by 0.2 s on both sides
  5. packs regions into chunks of at most **300 s** of speech (about 9.6 MB of WAV each, well under Groq's 25 MB limit regardless of the original upload size), with 0.2 s of silence inserted between regions
- **Graceful degradation.** If `webrtcvad` is unavailable, the stage falls back to fixed-length 300 s chunks without silence removal rather than crashing, and records `vad_used: false` in its statistics.
- **Careless Whisper hallucination filters** (Koenecke et al., FAccT 2024). The pipeline requests `verbose_json` and assesses every segment using Whisper's own diagnostics (`no_speech_prob`, `avg_logprob`, `compression_ratio`), plus repetition and stock-phrase checks. The first matching rule wins:

| # | Condition | Verdict |
|---|---|---|
| 1 | Empty text | drop |
| 2 | Same text as the previous kept segment for the third time in a row | drop |
| 3 | `compression_ratio > 2.4` (degenerate, repetitive decode) | drop |
| 4 | `no_speech_prob > 0.6` **and** `avg_logprob < −1.0` | drop |
| 5 | `avg_logprob < −1.5` | drop |
| 6 | Known stock phrase ("thanks for watching", "amara.org", …) with low confidence | drop |
| 7 | Same 1–4 word n-gram repeated back-to-back (4× for one word, 3× for longer) inside a segment | drop if low confidence, else flag |
| 8 | `no_speech_prob > 0.4` or `avg_logprob < −1.0` | flag (kept) |
| 9 | Otherwise | keep |

Dropped and flagged segments, with reasons, are returned by `transcribe_detailed` for auditing.

### Stage 2: Raw → Refined transcript

- **Programmatic edits, not regeneration** (RECOVER, arXiv 2603.16411). The LLM returns only a list of corrections. Pure Python validates each one and applies it with word-boundary regexes in a single, longest-match-first pass, so text that was not proposed for correction cannot change and no content drift is possible.
- **Deterministic safety filters.** An edit is rejected if it touches:
  - digits or number words (`touches_digits`, `touches_number_words`)
  - negation (`touches_negation`)
  - commitment words such as *will, should, might* (`touches_commitment_word`)
  - person names, found by heuristic detection (`touches_person_name`)
  - an overly long span (over 6 words), an ordinary word, or a span not found verbatim
- **Phonetic "sounds-like" gate.** A correction must sound like the original (consonant-skeleton similarity, spelled-letter handling such as "see eye see dee" → "CICD"), which blocks synonym swaps and meaning changes.
- **Glossary retrieval** (DARAG, Ghosh et al., ACL 2025). Phonetic and fuzzy matching of transcript n-grams against a user glossary. Only retrieved candidates are shown to the LLM.
- **Final integrity assertion.** If the numbers or negations in the refined text differ from the raw text, the whole refinement is reverted.
- **Transparent diff log.** `RefineResult` records every edit as *applied* (with count and context) or *rejected* (with reason).

### Stage 3: Refined transcript → Verified minutes

- **FRAME topic segmentation** (EMNLP 2025). The LLM inserts topic boundaries by sentence index rather than re-emitting the transcript, so segmentation cannot alter any words.
- **Critic pass** (Kirstein et al., COLING 2025). An automated audit checks the draft for:
  - tasks with no assigned person
  - proposals nobody agreed to, recorded as decisions
  - negation flips (for example, "we will not adopt X" recorded as "we will adopt X")

  Offending items are corrected or demoted to *open items*.
- **Chain-of-Verification** (Dhuliawala et al., ACL 2024):
  1. *Deterministic step:* every `source_quote` attached to a decision or action item is fuzzy-matched against the transcript. Items whose quote cannot be found are not trusted.
  2. *LLM step:* an isolated, independent check (it sees only the claim and the sentences around the quote, not the original draft's reasoning) verifies that the group *explicitly agreed* to each decision and that each task was *explicitly assigned*. Items failing the check are demoted.
- **Ground-truth adherence.** An owner or deadline that is missing from the transcript is set to `"unspecified"`, and an owner name that never appears in the transcript is forced to `"unspecified"`.
- **Multiple summary representations** (Asthana et al., CSCW 2024). The record carries both a one-sentence TL;DR and a detailed summary.
- **Audit log.** Quote matches, critic adjustments, demoted proposals and forced `unspecified` owners are all exposed in the UI.

### Evaluation

- **MESA-style ground-truth tests** (Kirstein et al., 2024) in `test_mesa.py` check that the three classic meeting-summary errors do not occur (see [Testing & Verification](#testing--verification)).

---

## Project Structure

```
.
├── app.py                  # Gradio web interface
├── test_mesa.py            # MESA-style automated evaluation
├── glossary.json           # Optional domain glossary (terms, aliases, protected names)
├── requirements.txt        # Python dependencies
├── .env                    # GROQ_API_KEY (create this; do not commit)
├── pipeline/
│   ├── __init__.py
│   ├── stt.py              # Stage 1: decode, VAD chunking, Whisper, hallucination filters
│   ├── refine.py           # Stage 2: RECOVER-style edits, DARAG retrieval, safety filters
│   ├── document.py         # Stage 3: FRAME, critic pass, Chain-of-Verification
│   └── run.py              # Orchestrator: streams progress, writes outputs
├── prompts/
│   └── document_prompt.txt # System prompt for the documentation LLM
└── outputs/
    └── run_xxxx/           # One folder per run
        ├── raw_transcript.txt
        ├── refined_transcript.txt
        ├── meeting_record.md
        └── meeting_record.json
```

---

## Installation & Setup

### Prerequisites

- Python 3.10 or newer
- A [Groq API key](https://console.groq.com/keys)
- FFmpeg. No manual install is needed: `imageio-ffmpeg` (in `requirements.txt`) bundles an FFmpeg binary, and a system-wide `ffmpeg` on your `PATH` is used first if present.

### Steps

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd <repository-folder>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure the API key
echo "GROQ_API_KEY=your_key_here" > .env
```

`requirements.txt`:

```
groq
gradio
python-dotenv
pydantic>=2
webrtcvad-wheels
imageio-ffmpeg
```

`webrtcvad-wheels` provides prebuilt wheels of the WebRTC voice-activity detector, so no C compiler is needed. If it fails to install, the pipeline still runs but without silence removal, which weakens the hallucination protection.

Optional environment variable:

| Variable | Default | Purpose |
|---|---|---|
| `GLOSSARY_PATH` | `glossary.json` | Location of the glossary file loaded by the refinement stage |

### Optional glossary file

`glossary.json` may be empty or absent. Three formats are accepted:

```json
["Kubernetes", "Redis", "Terraform"]
```

```json
{"Kubernetes": ["kuber natties", "cooper netties"], "Redis": ["red is"]}
```

```json
{
  "terms": [{"term": "CI/CD", "aliases": ["see eye see dee"]}, "Terraform"],
  "names": ["Priya", "Rahul"]
}
```

`names` are protected: the refinement stage will never edit them.

---

## Running the Application & Usage Guide

```bash
python app.py
```

Open the local URL printed in the terminal (by default `http://127.0.0.1:7860`).

1. **Upload audio.** Supported formats in the UI: `.mp3`, `.wav`, `.m4a`, `.mp4`, `.mpeg`, `.webm` (up to 500 MB). Invalid, empty or oversized files produce a red error banner immediately.
2. **(Optional) Add glossary terms.** Enter project-specific terms (for example product names or acronyms), comma- or newline-separated. They are merged with `glossary.json` for this run.
3. **Click "Process recording".** Step-by-step status is streamed live. If any stage fails, a clear error banner explains why.
4. **Review the results in the tabs:**

| Tab | What it shows |
|---|---|
| **Transcript comparison** | Raw (Whisper) and refined transcripts side by side |
| **Refinement diff log** | Table of `Original → Correction`, with status (✅ applied / ⛔ rejected) and count, context or rejection reason |
| **Meeting documentation** | TL;DR, summary, minutes by topic, key decisions, action items (owner and deadline), and open items |
| **Verification audit log** | Quote matches, critic adjustments, demoted proposals and owners forced to `unspecified` |

5. **Download.** Four one-click buttons at the bottom of the page provide every output file.

---

## Testing & Verification

```bash
python test_mesa.py
# or
pytest test_mesa.py -v
```

`test_mesa.py` needs no audio. It has two parts:

**Part A: offline** (no API key needed). Unit-tests the deterministic guards:
- an edit that turns "will not adopt" into "will adopt" is rejected
- edits touching commitment words or numbers are rejected
- the integrity signature detects a dropped negation
- a hallucinated owner name is forced to `"unspecified"`

**Part B: live** (requires `GROQ_API_KEY`; skipped otherwise). Runs the documentation stage on a scripted transcript and asserts that:

| MESA error class | Assertion |
|---|---|
| Proposed idea treated as decision | A tentatively proposed idea ("maybe try Redis… nobody has agreed") never appears under Key Decisions |
| Invented owner | A task spoken with no assignee ("someone needs to update the deployment documentation") keeps owner `"unspecified"` |
| Negation flip | "We will not adopt GraphQL" is never recorded as an affirmative adoption |

The script prints PASS/FAIL per check and exits with a non-zero status on any failure. Because Part B calls a stochastic LLM, a single pass is a smoke test rather than a statistical guarantee. Run it repeatedly to estimate reliability.

---

## Deliverables & Schema

Each run writes four files to `outputs/run_xxxx/`, all downloadable from the UI:

| File | Format | Contents |
|---|---|---|
| `raw_transcript.txt` | Plain text | Whisper output after VAD chunking and hallucination filtering |
| `refined_transcript.txt` | Plain text | Transcript after validated term corrections |
| `meeting_record.md` | Markdown | Human-readable record: TL;DR, summary, minutes, decisions, actions, open items |
| `meeting_record.json` | JSON | Machine-readable record following the schema below |

### `meeting_record.json` schema

```json
{
  "tldr": "One-sentence overview of the meeting.",
  "summary": "Three to five sentence detailed summary.",
  "minutes": [
    {
      "topic": "Topic name",
      "points": ["Point 1", "Point 2"]
    }
  ],
  "decisions": [
    {
      "decision": "What the group explicitly agreed.",
      "source_quote": "Verbatim transcript evidence."
    }
  ],
  "action_items": [
    {
      "task": "What must be done.",
      "owner": "Name as spoken in the transcript, or \"unspecified\"",
      "deadline": "Deadline as spoken in the transcript, or \"unspecified\"",
      "source_quote": "Verbatim transcript evidence."
    }
  ],
  "open_items": [
    {
      "item": "Topic discussed but not decided.",
      "note": "Why it is still open."
    }
  ]
}
```

Guarantees:
- `decisions` contains only items the group explicitly agreed on. Tentative proposals appear in `open_items`.
- `owner` and `deadline` are never inferred. If the transcript does not state them, they are `"unspecified"`.
- Every decision and action item is traceable to a transcript quote through `source_quote`.

---

## Limitations

- English-language recordings only (the STT stage is pinned to English).
- Speaker diarisation is not performed, so ownership is taken only from names spoken in the transcript.
- Person-name protection in refinement is heuristic (cue patterns and the optional `names` list in `glossary.json`), not a full NER model.
- Output quality depends on audio quality. Heavily overlapping speech or very noisy recordings reduce transcript accuracy.
- Uploads of up to 500 MB are accepted, but the whole recording is decoded into memory (about 115 MB of PCM per hour of audio), so multi-hour recordings need proportionally more RAM. Typical meetings of one to three hours are well within range.
- Segment timestamps in the dropped and flagged logs are relative to the concatenated speech chunk sent to Whisper, not to the original recording.

---

## References

- Radford et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper).
- Bain et al. (Interspeech 2023). *WhisperX: Time-Accurate Speech Transcription of Long-Form Audio.*
- Koenecke et al. (FAccT 2024). *Careless Whisper: Speech-to-Text Hallucination Harms.*
- *RECOVER*, arXiv 2603.16411. Programmatic LLM-proposed transcript edits.
- Ghosh et al. (ACL 2025). *DARAG: Retrieval-augmented ASR error correction.*
- *FRAME* (EMNLP 2025). Topic segmentation for meeting summarisation.
- Kirstein et al. (COLING 2025). Critic-based auditing of meeting summaries.
- Kirstein et al. (2024). *MESA* evaluation framework for meeting summarisation errors.
- Dhuliawala et al. (ACL 2024). *Chain-of-Verification Reduces Hallucination in Large Language Models.*
- Asthana et al. (CSCW 2024). Multiple summary representations for meeting needs.
- Xie et al. (2025). Inter-stage validation in LLM pipelines.

---

*Built for the Inter IIT Bootcamp Phase 2, ML Problem Statement, IIT Guwahati Tech Board.*