import json
import os
import tempfile

from pipeline.stt import transcribe, PipelineError
from pipeline.refine import refine
from pipeline.document import generate_record, to_markdown

try:
    from pipeline.stt import transcribe_detailed
except ImportError:          # old stt.py without WhisperX/Careless Whisper
    transcribe_detailed = None

OUT_DIR = "outputs"


def _save(folder, name, text):
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _unpack_transcription(res):
    """Accept (text, dropped, stats) as a tuple, dict or object, or a plain str."""
    if isinstance(res, str):
        return res, [], {}
    if isinstance(res, dict):
        return (res.get("text", ""),
                res.get("dropped") or res.get("dropped_segments") or [],
                res.get("stats") or {})
    if isinstance(res, (tuple, list)):
        t = list(res) + [None] * 3
        return t[0] or "", t[1] or [], t[2] or {}
    return (getattr(res, "text", "") or "",
            getattr(res, "dropped", None) or getattr(
                res, "dropped_segments", None) or [],
            getattr(res, "stats", None) or {})


def _stt_markdown(stats, dropped):
    """Shape-agnostic summary of VAD / hallucination-filter activity."""
    lines = []
    if isinstance(stats, dict) and stats:
        lines += [f"- **{k}**: {v}" for k, v in stats.items()]
    lines.append(
        f"- **segments dropped by hallucination filter**: {len(dropped)}")
    for d in list(dropped)[:10]:
        s = " ".join(str(d).split())
        lines.append(f"  - {s[:160]}")
    if len(dropped) > 10:
        lines.append(f"  - … and {len(dropped) - 10} more")
    return "\n".join(lines)


def _empty_state():
    return {"status": "", "raw": "", "refined": "", "stt_info": "",
            "diff_rows": [], "rejected_rows": [], "record_md": "",
            "record_json": "", "audit_rows": [], "files": {}, "error": None}


def run_pipeline_stream(audio_path, glossary_text=""):
    """Runs all 3 stages in order. Yields the full state after each step.

    state["files"] maps the download name -> path for the four deliverables:
    raw_transcript.txt, refined_transcript.txt, meeting_record.md,
    meeting_record.json (written into a per-run folder, so concurrent users
    never overwrite each other).
    """
    state = _empty_state()
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        run_dir = tempfile.mkdtemp(prefix="run_", dir=OUT_DIR)

        # ---- Stage 1: STT ------------------------------------------------
        state["status"] = "⏳ Step 1/3: Transcribing audio (VAD chunking + hallucination filter)..."
        yield dict(state)
        if transcribe_detailed is not None:
            raw, dropped, stats = _unpack_transcription(
                transcribe_detailed(audio_path))
        else:
            raw, dropped, stats = transcribe(audio_path), [], {}
        state["raw"] = raw
        state["stt_info"] = _stt_markdown(stats, dropped)
        state["files"]["raw_transcript.txt"] = _save(
            run_dir, "raw_transcript.txt", raw)

        # ---- Stage 2: Refinement ----------------------------------------
        state["status"] = "⏳ Step 2/3: Refining technical terms..."
        yield dict(state)
        ref = refine(raw, glossary_text=glossary_text or "")
        state["refined"] = ref.text
        state["diff_rows"] = ref.rows
        state["rejected_rows"] = ref.rejected_rows
        state["files"]["refined_transcript.txt"] = _save(
            run_dir, "refined_transcript.txt", ref.text)

        # ---- Stage 3: Documentation + verification ----------------------
        state["status"] = ("⏳ Step 3/3: Writing minutes, then verifying "
                           "(critic pass + chain-of-verification)...")
        yield dict(state)
        doc = generate_record(ref.text)
        state["record_md"] = to_markdown(doc.record)
        state["record_json"] = json.dumps(
            doc.record.model_dump(), indent=2, ensure_ascii=False)
        state["audit_rows"] = doc.audit_rows
        state["files"]["meeting_record.md"] = _save(
            run_dir, "meeting_record.md", state["record_md"])
        state["files"]["meeting_record.json"] = _save(
            run_dir, "meeting_record.json", state["record_json"])

        state["status"] = "✅ Done. Review the tabs below and download your files."
        yield dict(state)

    except PipelineError as e:
        state["error"] = str(e)
        state["status"] = f"❌ {e}"
        yield dict(state)
    except Exception as e:
        state["error"] = str(e)
        state["status"] = f"❌ Something went wrong: {e}"
        yield dict(state)


def run_pipeline(audio_path, glossary_text=""):
    """Non-streaming version: returns the final state."""
    final = None
    for final in run_pipeline_stream(audio_path, glossary_text):
        pass
    return final
