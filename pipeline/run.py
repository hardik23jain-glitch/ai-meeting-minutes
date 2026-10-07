import json
import os
import tempfile

from pipeline.stt import transcribe, PipelineError
from pipeline.refine import refine
from pipeline.document import generate_record, to_markdown

try:
    from pipeline.stt import transcribe_detailed
except ImportError:
    transcribe_detailed = None


OUT_DIR = "outputs"
TEXT_REPLACEMENTS = str.maketrans(
    {
        "\u00a0": " ",
        "\u202f": " ",
        "\u2009": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }
)


def _plain_text(text):
    return str(text).translate(TEXT_REPLACEMENTS)


def _save(folder, name, text):
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_plain_text(text))
    return path


def _unpack_transcription(res):
    """Accept (text, dropped, stats), dict/object equivalents, or a plain str."""
    if isinstance(res, str):
        return res, [], {}
    if isinstance(res, dict):
        return (
            res.get("text", ""),
            res.get("dropped") or res.get("dropped_segments") or [],
            res.get("stats") or {},
        )
    if isinstance(res, (tuple, list)):
        items = list(res) + [None] * 3
        return items[0] or "", items[1] or [], items[2] or {}
    return (
        getattr(res, "text", "") or "",
        getattr(res, "dropped", None)
        or getattr(res, "dropped_segments", None)
        or [],
        getattr(res, "stats", None) or {},
    )


def _stt_markdown(stats, dropped):
    """Shape-agnostic summary of VAD / hallucination-filter activity."""
    lines = []
    if isinstance(stats, dict) and stats:
        lines += [f"- **{key}**: {value}" for key, value in stats.items()]
    lines.append(f"- **segments dropped by hallucination filter**: {len(dropped)}")
    for item in list(dropped)[:10]:
        summary = " ".join(str(item).split())
        lines.append(f"  - {summary[:160]}")
    if len(dropped) > 10:
        lines.append(f"  - ... and {len(dropped) - 10} more")
    return "\n".join(lines)


def _empty_state():
    return {
        "status": "",
        "raw": "",
        "refined": "",
        "stt_info": "",
        "diff_rows": [],
        "rejected_rows": [],
        "record_md": "",
        "record_json": "",
        "audit_rows": [],
        "files": {},
        "error": None,
    }


def run_pipeline_stream(audio_path, glossary_text=""):
    """Run all three stages in order and yield state after each step."""
    state = _empty_state()
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        run_dir = tempfile.mkdtemp(prefix="run_", dir=OUT_DIR)

        state["status"] = (
            "Step 1/3: Transcribing audio "
            "(VAD chunking + hallucination filter)..."
        )
        yield dict(state)
        if transcribe_detailed is not None:
            raw, dropped, stats = _unpack_transcription(transcribe_detailed(audio_path))
        else:
            raw, dropped, stats = transcribe(audio_path), [], {}
        state["raw"] = raw
        state["stt_info"] = _stt_markdown(stats, dropped)
        state["files"]["raw_transcript.txt"] = _save(
            run_dir, "raw_transcript.txt", raw
        )

        state["status"] = "Step 2/3: Refining technical terms..."
        yield dict(state)
        ref = refine(raw, glossary_text=glossary_text or "")
        state["refined"] = ref.text
        state["diff_rows"] = ref.rows
        state["rejected_rows"] = ref.rejected_rows
        state["files"]["refined_transcript.txt"] = _save(
            run_dir, "refined_transcript.txt", ref.text
        )

        state["status"] = (
            "Step 3/3: Writing minutes, then verifying "
            "(critic pass + chain-of-verification)..."
        )
        yield dict(state)
        doc = generate_record(ref.text)
        state["record_md"] = _plain_text(to_markdown(doc.record))
        state["record_json"] = _plain_text(json.dumps(
            doc.record.model_dump(), indent=2, ensure_ascii=False
        ))
        state["audit_rows"] = doc.audit_rows
        state["files"]["meeting_record.md"] = _save(
            run_dir, "meeting_record.md", state["record_md"]
        )
        state["files"]["meeting_record.json"] = _save(
            run_dir, "meeting_record.json", state["record_json"]
        )

        state["status"] = "Done. Review the tabs below and download your files."
        yield dict(state)

    except PipelineError as exc:
        state["error"] = str(exc)
        state["status"] = f"Error: {exc}"
        yield dict(state)
    except Exception as exc:
        state["error"] = str(exc)
        state["status"] = f"Error: Something went wrong: {exc}"
        yield dict(state)


def run_pipeline(audio_path, glossary_text=""):
    """Non-streaming version: return the final state."""
    final = None
    for final in run_pipeline_stream(audio_path, glossary_text):
        pass
    return final
