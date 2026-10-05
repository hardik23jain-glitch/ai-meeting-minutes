import os
import json
from pipeline.stt import transcribe, PipelineError
from pipeline.refine import refine
from pipeline.document import generate_record, to_markdown

OUT_DIR = "outputs"


def _save(name, text):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def run_pipeline_stream(audio_path):
    """Runs all 3 stages in order. Yields the full state after each step."""
    state = {"status": "", "raw": "", "refined": "", "record_md": "",
             "record_json": "", "files": [], "error": None}
    try:
        os.makedirs(OUT_DIR, exist_ok=True)

        state["status"] = "⏳ Step 1/3: Transcribing audio..."
        yield dict(state)
        state["raw"] = transcribe(audio_path)
        state["files"].append(_save("raw_transcript.txt", state["raw"]))

        state["status"] = "⏳ Step 2/3: Refining technical terms..."
        yield dict(state)
        state["refined"] = refine(state["raw"])
        state["files"].append(_save("refined_transcript.txt", state["refined"]))

        state["status"] = "⏳ Step 3/3: Writing minutes, decisions and tasks..."
        yield dict(state)
        record = generate_record(state["refined"])
        state["record_md"] = to_markdown(record)
        state["record_json"] = json.dumps(record.model_dump(), indent=2, ensure_ascii=False)
        state["files"].append(_save("meeting_record.md", state["record_md"]))
        state["files"].append(_save("meeting_record.json", state["record_json"]))

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


def run_pipeline(audio_path):
    """Non-streaming version: returns the final state."""
    final = None
    for final in run_pipeline_stream(audio_path):
        pass
    return final