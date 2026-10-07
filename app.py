"""AI Meeting Assistant - Gradio front-end for pipeline/run.py.

Launch:
    python app.py
"""
import inspect
import os

import gradio as gr
from dotenv import load_dotenv

from pipeline.run import run_pipeline_stream

load_dotenv()


STATE_KEYS = {
    "status": ("status",),
    "error": ("error",),
    "raw": ("raw", "raw_text"),
    "refined": ("refined", "refined_text"),
    "diff": ("diff_rows", "refine_rows", "diff", "rows"),
    "rejected": ("rejected_rows", "rejected"),
    "record_md": ("record_md", "markdown"),
    "record_json": ("record_json", "json"),
    "audit": ("audit_rows", "audit", "verification_rows"),
    "files": ("files", "file_paths"),
}

ALLOWED_EXT = (".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".webm")
MAX_UPLOAD_MB = 500
OUTPUT_FILES = [
    ("raw_transcript.txt", "Download raw_transcript.txt"),
    ("refined_transcript.txt", "Download refined_transcript.txt"),
    ("meeting_record.md", "Download meeting_record.md"),
    ("meeting_record.json", "Download meeting_record.json"),
]
DIFF_HEADERS = ["Original", "Correction", "Status", "Detail"]
AUDIT_HEADERS = ["Check", "Item", "Outcome", "Detail"]

def _get(state, name, default=None):
    for key in STATE_KEYS[name]:
        if key in state and state[key] is not None:
            return state[key]
    return default


def _banner(msg, ok=False):
    if not msg:
        return ""
    if ok:
        style = (
            "background:#e8f5e9;border:1px solid #b7dfb9;color:#1b5e20;"
            "padding:12px 16px;border-radius:8px;font-weight:500"
        )
    else:
        style = (
            "background:#fdecea;border:1px solid #f5c2c0;color:#8a1c14;"
            "padding:12px 16px;border-radius:8px;font-weight:500"
        )
    safe = (
        str(msg)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f'<div style="{style}">{safe}</div>'


def _norm_diff(applied, rejected):
    """Normalize applied/rejected rows for the refinement log table."""
    rows = []
    for row in applied or []:
        if isinstance(row, dict):
            original = row.get("original", "")
            correction = row.get("correction", "")
            count = row.get("count", "")
            context = row.get("context", "")
        else:
            row = list(row) + [""] * 4
            original, correction, count, context = row[0], row[1], row[2], row[3]
        detail = (f"x{count}  " if count != "" else "") + str(context)
        rows.append([original, correction, "applied", detail.strip()])

    for row in rejected or []:
        if isinstance(row, dict):
            original = row.get("original", "")
            correction = row.get("correction", "")
            reason = row.get("reason", "")
        else:
            row = list(row) + [""] * 3
            original, correction, reason = row[0], row[1], row[2]
        rows.append([original, correction, "rejected", reason])
    return rows


def _norm_audit(rows):
    out = []
    for row in rows or []:
        if isinstance(row, dict):
            out.append(
                [
                    row.get("check") or row.get("stage") or row.get("type") or "",
                    row.get("item") or row.get("quote") or row.get("text") or "",
                    row.get("outcome") or row.get("action") or row.get("status") or "",
                    row.get("detail") or row.get("note") or row.get("reason") or "",
                ]
            )
        else:
            row = [str(x) for x in row] + [""] * 4
            out.append(row[:4])
    return out


def _download_buttons(paths):
    """Return one DownloadButton update per required output file."""
    if isinstance(paths, dict):
        by_name = paths
    else:
        by_name = {os.path.basename(path): path for path in (paths or []) if path}

    updates = []
    for filename, label in OUTPUT_FILES:
        path = by_name.get(filename)
        updates.append(
            gr.DownloadButton(label=label, value=path, interactive=bool(path))
        )
    return updates


def _validate_upload(path):
    """Fast checks so the user sees a clear banner immediately."""
    if not path:
        return "Please upload an audio file first."
    if not os.path.exists(path):
        return "The uploaded file could not be found. Please upload it again."
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXT:
        return (
            f"Unsupported file type '{ext or 'none'}'. "
            f"Please upload one of: {', '.join(ALLOWED_EXT)}."
        )
    size = os.path.getsize(path)
    if size == 0:
        return "The uploaded file is empty (0 bytes)."
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        return f"File exceeds the {MAX_UPLOAD_MB} MB upload limit."
    return None


def _call_pipeline(path, glossary_text):
    """Pass glossary text only if run_pipeline_stream accepts it."""
    params = inspect.signature(run_pipeline_stream).parameters
    if len(params) >= 2 or any(
        param.kind == param.VAR_POSITIONAL for param in params.values()
    ):
        return run_pipeline_stream(path, glossary_text or "")
    return run_pipeline_stream(path)


EMPTY_MD = "_The meeting record will appear here._"


def _idle(banner="", status="", **kw):
    return (
        banner,
        status,
        kw.get("raw", ""),
        kw.get("refined", ""),
        kw.get("diff", []),
        kw.get("md", EMPTY_MD),
        kw.get("json_text", ""),
        kw.get("audit", []),
        *_download_buttons(kw.get("files")),
    )


def process(audio_path, glossary_text):
    problem = _validate_upload(audio_path)
    if problem:
        yield _idle(_banner(problem), "Not started")
        return
    if not os.getenv("GROQ_API_KEY"):
        yield _idle(
            _banner(
                "GROQ_API_KEY is not set. Add it to your .env file and restart the app."
            ),
            "Not started",
        )
        return

    last = {}
    try:
        for state in _call_pipeline(audio_path, glossary_text):
            last = state
            err = _get(state, "error")
            yield _idle(
                _banner(err) if err else "",
                _get(state, "status", ""),
                raw=_get(state, "raw", ""),
                refined=_get(state, "refined", ""),
                diff=_norm_diff(_get(state, "diff", []), _get(state, "rejected", [])),
                md=_get(state, "record_md", "") or EMPTY_MD,
                json_text=_get(state, "record_json", ""),
                audit=_norm_audit(_get(state, "audit", [])),
                files=_get(state, "files", []),
            )
            if err:
                return
    except Exception as exc:
        yield _idle(
            _banner(f"Unexpected error: {exc}"),
            "Failed",
            raw=_get(last, "raw", ""),
            refined=_get(last, "refined", ""),
        )
        return

    if last and not _get(last, "error"):
        yield _idle(
            _banner("Processing complete. Review the tabs and download your files.", ok=True),
            _get(last, "status", "Done"),
            raw=_get(last, "raw", ""),
            refined=_get(last, "refined", ""),
            diff=_norm_diff(_get(last, "diff", []), _get(last, "rejected", [])),
            md=_get(last, "record_md", "") or EMPTY_MD,
            json_text=_get(last, "record_json", ""),
            audit=_norm_audit(_get(last, "audit", [])),
            files=_get(last, "files", []),
        )


with gr.Blocks(title="AI Meeting Assistant") as demo:
    gr.Markdown(
        "# AI Meeting Assistant\n"
        "Upload an English meeting recording. The app transcribes it "
        "(Whisper large-v3), corrects misheard technical terms with "
        "programmatic, auditable edits, and produces verified minutes, "
        "decisions and action items."
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.File(
                label=f"Meeting audio ({', '.join(ALLOWED_EXT)}; up to {MAX_UPLOAD_MB} MB)",
                file_types=list(ALLOWED_EXT),
                type="filepath",
            )
            glossary = gr.Textbox(
                label="Domain glossary (optional)",
                placeholder="Kubernetes, Redis, CI/CD, Terraform\n(comma or newline separated)",
                lines=4,
            )
            run_btn = gr.Button("Process recording", variant="primary")
        with gr.Column(scale=1):
            banner = gr.HTML()
            status = gr.Textbox(label="Pipeline status", interactive=False)

    with gr.Tabs():
        with gr.Tab("Transcript comparison"):
            with gr.Row():
                raw = gr.Textbox(
                    label="Raw transcript (Whisper)",
                    lines=18,
                    interactive=False,
                    buttons=["copy"],
                )
                refined = gr.Textbox(
                    label="Refined transcript",
                    lines=18,
                    interactive=False,
                    buttons=["copy"],
                )
        with gr.Tab("Refinement diff log"):
            gr.Markdown(
                "Every proposed edit, whether it was **applied** or "
                "**rejected** by the safety filters."
            )
            diff = gr.Dataframe(headers=DIFF_HEADERS, value=[], interactive=False, wrap=True)
        with gr.Tab("Meeting documentation"):
            record = gr.Markdown(EMPTY_MD)
        with gr.Tab("Structured JSON"):
            record_json = gr.Code(label="meeting_record.json", language="json")
        with gr.Tab("Verification audit log"):
            gr.Markdown(
                "Quote matching, critic adjustments, demoted proposals "
                "and owners forced to `unspecified`."
            )
            audit = gr.Dataframe(headers=AUDIT_HEADERS, value=[], interactive=False, wrap=True)

    gr.Markdown("### Downloads")
    with gr.Row():
        dl_buttons = [gr.DownloadButton(label, interactive=False) for _, label in OUTPUT_FILES]

    outputs = [
        banner,
        status,
        raw,
        refined,
        diff,
        record,
        record_json,
        audit,
        *dl_buttons,
    ]
    run_btn.click(process, inputs=[audio, glossary], outputs=outputs)

demo.queue()

if __name__ == "__main__":
    demo.launch(max_file_size=f"{MAX_UPLOAD_MB}mb")
