"""AI Meeting Assistant - Gradio front-end for pipeline/run.py.

Launch:  python app.py
"""
import inspect
import os

import gradio as gr
from dotenv import load_dotenv

from pipeline.run import run_pipeline_stream

load_dotenv()

# ---------------------------------------------------------------------------
# Config. If run.py uses different state keys, change them here only.
# ---------------------------------------------------------------------------
STATE_KEYS = {
    "status": ("status",),
    "error": ("error",),
    "raw": ("raw", "raw_text"),
    "refined": ("refined", "refined_text"),
    "diff": ("diff_rows", "refine_rows", "diff", "rows"),
    "rejected": ("rejected_rows", "rejected"),
    "record_md": ("record_md", "markdown"),
    "audit": ("audit_rows", "audit", "verification_rows"),
    "files": ("files", "file_paths"),
}
ALLOWED_EXT = (".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".webm")
MAX_UPLOAD_MB = 500
OUTPUT_FILES = [
    ("raw_transcript.txt", "⬇ raw_transcript.txt"),
    ("refined_transcript.txt", "⬇ refined_transcript.txt"),
    ("meeting_record.md", "⬇ meeting_record.md"),
    ("meeting_record.json", "⬇ meeting_record.json"),
]
DIFF_HEADERS = ["Original", "Correction", "Status", "Detail"]
AUDIT_HEADERS = ["Stage", "Section", "Item", "Action", "Detail"]

CSS = """
.banner-error {background:#fdecea;border:1px solid #f5c2c0;color:#8a1c14;
  padding:12px 16px;border-radius:8px;font-weight:500}
.banner-ok {background:#e8f5e9;border:1px solid #b7dfb9;color:#1b5e20;
  padding:12px 16px;border-radius:8px;font-weight:500}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get(state, name, default=None):
    for k in STATE_KEYS[name]:
        if k in state and state[k] is not None:
            return state[k]
    return default


def _banner(msg, ok=False):
    if not msg:
        return ""
    cls = "banner-ok" if ok else "banner-error"
    safe = (str(msg).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))
    return f'<div class="{cls}">{safe}</div>'


def _norm_diff(applied, rejected):
    """Normalise applied/rejected rows into [Original, Correction, Status, Detail]."""
    rows = []
    for r in applied or []:
        if isinstance(r, dict):
            o, c = r.get("original", ""), r.get("correction", "")
            n, ctx = r.get("count", ""), r.get("context", "")
        else:
            r = list(r) + [""] * 4
            o, c, n, ctx = r[0], r[1], r[2], r[3]
        detail = (f"x{n}  " if n != "" else "") + str(ctx)
        rows.append([o, c, "✅ applied", detail.strip()])
    for r in rejected or []:
        if isinstance(r, dict):
            o, c, why = r.get("original", ""), r.get(
                "correction", ""), r.get("reason", "")
        else:
            r = list(r) + [""] * 3
            o, c, why = r[0], r[1], r[2]
        rows.append([o, c, "⛔ rejected", why])
    return rows


def _norm_audit(rows):
    out = []
    for r in rows or []:
        if isinstance(r, dict):
            out.append([
                r.get("stage") or r.get("check") or r.get("type") or "",
                r.get("section") or "",
                r.get("item") or r.get("quote") or r.get("text") or "",
                r.get("action") or r.get("outcome") or r.get("status") or "",
                r.get("detail") or r.get("note") or r.get("reason") or "",
            ])
        else:
            r = [str(x) for x in r] + [""] * 5
            out.append(r[:5])
    return out


def _download_buttons(paths):
    """Return one gr.DownloadButton update per required output file."""
    by_name = {os.path.basename(p): p for p in (paths or []) if p}
    updates = []
    for fname, label in OUTPUT_FILES:
        p = by_name.get(fname)
        updates.append(gr.DownloadButton(label=label, value=p,
                                         interactive=bool(p)))
    return updates


def _validate_upload(path):
    """Fast client-side checks so the user sees a clear banner immediately."""
    if not path:
        return "Please upload an audio file first."
    if not os.path.exists(path):
        return "The uploaded file could not be found. Please upload it again."
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXT:
        return (f"Unsupported file type '{ext or 'none'}'. "
                f"Please upload one of: {', '.join(ALLOWED_EXT)}.")
    size = os.path.getsize(path)
    if size == 0:
        return "The uploaded file is empty (0 bytes)."
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        return f"File exceeds the {MAX_UPLOAD_MB} MB upload limit."
    return None


def _call_pipeline(path, glossary_text):
    """Pass the glossary only if run_pipeline_stream accepts it."""
    params = inspect.signature(run_pipeline_stream).parameters
    if len(params) >= 2 or any(
            p.kind == p.VAR_POSITIONAL for p in params.values()):
        return run_pipeline_stream(path, glossary_text or "")
    return run_pipeline_stream(path)


# ---------------------------------------------------------------------------
# Main handler (generator -> streams to UI)
# ---------------------------------------------------------------------------
N_DL = len(OUTPUT_FILES)
EMPTY_MD = "_The meeting record will appear here._"


def _idle(banner="", status="", **kw):
    return (
        banner, status, kw.get("raw", ""), kw.get("refined", ""),
        kw.get("diff", []), kw.get("md", EMPTY_MD), kw.get("audit", []),
        *_download_buttons(kw.get("files")),
    )


def process(audio_path, glossary_text):
    problem = _validate_upload(audio_path)
    if problem:
        yield _idle(_banner(problem), "❌ Not started")
        return
    if not os.getenv("GROQ_API_KEY"):
        yield _idle(_banner("GROQ_API_KEY is not set. Add it to your .env "
                            "file and restart the app."), "❌ Not started")
        return

    last = {}
    try:
        for s in _call_pipeline(audio_path, glossary_text):
            last = s
            err = _get(s, "error")
            status = _get(s, "status", "")
            yield _idle(
                _banner(err) if err else "",
                status,
                raw=_get(s, "raw", ""),
                refined=_get(s, "refined", ""),
                diff=_norm_diff(_get(s, "diff", []), _get(s, "rejected", [])),
                md=_get(s, "record_md", "") or EMPTY_MD,
                audit=_norm_audit(_get(s, "audit", [])),
                files=_get(s, "files", []),
            )
            if err:
                return
    except Exception as e:  # last-resort guard so the UI never hangs
        yield _idle(_banner(f"Unexpected error: {e}"), "❌ Failed",
                    raw=_get(last, "raw", ""), refined=_get(last, "refined", ""))
        return

    if last and not _get(last, "error"):
        yield _idle(
            _banner("Processing complete. Review the tabs and download your files.",
                    ok=True),
            _get(last, "status", "✅ Done"),
            raw=_get(last, "raw", ""),
            refined=_get(last, "refined", ""),
            diff=_norm_diff(_get(last, "diff", []),
                            _get(last, "rejected", [])),
            md=_get(last, "record_md", "") or EMPTY_MD,
            audit=_norm_audit(_get(last, "audit", [])),
            files=_get(last, "files", []),
        )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="AI Meeting Assistant", css=CSS) as demo:
    gr.Markdown(
        "# 🎙️ AI Meeting Assistant\n"
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
                raw = gr.Textbox(label="Raw transcript (Whisper)", lines=18,
                                 interactive=False, buttons=["copy"])
                refined = gr.Textbox(label="Refined transcript", lines=18,
                                     interactive=False, buttons=["copy"])
        with gr.Tab("Refinement diff log"):
            gr.Markdown("Every proposed edit, whether it was **applied** or "
                        "**rejected** by the safety filters.")
            diff = gr.Dataframe(headers=DIFF_HEADERS, value=[],
                                interactive=False, wrap=True)
        with gr.Tab("Meeting documentation"):
            record = gr.Markdown(EMPTY_MD)
        with gr.Tab("Verification audit log"):
            gr.Markdown("Quote matching, critic adjustments, demoted proposals "
                        "and owners forced to `unspecified`.")
            audit = gr.Dataframe(headers=AUDIT_HEADERS, value=[],
                                 interactive=False, wrap=True)

    gr.Markdown("### Downloads")
    with gr.Row():
        dl_buttons = [gr.DownloadButton(label, interactive=False)
                      for _, label in OUTPUT_FILES]

    outputs = [banner, status, raw, refined, diff, record, audit, *dl_buttons]
    run_btn.click(process, inputs=[audio, glossary], outputs=outputs)

demo.queue()

if __name__ == "__main__":
    demo.launch(max_file_size=f"{MAX_UPLOAD_MB}mb")
