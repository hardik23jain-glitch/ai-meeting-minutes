import gradio as gr
from pipeline.run import run_pipeline_stream


def process(audio_path):
    if not audio_path:
        yield "❌ Please upload an audio file first.", "", "", "", None
        return
    for s in run_pipeline_stream(audio_path):
        yield (
            s["status"],
            s["raw"],
            s["refined"],
            s["record_md"] or "_The meeting record will appear here._",
            s["files"] if s["files"] else None,
        )


with gr.Blocks(title="AI Meeting Assistant") as demo:
    gr.Markdown(
        "# 🎙️ AI Meeting Assistant\n"
        "Upload an English meeting recording. The app transcribes it, refines "
        "technical terms, and writes minutes, decisions and action items."
    )
    audio = gr.File(label="Upload meeting audio (mp3, wav, m4a, mpeg, ogg, flac, webm)")
    run_btn = gr.Button("Process recording", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)

    with gr.Tabs():
        with gr.Tab("Raw transcript"):
            raw = gr.Textbox(lines=14, label="Speech-to-text output")
        with gr.Tab("Refined transcript"):
            refined = gr.Textbox(lines=14, label="After term correction")
        with gr.Tab("Meeting record"):
            record = gr.Markdown()
    files = gr.File(label="Downloads (transcripts, Markdown and JSON record)", file_count="multiple")

    run_btn.click(process, inputs=audio, outputs=[status, raw, refined, record, files])

if __name__ == "__main__":
    demo.launch()