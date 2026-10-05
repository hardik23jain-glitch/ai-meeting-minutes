import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

STT_MODEL = "whisper-large-v3"
ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg", ".flac"}
MAX_MB = 25  # safe limit for free tier


class PipelineError(Exception):
    """Errors that are safe to show to the user."""


def validate_audio(path):
    if not path or not os.path.exists(path):
        raise PipelineError("No audio file was provided.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXT:
        raise PipelineError(
            f"Unsupported file type '{ext}'. Please upload: "
            + ", ".join(sorted(ALLOWED_EXT))
        )
    size = os.path.getsize(path)
    if size == 0:
        raise PipelineError("The uploaded file is empty.")
    if size > MAX_MB * 1024 * 1024:
        raise PipelineError(f"File is larger than {MAX_MB} MB. Please upload a shorter or compressed recording.")


def transcribe(path):
    """Audio file -> raw transcript text."""
    validate_audio(path)
    client = Groq()
    try:
        with open(path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=(os.path.basename(path), f.read()),
                model=STT_MODEL,
                language="en",
                temperature=0.0,
            )
    except Exception as e:
        raise PipelineError(f"Could not transcribe this audio (it may be corrupt or unreadable): {e}")

    text = (result.text or "").strip()
    if not text:
        raise PipelineError("No speech was detected in this recording.")
    return text