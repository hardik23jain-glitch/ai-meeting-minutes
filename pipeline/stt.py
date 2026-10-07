"""Stage 1: Speech-to-text.

Design (see technical description for full citations):
- Radford et al. (2022): Whisper large-v3 (served by Groq) is the STT model.
- Bain et al. (2023), WhisperX: run Voice Activity Detection first, drop
  non-speech, and transcribe speech in bounded chunks. Here: WebRTC VAD ->
  16 kHz mono PCM -> speech regions merged into <=300 s chunks (~9.6 MB WAV,
  far below Groq's 25 MB limit regardless of the original upload size).
- Koenecke et al. (2024), Careless Whisper: Whisper hallucinates on silence.
  We request verbose_json and drop/flag segments using no_speech_prob,
  avg_logprob, compression_ratio, repeated n-grams and known stock phrases.
- Xie et al. (2025): explicit inter-stage validation.

Requires: ffmpeg (system binary, or `pip install imageio-ffmpeg`) and
`pip install webrtcvad-wheels`. If webrtcvad is missing we degrade gracefully
to fixed-length chunking (no silence removal) instead of crashing.
"""
import io
import logging
import os
import re
import shutil
import subprocess
import time
import wave

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
STT_MODEL = "whisper-large-v3"
ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".mp4",
               ".mpeg", ".mpga", ".webm", ".ogg", ".flac"}
MAX_UPLOAD_MB = 500          # we re-encode, so the raw upload may be large
MAX_CHUNK_MB = 24            # hard guard for what we send to Groq
MIN_WORDS = 10               # Xie et al. (2025)

SAMPLE_RATE = 16000
BYTES_PER_SEC = SAMPLE_RATE * 2           # 16-bit mono
FRAME_MS = 30                              # webrtcvad accepts 10/20/30 ms
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2
VAD_AGGRESSIVENESS = 2                     # 0 (lenient) .. 3 (strict)
MERGE_GAP_S = 0.6                          # bridge pauses shorter than this
MIN_SPEECH_S = 0.25                        # drop blips shorter than this
PAD_S = 0.2                                # context padding around speech
JOIN_GAP_S = 0.2                           # silence inserted between regions
MAX_CHUNK_S = 300                          # speech seconds per request

# Careless Whisper-style filters (thresholds follow Whisper's own defaults)
NO_SPEECH_DROP = 0.6        # with low logprob => silence hallucination
LOGPROB_SOFT = -1.0
LOGPROB_HARD = -1.5
COMPRESSION_DROP = 2.4
NO_SPEECH_FLAG = 0.4
REPEAT_STREAK_DROP = 3      # identical consecutive segments

STOCK_HALLUCINATIONS = (
    "thanks for watching", "thank you for watching", "please subscribe",
    "like and subscribe", "see you in the next video", "amara.org",
    "subtitles by", "thank you for listening",
)

DOMAIN_PROMPT = (
    "This is a professional work meeting. Participants discuss technical "
    "topics, project updates, decisions, and action items."
)


class PipelineError(Exception):
    """Errors that are safe to show to the user."""


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
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
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        raise PipelineError(f"File is larger than {MAX_UPLOAD_MB} MB.")


# --------------------------------------------------------------------------
# Audio: decode -> VAD -> chunks   (WhisperX-style preprocessing)
# --------------------------------------------------------------------------
def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        raise PipelineError(
            "ffmpeg is required for audio processing. Install ffmpeg or "
            "`pip install imageio-ffmpeg`.")


def _decode_to_pcm(path):
    """Any audio/video container -> raw 16 kHz mono 16-bit PCM bytes."""
    cmd = [_ffmpeg_exe(), "-v", "error", "-i", path, "-vn",
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "ignore").strip()[:200]
        raise PipelineError(
            f"Could not decode this audio (it may be corrupt): {err}")
    return proc.stdout


def _speech_regions(pcm):
    """Return [(start_s, end_s)] of speech, or None if VAD is unavailable."""
    try:
        import webrtcvad
    except ImportError:
        return None
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    n = len(pcm) // FRAME_BYTES
    frame_s = FRAME_MS / 1000.0

    runs, start = [], None
    for i in range(n):
        frame = pcm[i * FRAME_BYTES:(i + 1) * FRAME_BYTES]
        try:
            speech = vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            speech = False
        if speech and start is None:
            start = i
        elif not speech and start is not None:
            runs.append((start * frame_s, i * frame_s))
            start = None
    if start is not None:
        runs.append((start * frame_s, n * frame_s))

    # bridge short pauses, drop short blips, pad, re-merge overlaps
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= MERGE_GAP_S:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    total_s = len(pcm) / BYTES_PER_SEC
    padded = []
    for s, e in merged:
        if e - s < MIN_SPEECH_S:
            continue
        s, e = max(0.0, s - PAD_S), min(total_s, e + PAD_S)
        if padded and s <= padded[-1][1]:
            padded[-1] = (padded[-1][0], max(padded[-1][1], e))
        else:
            padded.append((s, e))
    return padded


def _build_chunks(regions):
    """Group speech regions into chunks of <= MAX_CHUNK_S seconds."""
    pieces = []
    for s, e in regions:
        while e - s > MAX_CHUNK_S:             # very long unbroken speech
            pieces.append((s, s + MAX_CHUNK_S))
            s += MAX_CHUNK_S
        pieces.append((s, e))
    chunks, cur, cur_len = [], [], 0.0
    for s, e in pieces:
        d = (e - s) + JOIN_GAP_S
        if cur and cur_len + d > MAX_CHUNK_S:
            chunks.append(cur)
            cur, cur_len = [], 0.0
        cur.append((s, e))
        cur_len += d
    if cur:
        chunks.append(cur)
    return chunks


def _chunk_to_wav(pcm, regions):
    gap = bytes(int(JOIN_GAP_S * BYTES_PER_SEC) // 2 * 2)
    parts = []
    for s, e in regions:
        a = int(s * SAMPLE_RATE) * 2
        b = int(e * SAMPLE_RATE) * 2
        parts.append(pcm[a:b])
        parts.append(gap)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"".join(parts))
    data = buf.getvalue()
    if len(data) > MAX_CHUNK_MB * 1024 * 1024:   # should be impossible
        raise PipelineError("Internal error: audio chunk exceeds size limit.")
    return data


# --------------------------------------------------------------------------
# Groq call
# --------------------------------------------------------------------------
def _call_whisper(client, wav_bytes, name):
    last = None
    for attempt in range(3):
        try:
            return client.audio.transcriptions.create(
                file=(name, wav_bytes),
                model=STT_MODEL,
                language="en",
                temperature=0.0,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                prompt=DOMAIN_PROMPT,
            )
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    raise PipelineError(
        f"Could not transcribe this audio (it may be corrupt or unreadable): {last}")


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    val = getattr(obj, key, None)
    if val is None:
        extra = getattr(obj, "model_extra", None) or {}
        val = extra.get(key)
    return default if val is None else val


# --------------------------------------------------------------------------
# Hallucination filtering  (Koenecke et al., 2024)
# --------------------------------------------------------------------------
def _norm(text):
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def has_repeated_ngram(text):
    """True if some 1-4 word n-gram repeats back-to-back (loop hallucination)."""
    w = re.findall(r"[a-z']+", text.lower())
    for n in (1, 2, 3, 4):
        need = 4 if n == 1 else 3
        for i in range(len(w) - n * need + 1):
            g = w[i:i + n]
            if all(w[i + k * n:i + (k + 1) * n] == g for k in range(1, need)):
                return True
    return False


def assess_segment(seg, prev_norm, streak):
    """Return (verdict, reason, new_streak); verdict in keep|flag|drop."""
    text = (_get(seg, "text", "") or "").strip()
    nsp = float(_get(seg, "no_speech_prob", 0.0) or 0.0)
    alp = float(_get(seg, "avg_logprob", 0.0) or 0.0)
    cr = float(_get(seg, "compression_ratio", 0.0) or 0.0)
    norm = _norm(text)

    if not norm:
        return "drop", "empty", streak
    new_streak = streak + 1 if norm == prev_norm else 0
    if new_streak >= REPEAT_STREAK_DROP - 1:
        return "drop", "repeated_segment", new_streak
    if cr > COMPRESSION_DROP:
        return "drop", f"compression_ratio={cr:.2f}", new_streak
    if nsp > NO_SPEECH_DROP and alp < LOGPROB_SOFT:
        return "drop", f"no_speech_prob={nsp:.2f},avg_logprob={alp:.2f}", new_streak
    if alp < LOGPROB_HARD:
        return "drop", f"avg_logprob={alp:.2f}", new_streak
    low_conf = nsp > 0.3 or alp < -0.8
    if any(p in norm for p in STOCK_HALLUCINATIONS) and low_conf:
        return "drop", "stock_phrase_low_confidence", new_streak
    if has_repeated_ngram(text):
        if low_conf:
            return "drop", "repeated_ngram_low_confidence", new_streak
        return "flag", "repeated_ngram", new_streak
    if nsp > NO_SPEECH_FLAG or alp < LOGPROB_SOFT:
        return "flag", f"low_confidence(nsp={nsp:.2f},alp={alp:.2f})", new_streak
    return "keep", "", new_streak


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def transcribe_detailed(path):
    """Audio file -> dict(text, dropped, flagged, stats)."""
    validate_audio(path)
    pcm = _decode_to_pcm(path)
    total_s = len(pcm) / BYTES_PER_SEC
    if total_s < 0.5:
        raise PipelineError("The recording is too short.")

    regions = _speech_regions(pcm)
    vad_used = regions is not None
    if regions is None:
        log.warning("webrtcvad not installed: skipping silence removal.")
        regions = [(0.0, total_s)]
    if not regions:
        raise PipelineError("No speech was detected in this recording.")

    chunks = _build_chunks(regions)
    client = Groq()

    texts, dropped, flagged = [], [], []
    prev_norm, streak, n_segments = "", 0, 0

    for idx, chunk_regions in enumerate(chunks):
        wav = _chunk_to_wav(pcm, chunk_regions)
        result = _call_whisper(client, wav, f"chunk_{idx}.wav")
        segments = _get(result, "segments", []) or []

        if not segments:   # no per-segment info: keep text unfiltered
            t = (_get(result, "text", "") or "").strip()
            if t:
                texts.append(t)
            continue

        for seg in segments:
            n_segments += 1
            verdict, reason, streak = assess_segment(seg, prev_norm, streak)
            text = (_get(seg, "text", "") or "").strip()
            info = {
                "chunk": idx,
                "start": round(float(_get(seg, "start", 0.0) or 0.0), 2),
                "end": round(float(_get(seg, "end", 0.0) or 0.0), 2),
                "text": text,
                "reason": reason,
            }
            if verdict == "drop":
                dropped.append(info)
                continue
            if verdict == "flag":
                flagged.append(info)
            texts.append(text)
            prev_norm = _norm(text)

    text = " ".join(t for t in texts if t).strip()
    if not text:
        raise PipelineError("No speech was detected in this recording.")
    if len(text.split()) < MIN_WORDS:
        raise PipelineError(
            f"Recording contains only {len(text.split())} word(s) of speech. "
            "Please upload a meeting recording with clear discussion.")

    return {
        "text": text,
        "dropped": dropped,
        "flagged": flagged,
        "stats": {
            "audio_seconds": round(total_s, 1),
            "speech_seconds": round(sum(e - s for s, e in regions), 1),
            "chunks": len(chunks),
            "segments_total": n_segments,
            "segments_dropped": len(dropped),
            "segments_flagged": len(flagged),
            "vad_used": vad_used,
        },
    }


def transcribe(path):
    """Audio file -> raw transcript text (backward-compatible entry point)."""
    return transcribe_detailed(path)["text"]
