import os
from dotenv import load_dotenv
from groq import Groq
from pipeline.stt import PipelineError

load_dotenv()

REFINE_MODEL = "openai/gpt-oss-20b"
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "refine_prompt.txt")
CHUNK_WORDS = 1200


def _load_prompt():
    with open(PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def _chunks(text, size=CHUNK_WORDS):
    words = text.split()
    for i in range(0, len(words), size):
        yield " ".join(words[i:i + size])


def _refine_chunk(client, system_prompt, chunk):
    resp = client.chat.completions.create(
        model=REFINE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk},
        ],
        temperature=0,
        extra_body={"reasoning_effort": "low"},
    )
    out = (resp.choices[0].message.content or "").strip()
    n_in, n_out = len(chunk.split()), len(out.split())
    # Safety net: if the model summarized or rambled, keep the raw chunk
    if n_out < 0.7 * n_in or n_out > 1.3 * n_in:
        return chunk
    return out


def refine(raw_text):
    """Raw transcript -> refined transcript."""
    if not raw_text or not raw_text.strip():
        raise PipelineError("There is no transcript to refine.")
    try:
        client = Groq()
        prompt = _load_prompt()
        return " ".join(_refine_chunk(client, prompt, c) for c in _chunks(raw_text))
    except Exception as e:
        raise PipelineError(f"Transcript refinement failed: {e}")