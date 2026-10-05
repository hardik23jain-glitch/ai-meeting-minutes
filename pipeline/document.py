import os
import re
import json
from typing import List
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ValidationError
from pipeline.stt import PipelineError

load_dotenv()

DOC_MODEL = "openai/gpt-oss-120b"
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "document_prompt.txt")


class MinutesSection(BaseModel):
    topic: str
    points: List[str]


class Decision(BaseModel):
    decision: str
    source_quote: str = ""


class ActionItem(BaseModel):
    task: str
    owner: str = "unspecified"
    deadline: str = "unspecified"
    source_quote: str = ""


class OpenItem(BaseModel):
    item: str
    note: str = ""


class MeetingRecord(BaseModel):
    summary: str
    minutes: List[MinutesSection]
    decisions: List[Decision]
    action_items: List[ActionItem]
    open_items: List[OpenItem] = []


def _clean(record, transcript):
    """Safety net: never allow an owner that never appears in the transcript."""
    low = transcript.lower()
    for a in record.action_items:
        a.owner = (a.owner or "").strip() or "unspecified"
        a.deadline = (a.deadline or "").strip() or "unspecified"
        if a.owner.lower() != "unspecified":
            names = [t for t in re.split(r"\W+", a.owner.lower()) if len(t) > 2]
            if not any(n in low for n in names):
                a.owner = "unspecified"
    return record


def generate_record(refined_text):
    """Refined transcript -> MeetingRecord."""
    if not refined_text or not refined_text.strip():
        raise PipelineError("There is no transcript to document.")
    with open(PROMPT_PATH, encoding="utf-8") as f:
        system = f.read()
    client = Groq()
    last_err = None
    for _ in range(2):  # one retry on bad JSON
        try:
            resp = client.chat.completions.create(
                model=DOC_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": refined_text},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                max_completion_tokens=6000,
            )
            data = json.loads(resp.choices[0].message.content)
            return _clean(MeetingRecord.model_validate(data), refined_text)
        except (json.JSONDecodeError, ValidationError, TypeError) as e:
            last_err = e
        except Exception as e:
            raise PipelineError(f"Meeting documentation failed: {e}")
    raise PipelineError(f"The model returned an invalid record twice: {last_err}")


def to_markdown(r):
    """Human-readable version of the same record."""
    out = ["# Meeting Record", "", "## Summary", r.summary, "", "## Minutes"]
    for s in r.minutes:
        out.append(f"### {s.topic}")
        out += [f"- {p}" for p in s.points]
        out.append("")
    out.append("## Key Decisions")
    out += [f"- {d.decision}" for d in r.decisions] or ["- None recorded"]
    out += ["", "## Action Items"]
    for a in r.action_items:
        out.append(f"- {a.task} (Owner: {a.owner}; Deadline: {a.deadline})")
    if not r.action_items:
        out.append("- None recorded")
    out += ["", "## Open Items (discussed, not decided)"]
    out += [f"- {o.item}" + (f": {o.note}" if o.note else "") for o in r.open_items] or ["- None"]
    return "\n".join(out)