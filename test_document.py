from pipeline.stt import transcribe, PipelineError
from pipeline.refine import refine
from pipeline.document import generate_record, to_markdown

try:
    raw = transcribe("test.mpeg")
    refined = refine(raw)
    record = generate_record(refined)
except PipelineError as e:
    print("ERROR:", e)
    raise SystemExit

print(to_markdown(record))
open("record.json", "w", encoding="utf-8").write(record.model_dump_json(indent=2))
print("\nSaved record.json")