import re
import difflib
from pipeline.stt import transcribe, PipelineError
from pipeline.refine import refine

try:
    raw = transcribe("test.mpeg")
    refined = refine(raw)
except PipelineError as e:
    print("ERROR:", e)
    raise SystemExit

print("REFINED:\n", refined, "\n")
print("CHANGES:")
a, b = raw.split(), refined.split()
norm = lambda ws: [re.sub(r"\W", "", w.lower()) for w in ws]
sm = difflib.SequenceMatcher(None, norm(a), norm(b))
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag != "equal":
        print(" ", " ".join(a[i1:i2]), "->", " ".join(b[j1:j2]))