from pipeline.stt import transcribe, PipelineError

try:
    print(transcribe("test.mpeg"))
except PipelineError as e:
    print("ERROR:", e)