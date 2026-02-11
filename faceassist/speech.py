import queue, sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
BLOCK_SEC = 1.5
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

# 1) Zoek het device-id van je webcam-mic:
print(sd.query_devices())

MIC_DEVICE_ID = None  # zet dit op het juiste id uit query_devices()

model = WhisperModel("small", device="cuda", compute_type="float16")

with sd.InputStream(device=MIC_DEVICE_ID, samplerate=SAMPLE_RATE, channels=1,
                    dtype="float32", blocksize=int(SAMPLE_RATE*BLOCK_SEC),
                    callback=callback):
    print("Luisteren via webcam-microfoon... (Ctrl+C stop)")
    while True:
        audio = q.get().squeeze()
        segments, info = model.transcribe(audio, language="nl", vad_filter=True)
        text = "".join(seg.text for seg in segments).strip()
        if text:
            print(text)
