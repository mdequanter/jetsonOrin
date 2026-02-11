import queue, sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
BLOCK_SEC = 2.0
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())

# Kies model: "small" of "medium" voor NL; "tiny" is sneller maar minder accuraat
model = WhisperModel("small", device="cuda", compute_type="float16")

print("Luisteren... (Ctrl+C om te stoppen)")
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    blocksize=int(SAMPLE_RATE * BLOCK_SEC), callback=callback):
    while True:
        audio = q.get()
        audio = audio.squeeze()
        # Whisper verwacht float32 array op 16kHz
        segments, info = model.transcribe(audio, language="nl", vad_filter=True)
        text = "".join([seg.text for seg in segments]).strip()
        if text:
            print(text)
