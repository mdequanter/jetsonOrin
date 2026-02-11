#!/usr/bin/env python3
# speech.py — Webcam mic (device 0) -> Dutch STT on Jetson (faster-whisper)
# Requirements: pip install faster-whisper sounddevice numpy

import sys, queue, time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# --- Audio settings (good defaults for USB webcam mics) ---
DEVICE_ID = 0          # HP FHD Webcam ... USB Audio (hw:0,0)
IN_SR = 44100          # many USB mics are 44100; change to 48000 if needed
TARGET_SR = 16000      # Whisper prefers 16 kHz
CHANNELS = 2           # your device reports "2 in"
BLOCK_SEC = 0.5        # audio chunk size from mic
BUFFER_SEC = 2.0       # how much audio to transcribe at once

q = queue.Queue()

def callback(indata, frames, t, status):
    if status:
        print(status, file=sys.stderr)
    # stereo -> mono
    mono = indata.mean(axis=1)
    q.put(mono.copy())

def simple_resample(x: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    """Fast-enough resampling via interpolation (ok for speech)."""
    if in_sr == out_sr:
        return x.astype(np.float32, copy=False)
    n_out = int(len(x) * out_sr / in_sr)
    xp = np.linspace(0.0, 1.0, len(x), endpoint=False)
    xnew = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(xnew, xp, x).astype(np.float32)

def main():
    # Optional: show devices once
    # print(sd.query_devices())

    model = WhisperModel("small", device="cuda", compute_type="float16")

    blocksize = int(IN_SR * BLOCK_SEC)
    buf = []

    print(f"Listening on device {DEVICE_ID} @ {IN_SR} Hz (Ctrl+C to stop)")
    with sd.InputStream(
        device=DEVICE_ID,
        channels=CHANNELS,
        samplerate=IN_SR,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    ):
        while True:
            chunk = q.get()
            buf.append(chunk)

            total_len = sum(len(b) for b in buf)
            if total_len >= int(IN_SR * BUFFER_SEC):
                audio = np.concatenate(buf)
                buf = []

                # quick sanity check (uncomment for debugging)
                # level = float(np.abs(audio).mean())
                # print("level:", level)

                audio_16k = simple_resample(audio, IN_SR, TARGET_SR)

                segments, info = model.transcribe(
                    audio_16k,
                    language="nl",
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 400},
                    beam_size=1,
                )

                text = "".join(seg.text for seg in segments).strip()
                if text:
                    print(text)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
