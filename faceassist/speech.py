#!/usr/bin/env python3
import sys
import time
import queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# -------- Audio device (jouw webcam mic) --------
DEVICE_ID = 0
CHANNELS = 2          # device zegt "2 in"
IN_SR = 44100         # probeer 44100; als het niet werkt: 48000
TARGET_SR = 16000

# -------- Streaming parameters --------
CHUNK_SEC = 0.25      # hoe vaak we audio binnenhalen
WINDOW_SEC = 2.0      # hoeveel audio we telkens transcriben
OVERLAP_SEC = 0.7     # overlap om woorden niet te knippen
MIN_PRINT_CHARS = 2

# -------- Model choice (CPU realtime) --------
# tiny = snelste (aanrader voor live)
# base = iets beter, iets trager
MODEL_SIZE = "tiny"   # zet op "base" als het snel genoeg blijft

q = queue.Queue()

def callback(indata, frames, t, status):
    if status:
        print(status, file=sys.stderr)
    mono = indata.mean(axis=1)  # stereo -> mono
    q.put(mono.copy())

def resample_linear(x: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr:
        return x.astype(np.float32, copy=False)
    n_out = int(len(x) * out_sr / in_sr)
    xp = np.linspace(0.0, 1.0, len(x), endpoint=False)
    xnew = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(xnew, xp, x).astype(np.float32)

def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def main():
    print("Starting live Dutch STT (Ctrl+C to stop)")
    print(f"Device={DEVICE_ID}, model={MODEL_SIZE}, IN_SR={IN_SR} -> {TARGET_SR}")

    model = WhisperModel(
        MODEL_SIZE,
        device="cpu",
        compute_type="int8"  # snel op CPU
    )

    chunk_size = int(IN_SR * CHUNK_SEC)
    window_len = int(IN_SR * WINDOW_SEC)
    overlap_len = int(IN_SR * OVERLAP_SEC)

    ring = np.zeros(window_len, dtype=np.float32)
    filled = 0

    last_full_text = ""

    with sd.InputStream(
        device=DEVICE_ID,
        channels=CHANNELS,
        samplerate=IN_SR,
        dtype="float32",
        blocksize=chunk_size,
        callback=callback
    ):
        while True:
            chunk = q.get()  # mono float32 op IN_SR

            # ring buffer update
            n = len(chunk)
            if n >= window_len:
                ring[:] = chunk[-window_len:]
                filled = window_len
            else:
                ring = np.roll(ring, -n)
                ring[-n:] = chunk
                filled = min(window_len, filled + n)

            # pas starten als we genoeg audio hebben
            if filled < window_len:
                continue

            # neem window + resample
            audio_16k = resample_linear(ring, IN_SR, TARGET_SR)

            segments, info = model.transcribe(
                audio_16k,
                language="nl",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                beam_size=1
            )
            full_text = " ".join(s.text.strip() for s in segments).strip()
            if not full_text:
                continue

            # print enkel wat nieuw is (t.o.v. vorige output)
            p = common_prefix_len(last_full_text, full_text)
            new = full_text[p:].strip()

            # soms “verschuift” de zin; als new leeg is, maar full_text anders is, herstart
            if len(new) < MIN_PRINT_CHARS and full_text != last_full_text:
                # kleine reset om niet te missen
                new = full_text

            if len(new) >= MIN_PRINT_CHARS:
                print(new)
                # ---- HIER is je "assistant hook" ----
                # bv. stuur 'new' naar je LLM, of trigger acties:
                # handle_text(new)

            last_full_text = full_text

            # overlap behouden: schuif "window" zodat we niet alles opnieuw horen
            # (we laten overlap staan en “vergeten” een stuk van het begin)
            # => simuleer door 'filled' wat terug te zetten
            filled = window_len - overlap_len

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
