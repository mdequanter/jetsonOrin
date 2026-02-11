#!/usr/bin/env python3
import sys, queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

DEVICE_ID = 0
CHANNELS = 2
TARGET_SR = 16000

CHUNK_SEC = 0.25
WINDOW_SEC = 2.0
OVERLAP_SEC = 0.7

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

def pick_input_samplerate(device_id: int, channels: int) -> int:
    candidates = [16000, 48000, 44100, 32000, 24000, 8000]
    for sr in candidates:
        try:
            sd.check_input_settings(device=device_id, channels=channels, samplerate=sr)
            return sr
        except Exception:
            pass
    raise RuntimeError(
        f"No supported samplerate found for device {device_id}. "
        f"Tried {candidates}. Check ALSA/Pulse device."
    )

def main():
    in_sr = pick_input_samplerate(DEVICE_ID, CHANNELS)
    print("Starting live Dutch STT (Ctrl+C to stop)")
    print(f"Device={DEVICE_ID}, model=tiny, IN_SR={in_sr} -> {TARGET_SR}")

    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    chunk_size = int(in_sr * CHUNK_SEC)
    window_len = int(in_sr * WINDOW_SEC)
    overlap_len = int(in_sr * OVERLAP_SEC)

    ring = np.zeros(window_len, dtype=np.float32)
    filled = 0
    last_full_text = ""

    with sd.InputStream(
        device=DEVICE_ID,
        channels=CHANNELS,
        samplerate=in_sr,
        dtype="float32",
        blocksize=chunk_size,
        callback=callback
    ):
        while True:
            chunk = q.get()

            n = len(chunk)
            if n >= window_len:
                ring[:] = chunk[-window_len:]
                filled = window_len
            else:
                ring = np.roll(ring, -n)
                ring[-n:] = chunk
                filled = min(window_len, filled + n)

            if filled < window_len:
                continue

            audio_16k = resample_linear(ring, in_sr, TARGET_SR)

            segments, _info = model.transcribe(
                audio_16k,
                language="nl",
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                beam_size=1
            )

            full_text = " ".join(s.text.strip() for s in segments).strip()
            if not full_text:
                continue

            p = common_prefix_len(last_full_text, full_text)
            new = full_text[p:].strip()

            if len(new) >= 2:
                print(new)

            last_full_text = full_text
            filled = window_len - overlap_len

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
