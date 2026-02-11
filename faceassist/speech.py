import sys, queue, time
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

DEVICE_ID = 0          # <- jouw webcam mic
IN_SR = 48000          # veel webcam mics leveren 48k
TARGET_SR = 16000
CHANNELS = 1
BLOCK_SEC = 0.5        # korte chunks
BUFFER_SEC = 2.0       # transcribe om de ~2s
q = queue.Queue()

def callback(indata, frames, t, status):
    if status:
        print(status, file=sys.stderr)
    # neem kanaal 0 (mono)
    q.put(indata[:, 0].copy())

def simple_resample(x, in_sr, out_sr):
    """Simpel (maar prima) resampling via interpolatie."""
    if in_sr == out_sr:
        return x.astype(np.float32, copy=False)
    n_out = int(len(x) * out_sr / in_sr)
    xp = np.linspace(0, 1, len(x), endpoint=False)
    fp = x
    xnew = np.linspace(0, 1, n_out, endpoint=False)
    y = np.interp(xnew, xp, fp).astype(np.float32)
    return y

def main():
    print("Beschikbare devices:")
    print(sd.query_devices())

    model = WhisperModel("small", device="cuda", compute_type="float16")

    blocksize = int(IN_SR * BLOCK_SEC)
    buf = []

    print(f"\nLuisteren via device {DEVICE_ID}... (Ctrl+C om te stoppen)")
    with sd.InputStream(
        device=DEVICE_ID,
        channels=CHANNELS,
        samplerate=IN_SR,
        dtype="float32",
        blocksize=blocksize,
        callback=callback
    ):
        last_print = time.time()
        while True:
            chunk = q.get()
            buf.append(chunk)

            # zodra buffer vol genoeg is: transcribe
            total_len = sum(len(b) for b in buf)
            if total_len >= int(IN_SR * BUFFER_SEC):
                audio = np.concatenate(buf)
                buf = []

                audio_16k = simple_resample(audio, IN_SR, TARGET_SR)

                segments, info = model.transcribe(
                    audio_16k,
                    language="nl",
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 400},
                    beam_size=1
                )
                text = "".join(seg.text for seg in segments).strip()
                if text:
                    print(text)

if __name__ == "__main__":
    main()
