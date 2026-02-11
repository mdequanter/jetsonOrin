#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from faster_whisper import WhisperModel

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\n{p.stderr}")
    return p.stdout

def extract_audio_to_wav(input_path: str, wav_path: str, sr: int = 16000):
    # -vn: no video, 1 channel, 16kHz, pcm_s16le wav
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        wav_path
    ]
    # ffmpeg writes progress to stderr; we don't need it
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed extracting audio. Check your input file.")

def main():
    ap = argparse.ArgumentParser(description="Fast offline Dutch transcription on Jetson (CPU int8).")
    ap.add_argument("input", help="Path to video/audio file (e.g., input.mp4)")
    ap.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large-v3"],
                    help="Whisper model size (bigger = better, slower).")
    ap.add_argument("--lang", default="nl", help="Language code (default: nl)")
    ap.add_argument("--sr", type=int, default=16000, help="Audio sample rate for extraction (default: 16000)")
    ap.add_argument("--out", default=None, help="Output text file (default: input.txt)")
    ap.add_argument("--keep-wav", action="store_true", help="Keep extracted WAV")
    args = ap.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    base, _ = os.path.splitext(input_path)
    wav_path = base + f"_{args.sr}hz_mono.wav"
    out_path = args.out or (base + ".txt")

    print(f"[1/2] Extracting audio -> {wav_path}")
    extract_audio_to_wav(input_path, wav_path, sr=args.sr)

    print(f"[2/2] Transcribing (model={args.model}, CPU int8, lang={args.lang})")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        wav_path,
        language=args.lang,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        beam_size=1,  # fastest
    )

    lines = []
    for s in segments:
        # timestamps
        lines.append(f"[{s.start:7.2f} - {s.end:7.2f}] {s.text.strip()}")

    text = "\n".join(lines).strip()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"Saved transcript -> {out_path}")

    if not args.keep_wav:
        try:
            os.remove(wav_path)
        except OSError:
            pass

if __name__ == "__main__":
    main()
