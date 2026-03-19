#!/usr/bin/env python3

import os
import subprocess
import shutil


KNOWN_DIR = "known"
OUTPUT_DIR = "speech_mp3"
PIPER_MODEL = "/home/jetson/jetsonOrin/voices/nl_BE-nathalie-medium.onnx"


def load_names(known_dir):
    names = []
    if not os.path.isdir(known_dir):
        return names

    for fn in os.listdir(known_dir):
        if fn.lower().endswith(".npz"):
            name = os.path.splitext(fn)[0]
            names.append(name)

    return sorted(names)


def sanitize_filename(text: str) -> str:
    text = text.strip().replace(" ", "_")
    allowed = "-_.()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c for c in text if c in allowed)


def generate_sentences(name):
    return [
        ("left", f"{name} staat links van je"),
        ("right", f"{name} staat rechts van je"),
        ("front", f"{name} staat voor je"),
    ]


def generate_wav_with_piper(text: str, wav_path: str, model_path: str):
    cmd = [
        "piper",
        "--model", model_path,
        "--output_file", wav_path,
    ]

    result = subprocess.run(
        cmd,
        input=text + "\n",
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Piper fout voor '{text}':\n{result.stderr}"
        )


def convert_wav_to_mp3(wav_path: str, mp3_path: str):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", wav_path,
        "-codec:a", "libmp3lame",
        "-qscale:a", "2",
        mp3_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg fout bij omzetting naar mp3:\n{result.stderr}"
        )


def main():
    if not os.path.exists(PIPER_MODEL):
        raise FileNotFoundError(f"Piper model niet gevonden: {PIPER_MODEL}")

    if shutil.which("piper") is None:
        raise RuntimeError("Het commando 'piper' is niet gevonden in PATH.")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Het commando 'ffmpeg' is niet gevonden in PATH.")

    names = load_names(KNOWN_DIR)
    if not names:
        print(f"[INFO] Geen .npz bestanden gevonden in '{KNOWN_DIR}'")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name in names:
        safe_name = sanitize_filename(name)
        for pos_key, sentence in generate_sentences(name):
            wav_path = os.path.join(OUTPUT_DIR, f"{safe_name}_{pos_key}.wav")
            mp3_path = os.path.join(OUTPUT_DIR, f"{safe_name}_{pos_key}.mp3")

            print(f"[INFO] Genereer WAV: {sentence}")
            generate_wav_with_piper(sentence, wav_path, PIPER_MODEL)

            print(f"[INFO] Converteer naar MP3: {mp3_path}")
            convert_wav_to_mp3(wav_path, mp3_path)

            os.remove(wav_path)

    print(f"[OK] Klaar. MP3-bestanden staan in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()