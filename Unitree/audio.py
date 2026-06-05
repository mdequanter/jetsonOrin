#!/usr/bin/env python3

import logging
import asyncio
import os
import sys
import subprocess
import shutil
import wave
import argparse

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from aiortc.contrib.media import MediaPlayer


logging.basicConfig(level=logging.FATAL)

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "unitree.local")

PIPER_MODEL = "/home/jetson/jetsonOrin/voices/nl_BE-nathalie-medium.onnx"

OUTPUT_DIR = "speech_mp3"
WAV_FILE = "speech.wav"
MP3_FILE = "speech.mp3"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Laat de Unitree-robot een tekst uitspreken en sluit daarna af."
    )

    parser.add_argument(
        "-t",
        "--text",
        required=True,
        help="De tekst die de robot moet uitspreken."
    )

    return parser.parse_args()


def check_dependencies():
    if not os.path.exists(PIPER_MODEL):
        raise FileNotFoundError(f"Piper model not found: {PIPER_MODEL}")

    if shutil.which("piper") is None:
        raise RuntimeError("The command 'piper' was not found in PATH.")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("The command 'ffmpeg' was not found in PATH.")


def get_wav_duration(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        return frames / float(rate)


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
            f"Piper error for text '{text}':\n{result.stderr}"
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

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg error while converting to MP3:\n{result.stderr}"
        )


def create_speech_mp3(text: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wav_path = os.path.abspath(os.path.join(OUTPUT_DIR, WAV_FILE))
    mp3_path = os.path.abspath(os.path.join(OUTPUT_DIR, MP3_FILE))

    print(f"[INFO] Generating speech WAV: {wav_path}")
    generate_wav_with_piper(text, wav_path, PIPER_MODEL)

    duration = get_wav_duration(wav_path)

    print(f"[INFO] Creating MP3: {mp3_path}")
    convert_wav_to_mp3(wav_path, mp3_path)

    if os.path.exists(wav_path):
        os.remove(wav_path)

    print(f"[OK] Speech MP3 created: {mp3_path}")
    print(f"[INFO] Duration: {duration:.2f} seconds")

    return mp3_path, duration


async def send_mp3_to_robot(mp3_path: str, duration: float):
    if not os.path.isfile(mp3_path):
        raise FileNotFoundError(f"MP3 file not found: {mp3_path}")

    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=ROBOT_IP
    )

    print("[INFO] Connecting to robot...")
    await conn.connect()
    print("[OK] Connected to robot")

    print(f"[INFO] Sending MP3 to robot: {mp3_path}")

    player = MediaPlayer(mp3_path)
    audio_track = player.audio

    if audio_track is None:
        raise RuntimeError(f"No audio track found in file: {mp3_path}")

    conn.pc.addTrack(audio_track)

    print("[OK] Audio track added to WebRTC connection")
    print("[INFO] Speaking...")

    # Wacht tot de zin uitgesproken is.
    # De extra marge voorkomt dat de laatste woorden worden afgekapt.
    await asyncio.sleep(duration + 3.0)

    print("[OK] Speech finished")

    # Belangrijk:
    # Geen conn.pc.close()
    # Geen audio_track.stop()
    # Geen player.stop()
    #
    # Dit bootst het gedrag na van je werkende script,
    # waarbij het proces gewoon wordt gestopt na gebruik.


async def main():
    args = parse_args()
    text = args.text.strip()

    if not text:
        print("[ERROR] Empty text.")
        return 1

    check_dependencies()

    mp3_path, duration = create_speech_mp3(text)
    await send_mp3_to_robot(mp3_path, duration)

    print("[OK] Done.")
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())

        # Hard afsluiten zodat WebRTC/aiortc geen half-gesloten toestand achterlaat
        # in de Unitree AudioHub.
        os._exit(exit_code)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        os._exit(0)

    except Exception as e:
        print(f"[ERROR] {e}")
        os._exit(1)