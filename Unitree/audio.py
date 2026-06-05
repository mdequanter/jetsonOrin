#!/usr/bin/env python3

import logging
import asyncio
import os
import sys
import subprocess
import shutil
import wave

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from aiortc.contrib.media import MediaPlayer


# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "unitree.local")

PIPER_MODEL = "/home/jetson/jetsonOrin/voices/nl_BE-nathalie-medium.onnx"

OUTPUT_DIR = "speech_mp3"
WAV_FILE = "speech.wav"
MP3_FILE = "speech.mp3"


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

    # Other possible connection methods:
    # conn = UnitreeWebRTCConnection(
    #     WebRTCConnectionMethod.LocalSTA,
    #     serialNumber="B42D2000XXXXXXXX"
    # )
    # conn = UnitreeWebRTCConnection(
    #     WebRTCConnectionMethod.Remote,
    #     serialNumber="B42D2000XXXXXXXX",
    #     username="email@gmail.com",
    #     password="pass"
    # )
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    try:
        await conn.connect()

        print(f"[INFO] Sending MP3 to robot: {mp3_path}")

        player = MediaPlayer(mp3_path)
        audio_track = player.audio

        if audio_track is None:
            raise RuntimeError(f"No audio track found in file: {mp3_path}")

        conn.pc.addTrack(audio_track)

        print("[OK] Audio track added to WebRTC connection")
        print("[INFO] Speaking...")

        # Wait until speech is finished.
        # Extra margin avoids cutting off the last part.
        await asyncio.sleep(duration + 1.5)

        print("[OK] Speech finished")

    finally:
        try:
            await conn.pc.close()
        except Exception:
            pass


async def main():
    try:
        check_dependencies()

        while True:
            text = input("\nEnter text for the robot to say, or type 'q' to quit:\n> ").strip()

            if text.lower() in ["q", "quit", "exit", "stop"]:
                print("[INFO] Stopping.")
                break

            if not text:
                print("[INFO] Empty text ignored.")
                continue

            mp3_path, duration = create_speech_mp3(text)

            await send_mp3_to_robot(mp3_path, duration)

    except ValueError as e:
        logging.error(f"An error occurred: {e}")

    except FileNotFoundError as e:
        logging.error(e)

    except RuntimeError as e:
        logging.error(e)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)