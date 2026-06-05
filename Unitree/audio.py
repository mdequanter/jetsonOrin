#!/usr/bin/env python3

import logging
import asyncio
import os
import sys
import subprocess
import shutil

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


def check_dependencies():
    if not os.path.exists(PIPER_MODEL):
        raise FileNotFoundError(f"Piper model not found: {PIPER_MODEL}")

    if shutil.which("piper") is None:
        raise RuntimeError("The command 'piper' was not found in PATH.")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("The command 'ffmpeg' was not found in PATH.")


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


def create_speech_mp3(text: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wav_path = os.path.abspath(os.path.join(OUTPUT_DIR, WAV_FILE))
    mp3_path = os.path.abspath(os.path.join(OUTPUT_DIR, MP3_FILE))

    print(f"[INFO] Generating WAV: {wav_path}")
    generate_wav_with_piper(text, wav_path, PIPER_MODEL)

    print(f"[INFO] Creating MP3: {mp3_path}")
    convert_wav_to_mp3(wav_path, mp3_path)

    if os.path.exists(wav_path):
        os.remove(wav_path)

    return mp3_path


async def connect_to_robot():
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

    print("[INFO] Connecting to robot...")
    await conn.connect()
    print("[OK] Connected to robot")

    return conn


async def play_mp3_on_robot(conn, mp3_path: str):
    if not os.path.isfile(mp3_path):
        raise FileNotFoundError(f"MP3 file not found: {mp3_path}")

    print(f"[INFO] Playing MP3 on robot: {mp3_path}")

    player = MediaPlayer(mp3_path)
    audio_track = player.audio

    if audio_track is None:
        raise RuntimeError(f"No audio track found in file: {mp3_path}")

    conn.pc.addTrack(audio_track)

    print("[OK] Audio track added")

    # Wait until the MP3 has finished playing.
    # This is an approximation. Increase if long texts are cut off.
    await asyncio.sleep(5)


async def input_loop(conn):
    while True:
        text = input("\nEnter text for the robot to say, or type 'q' to quit:\n> ").strip()

        if text.lower() in ["q", "quit", "exit"]:
            print("[INFO] Exiting.")
            break

        if not text:
            print("[INFO] Empty text ignored.")
            continue

        try:
            mp3_path = create_speech_mp3(text)
            await play_mp3_on_robot(conn, mp3_path)

        except Exception as e:
            print(f"[ERROR] {e}")


async def main():
    check_dependencies()

    conn = await connect_to_robot()

    await input_loop(conn)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(0)