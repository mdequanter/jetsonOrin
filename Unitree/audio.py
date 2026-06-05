import logging
import asyncio
import os
import sys
import argparse

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from aiortc.contrib.media import MediaPlayer


# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

ROBOT_IP = os.environ.get("UNITREE_ROBOT_IP", "unitree.local")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play an MP3 file through the Unitree WebRTC audio connection."
    )

    parser.add_argument(
        "--mp3",
        required=True,
        help="Path to the MP3 file that should be played."
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    mp3_path = os.path.abspath(args.mp3)

    if not os.path.isfile(mp3_path):
        raise FileNotFoundError(f"MP3 file not found: {mp3_path}")

    try:
        # Choose a connection method
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

        await conn.connect()

        logging.info(f"Playing MP3: {mp3_path}")

        player = MediaPlayer(mp3_path)
        audio_track = player.audio

        if audio_track is None:
            raise RuntimeError(f"No audio track found in file: {mp3_path}")

        conn.pc.addTrack(audio_track)

        await asyncio.sleep(3600)

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