#!/usr/bin/env python3
"""
Take pictures from the Unitree video stream without ROS and without a display.

This follows the same WebRTC frame queue approach as Publisher.py:
- start Unitree WebRTC in a background asyncio thread
- receive camera frames in recv_camera_stream()
- keep only the newest frame in a Queue(maxsize=1)
- save one frame every N seconds

Examples:
  python Unitree/takePictures.py 5
  python Unitree/takePictures.py 2 --limit 10
  python Unitree/takePictures.py 5 --ip unitree.local
"""

import argparse
import asyncio
import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue

import cv2


BASE_DIR = Path(__file__).resolve().parent
PICTURES_DIR = BASE_DIR / "pictures"
PRINT_EVERY_N_FRAMES = 30

logging.basicConfig(level=logging.FATAL)


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def bounded_jpeg_quality(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 100:
        raise argparse.ArgumentTypeError("JPEG quality must be between 1 and 100")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Take a picture every N seconds from the Unitree video stream."
    )
    parser.add_argument(
        "interval_seconds",
        nargs="?",
        type=positive_float,
        help="Seconds between pictures.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=positive_float,
        help="Seconds between pictures. Overrides the positional value.",
    )
    parser.add_argument(
        "--ip",
        default="unitree.local",
        help="Unitree robot IP/host for LocalSTA mode. Default matches Publisher.py.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(PICTURES_DIR),
        help="Folder for captured pictures. Default: Unitree/pictures.",
    )
    parser.add_argument(
        "--prefix",
        default="picture",
        help="Filename prefix before the timestamp.",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=0,
        help="Stop after this many pictures. Default: 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=bounded_jpeg_quality,
        default=95,
        help="JPEG quality from 1 to 100.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=positive_float,
        default=30.0,
        help="Seconds to wait for the first frame before failing or trying fallback host.",
    )
    return parser


def queue_error(error_queue: Queue, exc: Exception) -> None:
    try:
        error_queue.put_nowait(exc)
    except Full:
        pass


def start_webrtc(frame_queue: Queue, error_queue: Queue, ip: str):
    """Start the Unitree WebRTC connection in a background asyncio thread."""

    try:
        from aiortc import MediaStreamTrack
        from unitree_webrtc_connect.webrtc_driver import (
            UnitreeWebRTCConnection,
            WebRTCConnectionMethod,
        )
    except ImportError as exc:
        print(
            "[ERROR] Missing Unitree WebRTC dependencies. Run this on the Jetson "
            "environment where Publisher.py works.",
            file=sys.stderr,
            flush=True,
        )
        print(f"[ERROR] Import error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ip)
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")

            # Drop stale frames so saving always uses the latest image.
            if not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
            frame_queue.put(img)

    async def setup():
        print(f"[INFO] Connecting to Unitree at {ip}", flush=True)
        await conn.connect()
        conn.video.switchVideoChannel(True)
        conn.video.add_track_callback(recv_camera_stream)

    def run_loop(loop):
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(setup())
            loop.run_forever()
        except Exception as exc:
            queue_error(error_queue, exc)

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=run_loop, args=(loop,), daemon=True)
    thread.start()
    return loop, thread


def stop_webrtc(loop, thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)


def wait_for_first_frame(frame_queue: Queue, error_queue: Queue, timeout: float):
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            return frame_queue.get(timeout=0.2), None
        except Empty:
            pass

        try:
            return None, error_queue.get_nowait()
        except Empty:
            pass

    return None, TimeoutError("No video frame received before startup timeout.")


def connection_hosts(primary_host: str) -> list[str]:
    hosts = [primary_host]
    if primary_host != "unitree.local":
        hosts.append("unitree.local")
    return hosts


def make_filename(output_dir: Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return output_dir / f"{prefix}_{timestamp}.jpg"


def save_picture(output_dir: Path, prefix: str, img, jpeg_quality: int) -> Path:
    out_path = make_filename(output_dir, prefix)
    ok = cv2.imwrite(
        str(out_path),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not ok:
        raise RuntimeError(f"Failed to save picture: {out_path}")
    return out_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    interval = args.interval if args.interval is not None else args.interval_seconds
    if interval is None:
        parser.error("interval is required, for example: python Unitree/takePictures.py 5")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_queue = None
    error_queue = None
    loop = None
    asyncio_thread = None
    first_img = None

    for host in connection_hosts(args.ip):
        frame_queue = Queue(maxsize=1)
        error_queue = Queue(maxsize=1)
        loop, asyncio_thread = start_webrtc(frame_queue, error_queue, host)
        first_img, connect_error = wait_for_first_frame(
            frame_queue,
            error_queue,
            timeout=args.startup_timeout,
        )

        if first_img is not None:
            break

        print(f"[ERROR] Connection attempt for {host} failed: {connect_error}", flush=True)
        stop_webrtc(loop, asyncio_thread)
        loop = None
        asyncio_thread = None

    if first_img is None or frame_queue is None or error_queue is None or loop is None:
        print(
            "[ERROR] Could not open the Unitree video stream. Make sure Publisher.py, "
            "viewVideoStream.py, or another WebRTC client is not still running.",
            flush=True,
        )
        return 1

    print(f"[INFO] Saving one picture every {interval:g}s to {output_dir}", flush=True)
    print("[INFO] Press Ctrl+C to stop.", flush=True)

    frame_id = 0
    saved = 0
    next_save_at = time.time()
    pending_img = first_img

    try:
        while True:
            try:
                stream_error = error_queue.get_nowait()
            except Empty:
                stream_error = None
            if stream_error is not None:
                print(f"[ERROR] Unitree video stream failed: {stream_error}", flush=True)
                return 1

            if pending_img is not None:
                img = pending_img
                pending_img = None
            else:
                if frame_queue.empty():
                    time.sleep(0.005)
                    continue
                img = frame_queue.get()

            if frame_id % PRINT_EVERY_N_FRAMES == 0:
                print(
                    f"[FRAME] id={frame_id} shape={img.shape} dtype={img.dtype}",
                    flush=True,
                )

            now = time.time()
            if now >= next_save_at:
                out_path = save_picture(output_dir, args.prefix, img, args.jpeg_quality)
                saved += 1
                print(f"[OK] Saved {out_path}", flush=True)

                if args.limit > 0 and saved >= args.limit:
                    break

                next_save_at = now + interval

            frame_id += 1

    except KeyboardInterrupt:
        print("[INFO] Stopping.", flush=True)
    finally:
        stop_webrtc(loop, asyncio_thread)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
