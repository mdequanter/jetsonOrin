#!/usr/bin/env python3
"""
Take pictures from the Unitree video stream without opening a display window.

Examples:
  python Unitree/takePictures.py 5
  python Unitree/takePictures.py 2 --ip 10.2.172.107 --limit 10
  python Unitree/takePictures.py --interval 30 --output-dir Unitree/pictures
"""

import argparse
import asyncio
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue

import cv2

try:
    from aiortc import MediaStreamTrack
    from unitree_webrtc_connect.webrtc_driver import (
        UnitreeWebRTCConnection,
        WebRTCConnectionMethod,
    )
except ImportError as exc:
    MediaStreamTrack = None
    UnitreeWebRTCConnection = None
    WebRTCConnectionMethod = None
    UNITREE_IMPORT_ERROR = exc
else:
    UNITREE_IMPORT_ERROR = None


logging.basicConfig(level=logging.FATAL)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "pictures"

stop_requested = False


def handle_stop_signal(signum, frame):
    global stop_requested
    stop_requested = True


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


def make_filename(output_dir: Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return output_dir / f"{prefix}_{timestamp}.jpg"


def put_latest_frame(frame_queue: Queue, img) -> None:
    try:
        frame_queue.put_nowait(img)
        return
    except Full:
        pass

    try:
        frame_queue.get_nowait()
    except Empty:
        pass

    try:
        frame_queue.put_nowait(img)
    except Full:
        pass


def get_latest_frame(frame_queue: Queue, timeout: float):
    try:
        img = frame_queue.get(timeout=timeout)
    except Empty:
        return None

    while True:
        try:
            img = frame_queue.get_nowait()
        except Empty:
            return img


def require_unitree_imports() -> None:
    if UNITREE_IMPORT_ERROR is None:
        return

    print(
        "[ERROR] Missing Unitree WebRTC dependencies. Install/run this on the Jetson "
        "environment that has unitree_webrtc_connect and aiortc available.",
        file=sys.stderr,
        flush=True,
    )
    print(f"[ERROR] Import error: {UNITREE_IMPORT_ERROR}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def build_connection(args):
    require_unitree_imports()

    method = getattr(WebRTCConnectionMethod, args.method)
    if args.method == "LocalSTA":
        if args.serial_number:
            return UnitreeWebRTCConnection(method, serialNumber=args.serial_number)
        return UnitreeWebRTCConnection(method, ip=args.ip)

    if args.method == "Remote":
        missing = [
            name
            for name, value in (
                ("--serial-number", args.serial_number),
                ("--username", args.username),
                ("--password", args.password),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"Remote connection requires: {', '.join(missing)}")
        return UnitreeWebRTCConnection(
            method,
            serialNumber=args.serial_number,
            username=args.username,
            password=args.password,
        )

    return UnitreeWebRTCConnection(method)


def run_asyncio_loop(loop, conn, frame_queue: Queue, stop_event: threading.Event, error_queue: Queue):
    asyncio.set_event_loop(loop)

    async def recv_camera_stream(track: MediaStreamTrack):
        while not stop_event.is_set():
            try:
                frame = await track.recv()
            except Exception as exc:
                if not stop_event.is_set():
                    try:
                        error_queue.put_nowait(exc)
                    except Full:
                        pass
                break

            img = frame.to_ndarray(format="bgr24")
            put_latest_frame(frame_queue, img)

    async def setup():
        await conn.connect()
        conn.video.switchVideoChannel(True)
        conn.video.add_track_callback(recv_camera_stream)

    try:
        loop.run_until_complete(setup())
        loop.run_forever()
    except Exception as exc:
        try:
            error_queue.put_nowait(exc)
        except Full:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Take a picture every N seconds from the Unitree video frame queue."
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
        default="10.2.172.107",
        help="Unitree robot IP for LocalSTA mode.",
    )
    parser.add_argument(
        "--method",
        choices=("LocalSTA", "LocalAP", "Remote"),
        default="LocalSTA",
        help="Unitree WebRTC connection method.",
    )
    parser.add_argument(
        "--serial-number",
        default="",
        help="Unitree serial number, used for LocalSTA serial or Remote mode.",
    )
    parser.add_argument("--username", default="", help="Remote mode username.")
    parser.add_argument("--password", default="", help="Remote mode password.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder for captured pictures. Default: Unitree/pictures.",
    )
    parser.add_argument("--prefix", default="picture", help="Filename prefix.")
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=0,
        help="Stop after this many pictures. Default: 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--frame-timeout",
        type=positive_float,
        default=10.0,
        help="Seconds to wait for a frame before warning and retrying.",
    )
    parser.add_argument(
        "--frame-queue-size",
        type=int,
        default=1,
        help="Maximum queued frames. Default keeps only the newest frame.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=bounded_jpeg_quality,
        default=95,
        help="JPEG quality from 1 to 100.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    interval = args.interval if args.interval is not None else args.interval_seconds
    if interval is None:
        parser.error("interval is required, for example: python Unitree/takePictures.py 5")

    signal.signal(signal.SIGINT, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        conn = build_connection(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1

    frame_queue = Queue(maxsize=max(1, args.frame_queue_size))
    error_queue = Queue(maxsize=1)
    stop_event = threading.Event()
    loop = asyncio.new_event_loop()

    asyncio_thread = threading.Thread(
        target=run_asyncio_loop,
        args=(loop, conn, frame_queue, stop_event, error_queue),
        daemon=True,
    )
    asyncio_thread.start()

    print(
        f"[INFO] Saving one picture every {interval:g}s to {output_dir}",
        flush=True,
    )

    saved = 0
    next_capture = time.monotonic()

    try:
        while not stop_requested:
            try:
                stream_error = error_queue.get_nowait()
            except Empty:
                stream_error = None
            if stream_error is not None:
                print(f"[ERROR] Unitree video stream failed: {stream_error}", file=sys.stderr, flush=True)
                return 1

            wait_time = next_capture - time.monotonic()
            if wait_time > 0:
                time.sleep(min(wait_time, 0.2))
                continue

            img = get_latest_frame(frame_queue, timeout=args.frame_timeout)
            if img is None:
                print("[WARN] No frame available in queue; retrying in 1 second.", flush=True)
                next_capture = time.monotonic() + 1.0
                continue

            out_path = make_filename(output_dir, args.prefix)
            ok = cv2.imwrite(
                str(out_path),
                img,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            if not ok:
                print(f"[ERROR] Failed to save picture: {out_path}", file=sys.stderr, flush=True)
                return 1

            saved += 1
            print(f"[OK] Saved {out_path}", flush=True)

            if args.limit > 0 and saved >= args.limit:
                break

            next_capture += interval
            if next_capture <= time.monotonic():
                next_capture = time.monotonic() + interval

        return 0
    finally:
        stop_event.set()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join(timeout=5.0)
        print("[INFO] Stopped Unitree frame capture.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
