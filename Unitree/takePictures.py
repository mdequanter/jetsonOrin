#!/usr/bin/env python3
"""
Capture pictures at a fixed interval without opening a video window.

Examples:
  python Unitree/takePictures.py 5
  python Unitree/takePictures.py --interval 2 --source tcp://10.2.172.126:3000
  python Unitree/takePictures.py 10 --limit 12
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "pictures"


stop_requested = False


def handle_stop_signal(signum, frame):
    global stop_requested
    stop_requested = True


def parse_source(source: str):
    source = str(source).strip()
    try:
        return int(source)
    except ValueError:
        return source


def set_capture_properties(cap, width: int, height: int, fps: int) -> None:
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)


def open_capture(source_arg: str, width: int, height: int, fps: int, backend: str):
    source = parse_source(source_arg)

    if backend == "gstreamer":
        cap = cv2.VideoCapture(source_arg, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("[INFO] Camera opened via GStreamer.", flush=True)
            return cap
        return cap

    if isinstance(source, int):
        if backend == "auto":
            dev = f"/dev/video{source}"
            gst_pipeline = (
                f"v4l2src device={dev} ! "
                f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
                "jpegdec ! videoconvert ! appsink drop=true sync=false max-buffers=1"
            )
            cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print("[INFO] Camera opened via GStreamer V4L2 pipeline.", flush=True)
                return cap

        if backend in ("auto", "v4l2"):
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if cap.isOpened():
                set_capture_properties(cap, width, height, fps)
                print("[INFO] Camera opened via V4L2.", flush=True)
                return cap

        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            set_capture_properties(cap, width, height, fps)
            print("[INFO] Camera opened via default OpenCV backend.", flush=True)
        return cap

    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        set_capture_properties(cap, width, height, fps)
        print(f"[INFO] Camera opened from source: {source}", flush=True)
    return cap


def read_frame(cap, flush_frames: int):
    frame = None
    ok = False
    reads = max(1, flush_frames + 1)
    for _ in range(reads):
        ok, frame = cap.read()
        if not ok:
            return False, None
    return True, frame


def make_filename(output_dir: Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return output_dir / f"{prefix}_{timestamp}.jpg"


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
        description="Take a picture every N seconds and save it to Unitree/pictures."
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
        "-s",
        "--source",
        default="0",
        help="Camera index, stream URL, or GStreamer pipeline. Default: 0.",
    )
    parser.add_argument("--width", type=int, default=640, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=480, help="Requested frame height.")
    parser.add_argument("--fps", type=int, default=15, help="Requested camera FPS.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
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
        "--warmup-frames",
        type=int,
        default=5,
        help="Frames to discard after opening the camera.",
    )
    parser.add_argument(
        "--flush-frames",
        type=int,
        default=2,
        help="Extra frames to discard before saving each picture.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=bounded_jpeg_quality,
        default=95,
        help="JPEG quality from 1 to 100.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "v4l2", "default", "gstreamer"),
        default="auto",
        help="Capture backend. Use 'gstreamer' when --source is a pipeline string.",
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

    cap = open_capture(args.source, args.width, args.height, args.fps, args.backend)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera/source: {args.source}", file=sys.stderr, flush=True)
        return 1

    try:
        for _ in range(max(0, args.warmup_frames)):
            if stop_requested:
                return 0
            cap.read()

        print(
            f"[INFO] Saving one picture every {interval:g}s to {output_dir}",
            flush=True,
        )

        saved = 0
        next_capture = time.monotonic()

        while not stop_requested:
            wait_time = next_capture - time.monotonic()
            if wait_time > 0:
                time.sleep(min(wait_time, 0.2))
                continue

            ok, frame = read_frame(cap, max(0, args.flush_frames))
            if not ok:
                print("[WARN] Could not read frame; retrying in 1 second.", flush=True)
                next_capture = time.monotonic() + 1.0
                continue

            out_path = make_filename(output_dir, args.prefix)
            write_ok = cv2.imwrite(
                str(out_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            if not write_ok:
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
        cap.release()
        print("[INFO] Camera released.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
