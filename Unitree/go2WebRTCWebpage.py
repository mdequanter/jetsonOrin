#!/usr/bin/env python3
"""
Show the Unitree Go2 WebRTC camera stream on a local webpage.

Examples:
  python Unitree/go2WebRTCWebpage.py
  python Unitree/go2WebRTCWebpage.py --ip 192.168.123.18 --host 0.0.0.0 --port 8080

Open the printed URL in a browser. The page displays an MJPEG stream generated
from the Go2 WebRTC video frames.
"""

import argparse
import asyncio
import logging
import sys
import threading
import time
from queue import Empty, Full, Queue

import cv2
from flask import Flask, Response, render_template_string, stream_with_context


logging.basicConfig(level=logging.FATAL)

DEFAULT_ROBOT_HOST = "unitree.local"
DEFAULT_WEB_HOST = "0.0.0.0"
DEFAULT_WEB_PORT = 8080

HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Go2 Camera Stream</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Arial, Helvetica, sans-serif;
      background: #101418;
      color: #f4f7fb;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 18px;
      background: #1b232d;
      border-bottom: 1px solid #303b47;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }
    .status {
      font-size: 14px;
      color: #a9b8c8;
    }
    main {
      min-height: 0;
      display: grid;
      place-items: center;
      padding: 16px;
    }
    .stream {
      width: min(100%, 1280px);
      max-height: calc(100vh - 90px);
      object-fit: contain;
      background: #050607;
      border: 1px solid #303b47;
    }
  </style>
</head>
<body>
  <header>
    <h1>Go2 Camera Stream</h1>
    <div class="status">Robot: {{ robot_host }}</div>
  </header>
  <main>
    <img class="stream" src="{{ stream_url }}" alt="Live Go2 camera stream">
  </main>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the Unitree Go2 WebRTC camera stream on a webpage."
    )
    parser.add_argument(
        "--ip",
        default=DEFAULT_ROBOT_HOST,
        help=f"Unitree robot IP/host for LocalSTA mode. Default: {DEFAULT_ROBOT_HOST}",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_WEB_HOST,
        help=f"Web server bind address. Default: {DEFAULT_WEB_HOST}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_WEB_PORT,
        help=f"Web server port. Default: {DEFAULT_WEB_PORT}",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality from 1 to 100. Default: 85",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the first frame before reporting a timeout.",
    )
    return parser


def queue_error(error_queue: Queue, exc: Exception) -> None:
    try:
        error_queue.put_nowait(exc)
    except Full:
        pass


def drop_and_put(frame_queue: Queue, frame) -> None:
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except Empty:
            pass
    frame_queue.put_nowait(frame)


def start_webrtc(frame_queue: Queue, error_queue: Queue, robot_host: str):
    try:
        from aiortc import MediaStreamTrack
        from unitree_webrtc_connect.webrtc_driver import (
            UnitreeWebRTCConnection,
            WebRTCConnectionMethod,
        )
    except ImportError as exc:
        print(
            "[ERROR] Missing Unitree WebRTC dependencies. Run this in the same "
            "environment where Unitree/viewVideoStream.py works.",
            file=sys.stderr,
            flush=True,
        )
        print(f"[ERROR] Import error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=robot_host)

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            drop_and_put(frame_queue, img)

    async def setup():
        print(f"[INFO] Connecting to Unitree Go2 at {robot_host}", flush=True)
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
    if loop is not None:
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None:
        thread.join(timeout=5.0)


def wait_for_first_frame(frame_queue: Queue, error_queue: Queue, timeout: float):
    deadline = time.time() + max(0.1, timeout)

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


def make_app(frame_queue: Queue, error_queue: Queue, robot_host: str, jpeg_quality: int):
    app = Flask(__name__)
    jpeg_quality = min(100, max(1, jpeg_quality))

    @app.route("/")
    def index():
        return render_template_string(
            HTML_PAGE,
            robot_host=robot_host,
            stream_url="/stream.mjpg",
        )

    @app.route("/stream.mjpg")
    def stream():
        return Response(
            stream_with_context(generate_mjpeg(frame_queue, error_queue, jpeg_quality)),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def generate_mjpeg(frame_queue: Queue, error_queue: Queue, jpeg_quality: int):
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    last_frame = None

    while True:
        try:
            stream_error = error_queue.get_nowait()
        except Empty:
            stream_error = None
        if stream_error is not None:
            print(f"[ERROR] Unitree video stream failed: {stream_error}", flush=True)
            return

        try:
            last_frame = frame_queue.get(timeout=1.0)
        except Empty:
            if last_frame is None:
                continue

        ok, jpeg = cv2.imencode(".jpg", last_frame, encode_params)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )


def main() -> int:
    args = build_parser().parse_args()
    frame_queue = Queue(maxsize=1)
    error_queue = Queue(maxsize=1)

    loop, thread = start_webrtc(frame_queue, error_queue, args.ip)
    first_frame, connect_error = wait_for_first_frame(
        frame_queue,
        error_queue,
        args.startup_timeout,
    )

    if first_frame is None:
        stop_webrtc(loop, thread)
        print(f"[ERROR] Could not open the Go2 video stream: {connect_error}", flush=True)
        return 1

    drop_and_put(frame_queue, first_frame)
    app = make_app(frame_queue, error_queue, args.ip, args.jpeg_quality)

    shown_host = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    print(f"[INFO] Open http://{shown_host}:{args.port}/", flush=True)

    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        stop_webrtc(loop, thread)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
