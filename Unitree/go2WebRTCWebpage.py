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
import threading
from queue import Empty, Queue

import cv2
from aiortc import MediaStreamTrack
from flask import Flask, Response, render_template_string, stream_with_context
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod


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
    return parser


def drop_and_put(frame_queue: Queue, frame) -> None:
    if not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except Empty:
            pass
    frame_queue.put(frame)


def start_webrtc(frame_queue: Queue, robot_host: str):
    # Choose a connection method (uncomment the correct one)
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=robot_host)
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    # Async function to receive video frames and put them in the queue
    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            # Convert the frame to a NumPy array
            img = frame.to_ndarray(format="bgr24")
            drop_and_put(frame_queue, img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)

        async def setup():
            try:
                print(f"[INFO] Connecting to Unitree Go2 at {robot_host}", flush=True)

                # Connect to the device
                await conn.connect()

                # Switch video channel on and start receiving video frames
                conn.video.switchVideoChannel(True)

                # Add callback to handle received video frames
                conn.video.add_track_callback(recv_camera_stream)
            except Exception as e:
                logging.error(f"Error in WebRTC connection: {e}")

        # Run the setup coroutine and then start the event loop
        loop.run_until_complete(setup())
        loop.run_forever()

    # Create a new event loop for the asyncio code
    loop = asyncio.new_event_loop()

    # Start the asyncio event loop in a separate thread
    thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
    thread.start()
    return loop, thread


def stop_webrtc(loop, thread) -> None:
    if loop is not None:
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None:
        thread.join(timeout=5.0)


def make_app(frame_queue: Queue, robot_host: str, jpeg_quality: int):
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
            stream_with_context(generate_mjpeg(frame_queue, jpeg_quality)),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def generate_mjpeg(frame_queue: Queue, jpeg_quality: int):
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    last_frame = None

    while True:
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
    frame_queue = Queue()

    loop, thread = start_webrtc(frame_queue, args.ip)
    app = make_app(frame_queue, args.ip, args.jpeg_quality)

    shown_host = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    print(f"[INFO] Open http://{shown_host}:{args.port}/", flush=True)

    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        stop_webrtc(loop, thread)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
