import argparse
import asyncio
import logging
import threading
import time
from queue import Empty, Queue

import cv2
from aiortc import MediaStreamTrack
from flask import Flask, Response, render_template_string, stream_with_context
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod


# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Go2 WebRTC Stream</title>
  <style>
    body {
      margin: 0;
      min-height: 100vh;
      background: #111;
      color: #f5f5f5;
      font-family: Arial, sans-serif;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 12px 16px;
      background: #202020;
      border-bottom: 1px solid #333;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 18px;
    }
    .host {
      color: #bdbdbd;
      font-size: 14px;
    }
    main {
      flex: 1;
      min-height: 0;
      display: grid;
      place-items: center;
      padding: 16px;
    }
    img {
      width: min(100%, 1280px);
      max-height: calc(100vh - 80px);
      object-fit: contain;
      background: #000;
      border: 1px solid #333;
    }
  </style>
</head>
<body>
  <header>
    <h1>Go2 WebRTC Stream</h1>
    <div class="host">Robot: {{ robot_ip }}</div>
  </header>
  <main>
    <img src="/video_feed" alt="Live Go2 camera stream">
  </main>
</body>
</html>
"""


def build_parser():
    parser = argparse.ArgumentParser(
        description="Show the Unitree Go2 WebRTC camera stream on a webpage."
    )
    parser.add_argument(
        "--ip",
        default="unitree.local",
        help="Unitree robot IP/host for LocalSTA mode.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Web server bind address.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Web server port.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality from 1 to 100.",
    )
    return parser


def put_latest_frame(frame_queue, img):
    if not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except Empty:
            pass
    frame_queue.put(img)


def start_webrtc(frame_queue, robot_ip):
    # Choose a connection method (uncomment the correct one)
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=robot_ip)
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    # Async function to receive video frames and put them in the queue
    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            # Convert the frame to a NumPy array
            img = frame.to_ndarray(format="bgr24")
            put_latest_frame(frame_queue, img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)

        async def setup():
            try:
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
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
    asyncio_thread.start()
    return loop, asyncio_thread


def stop_webrtc(loop, asyncio_thread):
    loop.call_soon_threadsafe(loop.stop)
    asyncio_thread.join()


def create_app(frame_queue, robot_ip, jpeg_quality):
    app = Flask(__name__)
    jpeg_quality = max(1, min(100, jpeg_quality))

    @app.route("/")
    def index():
        return render_template_string(HTML_PAGE, robot_ip=robot_ip)

    @app.route("/video_feed")
    def video_feed():
        return Response(
            stream_with_context(generate_frames(frame_queue, jpeg_quality)),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def generate_frames(frame_queue, jpeg_quality):
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    last_img = None

    while True:
        if not frame_queue.empty():
            last_img = frame_queue.get()
        elif last_img is None:
            time.sleep(0.01)
            continue

        ok, jpeg = cv2.imencode(".jpg", last_img, encode_params)
        if not ok:
            time.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )
        time.sleep(0.01)


def main():
    args = build_parser().parse_args()
    frame_queue = Queue()

    loop, asyncio_thread = start_webrtc(frame_queue, args.ip)
    app = create_app(frame_queue, args.ip, args.jpeg_quality)

    shown_host = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    print(f"Open http://{shown_host}:{args.port}/", flush=True)

    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        stop_webrtc(loop, asyncio_thread)


if __name__ == "__main__":
    main()
