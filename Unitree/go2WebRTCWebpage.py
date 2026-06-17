import cv2

import asyncio
import logging
import threading
import time
from queue import Queue
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack
from flask import Flask, Response, render_template_string


# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)


ROBOT_IP = "unitree.local"
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080
JPEG_QUALITY = 85

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


def create_app(frame_queue):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML_PAGE, robot_ip=ROBOT_IP)

    @app.route("/video_feed")
    def video_feed():
        return Response(
            gen_frames(frame_queue),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def gen_frames(frame_queue):
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    while True:
        if not frame_queue.empty():
            img = frame_queue.get()
            ok, jpeg = cv2.imencode(".jpg", img, encode_params)
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
        else:
            # Sleep briefly to prevent high CPU usage
            time.sleep(0.01)


def main():
    frame_queue = Queue()

    # Choose a connection method (uncomment the correct one)
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP)
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    # Async function to receive video frames and put them in the queue
    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            # Convert the frame to a NumPy array
            img = frame.to_ndarray(format="bgr24")
            frame_queue.put(img)

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

    app = create_app(frame_queue)
    print(f"Open http://localhost:{WEB_PORT}/", flush=True)

    try:
        app.run(host=WEB_HOST, port=WEB_PORT, threaded=True, use_reloader=False)
    finally:
        # Stop the asyncio event loop
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()


if __name__ == "__main__":
    main()
