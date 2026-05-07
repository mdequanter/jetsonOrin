import asyncio
import json
import time
import ssl
import logging
import threading
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
from ultralytics import YOLO
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from aiortc import MediaStreamTrack

# --- Settings ---
MODEL_PATH        = "/home/jetson/Models/kaai.pt"  # path to your trained YOLO model
DETECTION_CONF    = 0.3
SCAN_HEIGHTS      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
RECORDS_DIR       = Path("records")
SAVE_INTERVAL_SEC = 10.0
ROS_TOPIC         = "unitree_key_cmd"
SHOW_PREVIEW      = False  # <- keep False for headless run (no window)
PRINT_EVERY_N_FRAMES = 5   # console feedback

logging.basicConfig(level=logging.FATAL)

model = YOLO(MODEL_PATH, verbose=False)



# ── WebRTC → frame queue ──────────────────────────────────────────────────────

def start_webrtc(frame_queue: Queue):
    """Start the Unitree WebRTC connection in a background asyncio thread."""

    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="unitree.local")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img   = frame.to_ndarray(format="bgr24")
            # Drop stale frames so inference always sees the latest one
            if not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except Exception:
                    pass
            frame_queue.put(img)

    async def setup():
        await conn.connect()
        conn.video.switchVideoChannel(True)
        conn.video.add_track_callback(recv_camera_stream)

    def run_loop(loop):
        asyncio.set_event_loop(loop)
        loop.run_until_complete(setup())
        loop.run_forever()

    loop   = asyncio.new_event_loop()
    thread = threading.Thread(target=run_loop, args=(loop,), daemon=True)
    thread.start()
    return loop


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)

    frame_queue = Queue(maxsize=1)   # keep only the freshest frame
    loop        = start_webrtc(frame_queue)

    frame_id = 0

    next_save_at = time.time()

    try:
        while True:
            if frame_queue.empty():
                time.sleep(0.005)
                continue

            img = frame_queue.get()

            # Console feedback (headless visual)
            if frame_id % PRINT_EVERY_N_FRAMES == 0:
                print(
                    f"[PUB] frame_id={frame_id}",
                    flush=True
                )

            frame_id += 1

            # Periodic save
            now = time.time()
            if now >= next_save_at:
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = RECORDS_DIR / f"frame_{ts}.jpg"
                cv2.imwrite(str(out_path), img)
                next_save_at = now + SAVE_INTERVAL_SEC

    finally:
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)

if __name__ == "__main__":
    main()


