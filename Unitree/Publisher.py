import asyncio
import json
import time
import ssl
import logging
import threading
from pathlib import Path
from queue import Queue
import rclpy
from std_msgs.msg import String

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

# ── Inference ────────────────────────────────────────────────────────────────

def process_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Run YOLO segmentation on a BGR frame.
    Returns (annotated_overlay, direction_angle_degrees).
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    results = model(frame, conf=DETECTION_CONF, verbose=False)
    midpoints = []

    for r in results:
        if r.masks is None or len(r.masks.data) == 0:
            continue

        mask = r.masks.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Green overlay on segmented area
        green   = np.full_like(frame, (0, 255, 0))
        blended = cv2.addWeighted(frame, 0.3, green, 0.7, 0)
        overlay[mask > 0] = blended[mask > 0]

        # Scan lines + midpoints
        for rr in SCAN_HEIGHTS:
            y = int(h * rr)
            if y >= h:
                continue
            scan_row = mask[y, :]
            idx = np.where(scan_row > 0)[0]
            if len(idx) > 0:
                mx = int(np.mean(idx))
                midpoints.append((mx, y))
                cv2.circle(overlay, (mx, y), 5, (255, 0, 0), -1)
            cv2.line(overlay, (0, y), (w, y), (150, 150, 150), 1)

    # Heading arrow
    start_point     = (w // 2, h)
    direction_angle = 90.0

    if midpoints:
        avg_x        = int(np.mean([p[0] for p in midpoints]))
        target_point = (avg_x, min(p[1] for p in midpoints))
        dx           = avg_x - start_point[0]
        dy           = start_point[1] - target_point[1]
        direction_angle = float(np.degrees(np.arctan2(dy, dx)))
        cv2.arrowedLine(overlay, start_point, target_point, (0, 0, 255), 5, tipLength=0.2)
    else:
        cv2.arrowedLine(overlay, start_point, (w // 2, int(h * 0.6)), (0, 0, 255), 5, tipLength=0.2)

    return overlay, direction_angle


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

    # ROS2 publisher setup
    rclpy.init(args=None)
    ros_node = rclpy.create_node("unitree_heading_publisher")
    heading_pub = ros_node.create_publisher(String, ROS_TOPIC, 10)
    frame_id = 0

    next_save_at = time.time()
    current_speed = 0.0  # later vervangen door aruco/path logic

    # Only create preview window when enabled
    if SHOW_PREVIEW:
        cv2.imshow("Segmentation + Heading", np.zeros((480, 640, 3), dtype=np.uint8))
        cv2.waitKey(1)

    try:
        while True:
            if frame_queue.empty():
                time.sleep(0.005)
                continue

            img = frame_queue.get()
            overlay, direction_angle = process_frame(img)

            if SHOW_PREVIEW:
                cv2.imshow("Segmentation + Heading", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Convert to radians for subscriber:
            # direction_angle is image heading in deg (90 = straight).
            # heading_rad: 0 = straight, + right, - left.
            heading_rad = float(np.deg2rad(90.0 - direction_angle))

            msg = String()
            msg.data = json.dumps({
                "heading": heading_rad,
                "speed": current_speed,
                "frame_id": frame_id
            })
            heading_pub.publish(msg)

            # Console feedback (headless visual)
            if frame_id % PRINT_EVERY_N_FRAMES == 0:
                print(
                    f"[PUB] frame_id={frame_id} | direction_deg={direction_angle:.2f} | heading_rad={heading_rad:.3f} | speed={current_speed:.2f}",
                    flush=True
                )

            frame_id += 1

            # Let ROS process callbacks/events (non-blocking)
            rclpy.spin_once(ros_node, timeout_sec=0.0)

            # Periodic save
            now = time.time()
            if now >= next_save_at:
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = RECORDS_DIR / f"frame_{ts}.jpg"
                cv2.imwrite(str(out_path), overlay)
                next_save_at = now + SAVE_INTERVAL_SEC

    finally:
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


