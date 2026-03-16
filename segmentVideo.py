import asyncio
import json
import base64

import websockets
import cv2
import numpy as np
from ultralytics import YOLO
import ssl

# --- Minimale vaste instellingen ---

DEFAULT_ROOM = "/ws/pathnavigation"
DEFAULT_TOKEN = "B6zifTK3JWeH6E2tThPKLMwxt0QdqXVJ76GHfq7kTvs"

#SIGNALING_SERVER = f"ws://localhost:9000{DEFAULT_ROOM}"
SIGNALING_SERVER = f"wss://signaling.ehb.be{DEFAULT_ROOM}"

MODEL_PATH = r"faceassist/models/unrealsim.pt"
BEARER_TOKEN = DEFAULT_TOKEN
DETECTION_CONFIDENCE = 0.3
SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

model = YOLO(MODEL_PATH, verbose=False)

def decode_message_to_frame(msg):
    """
    msg kan bytes (raw JPEG) of str (JSON met base64 JPEG) zijn.
    Retourneert OpenCV BGR frame of None.
    """
    try:
        if isinstance(msg, (bytes, bytearray)):
            jpeg_bytes = bytes(msg)

        elif isinstance(msg, str):
            try:
                payload = json.loads(msg)
            except json.JSONDecodeError:
                return None

            b64 = payload.get("data")
            if not b64:
                return None
            jpeg_bytes = base64.b64decode(b64)

        else:
            return None

        np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None

async def receive_and_infer():
    ssl_context = ssl.create_default_context()  # Uncomment for wss:// with valid certs
    #ssl_context = None

    async with websockets.connect(
        SIGNALING_SERVER,
        ssl=ssl_context,
        origin="http://localhost",
        compression=None,
        extra_headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            )
        },
    ) as ws:
        
        print(f"Verbonden met signaling server ({SIGNALING_SERVER})")

        pending_frame_id = None

        while True:
            msg = await ws.recv()

            frame_id = None

            print(f"Ontvangen bericht van server (type: {type(msg)})")

            # frame_meta stil negeren (frame_id bijhouden)
            if isinstance(msg, str):
                try:
                    payload = json.loads(msg)
                    if payload.get("type") == "frame_meta":
                        pending_frame_id = payload.get("frame_id")
                        continue
                except json.JSONDecodeError:
                    pass

            frame = decode_message_to_frame(msg)

            if isinstance(msg, (bytes, bytearray)):
                frame_id = pending_frame_id
                pending_frame_id = None
            elif isinstance(msg, str):
                try:
                    payload = json.loads(msg)
                    frame_id = payload.get("frame_id", pending_frame_id)
                    pending_frame_id = None
                except Exception:
                    frame_id = pending_frame_id
                    pending_frame_id = None

            if frame is None:
                continue

            h, w = frame.shape[:2]

            # --- Inference ---
            results = model(frame, conf=DETECTION_CONFIDENCE, verbose=False)

            midpoints = []

            for r in results:
                if r.masks is None or len(r.masks.data) == 0:
                    continue

                mask = r.masks.data[0].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                # scanlijnen + midpoints (zonder tekenen)
                for rr in SCAN_HEIGHTS:
                    y = int(h * rr)
                    if y >= h:
                        continue

                    scan_row = mask[y, :]
                    idx = np.where(scan_row > 0)[0]
                    if len(idx) > 0:
                        mx = int(np.mean(idx))
                        midpoints.append((mx, y))

            # --- Heading berekenen ---
            direction_angle = 90.0  # default
            start_x = w // 2
            start_y = h

            if midpoints:
                avg_x = int(np.mean([p[0] for p in midpoints]))
                target_y = min([p[1] for p in midpoints])

                dx = avg_x - start_x
                dy = start_y - target_y
                direction_angle = float(np.degrees(np.arctan2(dy, dx)))

            print(f"xxxx Frame ID: {frame_id}, Direction Angle: {direction_angle:.2f}°")
            await ws.send(json.dumps({"heading": round(direction_angle, 2), "frame_id": frame_id}))

asyncio.run(receive_and_infer())