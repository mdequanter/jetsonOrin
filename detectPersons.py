import asyncio
import json
import base64
import os
import ssl
import urllib.request

import websockets
import cv2
import numpy as np

# -----------------------------
# WebSocket instellingen
# -----------------------------
DEFAULT_ROOM = "/ws/jetsonDetectPersons"
BEARER_TOKEN = "B6zifTK3JWeH6E2tThPKLMwxt0QdqXVJ76GHfq7kTvs"
SIGNALING_SERVER = f"wss://signaling.ehb.be{DEFAULT_ROOM}"

# -----------------------------
# Gezichtsherkenning instellingen
# -----------------------------
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

YUNET_PATH = os.path.join("models", "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join("models", "face_recognition_sface_2021dec.onnx")
KNOWN_DIR = "faceassist/known"

MIN_FACE = 50
SCORE_TH = 0.9
NMS_TH = 0.3
TOPK = 5000

THRESHOLD = 0.60
MARGIN = 0.06


# -----------------------------
# Helpers
# -----------------------------
def download_if_missing(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"[INFO] Downloading {os.path.basename(path)}")
        urllib.request.urlretrieve(url, path)


def load_known(known_dir: str):
    known = {}
    if not os.path.isdir(known_dir):
        return known

    for fn in os.listdir(known_dir):
        if fn.lower().endswith(".npz"):
            name = os.path.splitext(fn)[0]
            data = np.load(os.path.join(known_dir, fn))
            feats = data["features"].astype(np.float32)
            known[name] = feats

    return known


def best_match(recognizer, feat, known: dict):
    scores = []
    for name, feats in known.items():
        best = -1.0
        for f in feats:
            s = float(recognizer.match(feat, f, cv2.FaceRecognizerSF_FR_COSINE))
            if s > best:
                best = s
        scores.append((name, best))

    if not scores:
        return None, -1.0, -1.0

    scores.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else -1.0
    return best_name, best_score, second_score


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


def unique_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# -----------------------------
# Main verwerking
# -----------------------------
async def receive_and_recognize():
    download_if_missing(YUNET_URL, YUNET_PATH)
    download_if_missing(SFACE_URL, SFACE_PATH)

    known = load_known(KNOWN_DIR)
    print(f"[INFO] Loaded known persons: {list(known.keys())}")

    detector = None
    recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")

    ssl_context = ssl.create_default_context()

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
            ),
            "Authorization": f"Bearer {BEARER_TOKEN}",
        },
    ) as ws:
        print(f"[INFO] Connected to {SIGNALING_SERVER}")

        while True:
            msg = await ws.recv()
            frame = decode_message_to_frame(msg)

            if frame is None:
                continue

            h, w = frame.shape[:2]

            if detector is None:
                detector = cv2.FaceDetectorYN.create(
                    YUNET_PATH, "", (w, h), SCORE_TH, NMS_TH, TOPK
                )
            else:
                detector.setInputSize((w, h))

            _, faces = detector.detect(frame)

            detected_names = []

            persons = []

            if faces is not None:
                frame_width = frame.shape[1]

                for face in faces:
                    x, y, fw, fh = face[:4].astype(int)

                    if fw < MIN_FACE:
                        continue

                    aligned = recognizer.alignCrop(frame, face)
                    feat = recognizer.feature(aligned).astype(np.float32)

                    best_name, best_score, second_score = best_match(recognizer, feat, known)

                    confident = (
                        best_name is not None
                        and best_score >= THRESHOLD
                        and (best_score - second_score) >= MARGIN
                    )

                    name = best_name if confident else "Onbekend"

                    face_center_x = x + (fw / 2.0)

                    if face_center_x < frame_width / 3:
                        face_position = "LEFT"
                    elif face_center_x > (2 * frame_width / 3):
                        face_position = "RIGHT"
                    else:
                        face_position = "FRONT"

                    persons.append({
                        "name": name,
                        "x": int(x),
                        "y": int(y),
                        "w": int(fw),
                        "h": int(fh),
                        "face_position": face_position
                    })

            await ws.send(json.dumps({"persons": persons}))
            print(f"[SEND] {persons}")


if __name__ == "__main__":
    asyncio.run(receive_and_recognize())