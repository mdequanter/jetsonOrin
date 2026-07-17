from __future__ import annotations

import base64
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent

MODELS_DIRS = []
raw_models_dir = os.environ.get("MODELS_DIR")
if raw_models_dir:
    MODELS_DIRS.append(Path(raw_models_dir).resolve())
MODELS_DIRS.append(HERE / "models")
MODELS_DIRS.append(HERE)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path is None:
            continue
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


MODELS_DIRS = _dedupe_paths(MODELS_DIRS)

ALLOWED_PATH_LABELS = {"path", "path-oxod"}
SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ARUCO_DICTIONARY_NAME = "DICT_4X4_50"
ARUCO_DISTANCE_CALIBRATION_POINTS = [
    (1.0, 1587.0),
    (0.5, 6200.0),
    (0.3, 16200.0),
    (1.4, 750.0),
]
ARUCO_AREA_AT_1M_PX2 = sum(
    area_px2 * (distance_m**2)
    for distance_m, area_px2 in ARUCO_DISTANCE_CALIBRATION_POINTS
) / len(ARUCO_DISTANCE_CALIBRATION_POINTS)

app = Flask(__name__)

_lock = threading.Lock()
_models: dict[str, YOLO] = {}


def find_model_files() -> list[Path]:
    for models_dir in MODELS_DIRS:
        if not models_dir.exists():
            continue
        model_paths = sorted(models_dir.glob("*.pt"))
        if model_paths:
            return model_paths
    return []


MODEL_FILES = find_model_files()
MODELS_BY_NAME = {}
MODEL_ORDER = []
for model_path in MODEL_FILES:
    model_name = model_path.stem
    MODEL_ORDER.append(model_name)
    MODELS_BY_NAME[model_name] = {"path": model_path}

DEFAULT_MODEL_NAME = "unrealsim" if "unrealsim" in MODELS_BY_NAME else (MODEL_ORDER[0] if MODEL_ORDER else "")


def load_model(model_name: str) -> YOLO:
    if model_name not in _models:
        if model_name not in MODELS_BY_NAME:
            raise FileNotFoundError(f"Model '{model_name}' not found")
        model_path = MODELS_BY_NAME[model_name]["path"]
        _models[model_name] = YOLO(str(model_path), verbose=False)
    return _models[model_name]


def create_aruco_detector():
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, "ArucoDetector"):
        print("OpenCV ArUcoDetector is not available; install opencv-contrib-python to enable marker detection.")
        return None

    dictionary_id = getattr(aruco, ARUCO_DICTIONARY_NAME, None)
    if dictionary_id is None:
        print(f"Unknown ArUco dictionary: {ARUCO_DICTIONARY_NAME}")
        return None

    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    parameters = aruco.DetectorParameters()
    return aruco.ArucoDetector(dictionary, parameters)


ARUCO_DETECTOR = create_aruco_detector()


def resolve_model_name(selected_model: Any) -> str:
    if selected_model is None:
        return DEFAULT_MODEL_NAME or ""

    model_key = str(selected_model).strip()
    if not model_key:
        return DEFAULT_MODEL_NAME or ""

    if model_key in MODELS_BY_NAME:
        return model_key

    model_key_no_ext = Path(model_key).stem
    if model_key_no_ext in MODELS_BY_NAME:
        return model_key_no_ext

    if model_key.isdigit():
        model_index = int(model_key) - 1
        if 0 <= model_index < len(MODEL_ORDER):
            return MODEL_ORDER[model_index]

    return DEFAULT_MODEL_NAME or ""


def parse_detection_confidence(payload: Any, fallback: float) -> float:
    if not isinstance(payload, dict):
        return fallback

    raw_value = (
        payload.get("DETECTION_CONFIDENCE")
        if payload.get("DETECTION_CONFIDENCE") is not None
        else payload.get("detection_confidence")
    )
    if raw_value is None:
        raw_value = payload.get("confidence")
    if raw_value is None:
        raw_value = payload.get("conficence")

    if raw_value is None:
        return fallback

    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return fallback

    return max(0.0, min(1.0, parsed))


def decode_frame_bytes(frame_bytes: bytes) -> Any:
    try:
        np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("unable to decode image")
        return frame
    except Exception as exc:
        raise ValueError(f"invalid image payload: {exc}") from exc


def extract_frame_bytes() -> bytes:
    if request.files:
        uploaded = None
        for key in ("image", "frame", "file"):
            if key in request.files:
                uploaded = request.files[key]
                break
        if uploaded is not None and uploaded.filename:
            return uploaded.read()

    body = request.get_json(silent=True) or {}
    for key in ("image", "frame", "data", "image_data"):
        value = body.get(key)
        if isinstance(value, str):
            if value.startswith("data:image"):
                header, _, encoded = value.partition(",")
                if not encoded:
                    raise ValueError("empty image payload")
                return base64.b64decode(encoded)
            return base64.b64decode(value)

    if isinstance(body.get("frame_bytes"), (bytes, bytearray)):
        return bytes(body["frame_bytes"])

    raise ValueError("no image payload supplied")


def get_allowed_mask_indices(result, model_names):
    if result.boxes is None or result.boxes.cls is None:
        return []

    allowed_indices = []
    class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
    for index, class_id in enumerate(class_ids):
        label = str(model_names.get(class_id, "")).strip().lower()
        if label in ALLOWED_PATH_LABELS:
            allowed_indices.append(index)
    return allowed_indices


def get_aruco_marker_area(corners):
    points = corners.reshape(4, 2).astype(np.float32)
    area = float(abs(cv2.contourArea(points)))
    return round(area, 2)


def estimate_aruco_marker_distance_m(area_px2):
    if area_px2 <= 0:
        return None
    return round(float(np.sqrt(ARUCO_AREA_AT_1M_PX2 / area_px2)), 2)


def get_horizontal_position_hour(offset_x_px, image_center_x, center_tolerance_px):
    if abs(offset_x_px) <= center_tolerance_px:
        return 12

    max_offset_px = max(image_center_x - center_tolerance_px, 1)
    offset_ratio = min((abs(offset_x_px) - center_tolerance_px) / max_offset_px, 1.0)

    if offset_x_px < 0:
        if offset_ratio <= 1 / 3:
            return 11
        if offset_ratio <= 2 / 3:
            return 10
        return 9

    if offset_ratio <= 1 / 3:
        return 1
    if offset_ratio <= 2 / 3:
        return 2
    return 3


def detect_aruco_markers(frame, center_tolerance_px=30):
    if ARUCO_DETECTOR is None:
        return []

    h, w = frame.shape[:2]
    image_center_x = w / 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _ = ARUCO_DETECTOR.detectMarkers(gray)

    if marker_ids is None:
        return []

    markers = []
    marker_ids = marker_ids.flatten().astype(int).tolist()

    for marker_id, corners in zip(marker_ids, marker_corners):
        points = corners.reshape(4, 2).astype(np.float32)
        marker_center_x = float(np.mean(points[:, 0]))
        marker_center_y = float(np.mean(points[:, 1]))
        offset_x_px = marker_center_x - image_center_x
        horizontal_position = get_horizontal_position_hour(offset_x_px, image_center_x, center_tolerance_px)
        area_px2 = get_aruco_marker_area(corners)
        markers.append(
            {
                "id": marker_id,
                "area_px2": area_px2,
                "distance_m": estimate_aruco_marker_distance_m(area_px2),
                "center_x_px": round(marker_center_x, 2),
                "center_y_px": round(marker_center_y, 2),
                "offset_x_px": round(offset_x_px, 2),
                "horizontal_position": horizontal_position,
            }
        )

    return sorted(markers, key=lambda marker: marker["area_px2"], reverse=True)


def compute_heading_to_point(frame, target_x, target_y):
    h, w = frame.shape[:2]
    start_x = w // 2
    start_y = h
    dx = target_x - start_x
    dy = start_y - target_y
    return float(np.degrees(np.arctan2(dy, dx)))


def compute_heading_to_marker(frame, aruco_markers):
    if not aruco_markers:
        return None
    marker = aruco_markers[0]
    return compute_heading_to_point(frame, marker["center_x_px"], marker["center_y_px"])


def compute_heading(frame, model=None, return_masks=False, detection_confidence=0.8):
    h, w = frame.shape[:2]

    model_name = resolve_model_name(model)
    if not model_name:
        return 90.0, []

    with _lock:
        yolo_model = load_model(model_name)
        model_names = getattr(yolo_model, "names", {})
        results = yolo_model(frame, conf=detection_confidence, verbose=False)

    midpoints = []
    result_masks = []
    for r in results:
        if r.masks is None or len(r.masks.data) == 0:
            continue

        allowed_mask_indices = get_allowed_mask_indices(r, model_names)
        for mask_index in allowed_mask_indices:
            if mask_index >= len(r.masks.data):
                continue

            mask_tensor = r.masks.data[mask_index]
            mask = mask_tensor.cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            if return_masks:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_points = []
                for contour in contours:
                    if len(contour) == 0:
                        continue
                    contour_points.append([[int(point[0][0]), int(point[0][1])] for point in contour])
                if contour_points:
                    result_masks.append(contour_points)

            for rr in SCAN_HEIGHTS:
                y = int(h * rr)
                if y >= h:
                    continue
                idx = np.where(mask[y, :] > 0)[0]
                if len(idx) > 0:
                    midpoints.append((int(np.mean(idx)), y))

    if not midpoints:
        return 90.0, result_masks

    avg_x = int(np.mean([p[0] for p in midpoints]))
    target_y = min([p[1] for p in midpoints])
    return compute_heading_to_point(frame, avg_x, target_y), result_masks


@app.get("/")
def index():
    return jsonify({
        "ok": True,
        "service": "freewalkFlask",
        "models": MODEL_ORDER,
        "default_model": DEFAULT_MODEL_NAME,
        "endpoints": {
            "health": "/health",
            "segment": "/api/freewalk/segment",
        },
    })


@app.get("/health")
def health():
    return jsonify({"ok": True, "models": MODEL_ORDER, "default_model": DEFAULT_MODEL_NAME})


@app.post("/api/freewalk/segment")
def segment():
    try:
        frame_bytes = extract_frame_bytes()
        frame = decode_frame_bytes(frame_bytes)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    payload = request.get_json(silent=True) or {}
    model_name = payload.get("model", DEFAULT_MODEL_NAME)
    detection_confidence = parse_detection_confidence(payload, 0.8)
    return_masks = bool(payload.get("returnMasks", False))

    aruco_markers = detect_aruco_markers(frame)
    marker_heading = compute_heading_to_marker(frame, aruco_markers)

    if marker_heading is None:
        heading, result_masks = compute_heading(frame, model=model_name, return_masks=return_masks, detection_confidence=detection_confidence)
    else:
        heading, result_masks = compute_heading(frame, model=model_name, return_masks=return_masks, detection_confidence=detection_confidence)

    if marker_heading is not None:
        heading = marker_heading

    response_payload = {
        "ok": True,
        "heading": round(float(heading), 2),
        "model": resolve_model_name(model_name),
        "detection_confidence": round(detection_confidence, 2),
        "aruco_markers": aruco_markers,
    }

    if marker_heading is not None:
        response_payload["marker_heading"] = round(float(marker_heading), 2)

    if return_masks:
        response_payload["resultMasks"] = result_masks

    return jsonify(response_payload)


if __name__ == "__main__":
    port = int(os.environ.get("FREEWALK_PORT", "5003"))
    host = os.environ.get("FREEWALK_HOST", "127.0.0.1")
    print(f"freewalkFlask listening on http://{host}:{port}")
    print(f"Models found: {MODEL_ORDER or '(none — drop a .pt into the repo root or models/)'}")
    app.run(host=host, port=port, debug=False, threaded=True)
