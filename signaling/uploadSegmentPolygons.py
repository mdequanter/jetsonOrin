import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template_string, request, url_for
from werkzeug.utils import secure_filename

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: ultralytics. Install it with: pip install ultralytics"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
UPLOAD_DIR = SCRIPT_DIR / "uploads"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMGSZ = 640
JPEG_QUALITY = 90

COLORS = (
    (0, 255, 255),
    (255, 0, 255),
    (0, 180, 255),
    (80, 220, 80),
    (255, 140, 0),
    (220, 120, 220),
    (120, 220, 255),
)

app = Flask(__name__)
app.secret_key = "segment-upload-dev"

MODEL_CACHE = {}

HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Segmentation Polygons</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: #f5f7fa;
      color: #17202a;
      font-family: Arial, Helvetica, sans-serif;
    }
    header {
      background: #1f2933;
      color: #f8fafc;
      padding: 16px 24px;
      border-bottom: 1px solid #111827;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
    }
    main {
      width: min(1180px, calc(100% - 32px));
      margin: 20px auto 36px;
    }
    form {
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(160px, 220px) 120px 120px;
      gap: 12px;
      align-items: end;
      padding: 16px;
      background: #ffffff;
      border: 1px solid #d7dee8;
      border-radius: 8px;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 13px;
      font-weight: 700;
      color: #334155;
    }
    input, select, button {
      min-height: 38px;
      font: inherit;
      border: 1px solid #c7d0dc;
      border-radius: 6px;
      padding: 8px 10px;
      background: #fff;
    }
    button {
      cursor: pointer;
      border-color: #2563eb;
      background: #2563eb;
      color: #fff;
      font-weight: 700;
    }
    .messages {
      margin: 12px 0;
      padding: 10px 12px;
      background: #fff1f2;
      border: 1px solid #fecdd3;
      color: #9f1239;
      border-radius: 6px;
    }
    .summary {
      margin: 16px 0;
      color: #475569;
      font-size: 14px;
    }
    .result {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(320px, 0.8fr);
      gap: 16px;
      align-items: start;
    }
    .panel {
      background: #ffffff;
      border: 1px solid #d7dee8;
      border-radius: 8px;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 12px 14px;
      font-size: 16px;
      border-bottom: 1px solid #d7dee8;
      background: #f8fafc;
    }
    .image-wrap {
      padding: 12px;
      background: #111827;
    }
    img {
      display: block;
      width: 100%;
      height: auto;
      max-height: 78vh;
      object-fit: contain;
    }
    pre {
      margin: 0;
      padding: 12px;
      overflow: auto;
      max-height: 78vh;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
      background: #0f172a;
      color: #e2e8f0;
    }
    @media (max-width: 840px) {
      form, .result {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Segmentation Polygons</h1>
  </header>
  <main>
    <form method="post" enctype="multipart/form-data">
      <label>
        Image file
        <input type="file" name="file" accept="image/*" required>
      </label>
      <label>
        Model
        <select name="model">
          {% for model in models %}
            <option value="{{ model }}" {% if model == selected_model %}selected{% endif %}>{{ model }}</option>
          {% endfor %}
        </select>
      </label>
      <label>
        Confidence
        <input type="number" name="conf" min="0.01" max="1" step="0.01" value="{{ conf }}">
      </label>
      <button type="submit">Segment</button>
    </form>

    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="messages">
          {% for message in messages %}<div>{{ message }}</div>{% endfor %}
        </div>
      {% endif %}
    {% endwith %}

    {% if result %}
      <div class="summary">
        Model: <strong>{{ result.model }}</strong> |
        File: <strong>{{ result.filename }}</strong> |
        Masks: <strong>{{ result.mask_count }}</strong> |
        Inference: <strong>{{ "%.1f"|format(result.inference_ms) }} ms</strong>
      </div>
      <section class="result">
        <div class="panel">
          <h2>Segmented Image</h2>
          <div class="image-wrap">
            <img src="data:image/jpeg;base64,{{ result.image_b64 }}" alt="Segmented upload">
          </div>
        </div>
        <div class="panel">
          <h2>Polygons</h2>
          <pre>{{ result.polygons_json }}</pre>
        </div>
      </section>
    {% endif %}
  </main>
</body>
</html>
"""


def model_paths():
    paths = sorted(MODELS_DIR.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt models found in {MODELS_DIR}")
    return paths


def model_names():
    return [path.stem for path in model_paths()]


def resolve_model_path(model_name):
    selected = (model_name or "").strip()
    paths = {path.stem: path for path in model_paths()}
    if selected in paths:
        return paths[selected]
    return next(iter(paths.values()))


def load_model(model_path):
    key = str(model_path.resolve())
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = YOLO(key, verbose=False)
    return MODEL_CACHE[key]


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def class_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def result_arrays(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return [], []

    classes = []
    confidences = []
    if boxes.cls is not None:
        classes = boxes.cls.detach().cpu().numpy().astype(int).tolist()
    if boxes.conf is not None:
        confidences = boxes.conf.detach().cpu().numpy().tolist()
    return classes, confidences


def draw_label(frame, text, anchor, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    padding = 5

    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, int(anchor[0]))
    y = max(text_size[1] + padding * 2, int(anchor[1]))

    cv2.rectangle(
        frame,
        (x, y - text_size[1] - padding * 2),
        (x + text_size[0] + padding * 2, y + baseline),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + padding, y - padding),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_polygons(frame, result, alpha=0.35, line_width=2):
    masks = getattr(result, "masks", None)
    if masks is None or masks.xy is None:
        return []

    height, width = frame.shape[:2]
    overlay = frame.copy()
    classes, confidences = result_arrays(result)
    names = getattr(result, "names", {})
    detections = []
    polygons = []

    for index, polygon in enumerate(masks.xy):
        if polygon is None or len(polygon) < 3:
            continue

        points = np.round(polygon).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)

        class_id = classes[index] if index < len(classes) else index
        confidence = confidences[index] if index < len(confidences) else None
        label = class_name(names, class_id)
        color = COLORS[class_id % len(COLORS)]
        contour = points.reshape((-1, 1, 2))

        cv2.fillPoly(overlay, [contour], color)
        detections.append((contour, label, confidence, color))
        polygons.append(
            {
                "index": len(polygons),
                "class_id": int(class_id),
                "label": label,
                "confidence": None if confidence is None else round(float(confidence), 4),
                "points": [[int(x), int(y)] for x, y in points.tolist()],
            }
        )

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)

    for contour, label, confidence, color in detections:
        cv2.polylines(frame, [contour], True, color, line_width, cv2.LINE_AA)
        min_x = int(contour[:, 0, 0].min())
        min_y = int(contour[:, 0, 1].min())
        text = label if confidence is None else f"{label} {confidence:.2f}"
        draw_label(frame, text, (min_x, min_y), color)

    return polygons


def image_to_base64(frame):
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Could not encode segmented image.")
    return base64.b64encode(jpeg.tobytes()).decode("ascii")


def segment_image(file_storage, model_name, conf):
    filename = secure_filename(file_storage.filename or "")
    if not filename:
        raise ValueError("No file selected.")
    if not allowed_file(filename):
        raise ValueError("Upload an image file: jpg, jpeg, png, bmp, or webp.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / f"{int(time.time() * 1000)}_{filename}"
    file_storage.save(upload_path)

    frame = cv2.imread(str(upload_path))
    if frame is None:
        raise ValueError("OpenCV could not read this image.")

    model_path = resolve_model_path(model_name)
    model = load_model(model_path)
    started = time.perf_counter()
    result = model.predict(frame, conf=conf, imgsz=DEFAULT_IMGSZ, verbose=False)[0]
    total_ms = (time.perf_counter() - started) * 1000.0
    inference_ms = result.speed.get("inference", total_ms) if result.speed else total_ms

    polygons = draw_polygons(frame, result)
    return {
        "filename": filename,
        "model": model_path.stem,
        "mask_count": len(polygons),
        "inference_ms": inference_ms,
        "image_b64": image_to_base64(frame),
        "polygons_json": json.dumps(polygons, indent=2),
    }


def parse_confidence(value):
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE
    return min(1.0, max(0.01, conf))


@app.route("/", methods=["GET", "POST"])
def index():
    models = model_names()
    selected_model = request.form.get("model") or models[0]
    conf = parse_confidence(request.form.get("conf", DEFAULT_CONFIDENCE))
    result = None

    if request.method == "POST":
        uploaded = request.files.get("file")
        if uploaded is None:
            flash("No file was uploaded.")
            return redirect(url_for("index"))

        try:
            result = segment_image(uploaded, selected_model, conf)
            selected_model = result["model"]
        except Exception as exc:
            flash(str(exc))

    return render_template_string(
        HTML_PAGE,
        models=models,
        selected_model=selected_model,
        conf=f"{conf:.2f}",
        result=result,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, threaded=True)
