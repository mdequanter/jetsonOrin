from flask import Flask, render_template, send_from_directory, jsonify, redirect, url_for, Response, request
import os
import re
import shutil
import uuid
import base64
from datetime import datetime
import subprocess
import threading
from collections import deque
import cv2
import time
import numpy as np
import json


app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
KNOWN_DIR = os.path.join(BASE_DIR, "known")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown")
MODELS_DIR = os.path.join(BASE_DIR, "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
FACE_EXTRACT_TMP_DIR = os.path.join(BASE_DIR, "_tmp_face_extract")

# ---- Snapshot parsing (zoals je had) ----
# Verwacht: Naam_yyyy_mm_dd_hh_mm_ss.jpg
FILENAME_RE = re.compile(
    r"^(?P<name>.+)_(?P<dt>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.jpe?g$",
    re.IGNORECASE
)

def parse_snapshot_filename(filename: str):
    m = FILENAME_RE.match(filename)
    if not m:
        return {"filename": filename, "name": "Onbekend", "dt": None, "dt_str": "Onbekende datum"}

    name = m.group("name")
    dt_raw = m.group("dt")
    try:
        dt = datetime.strptime(dt_raw, "%Y_%m_%d_%H_%M_%S")
        dt_str = dt.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        dt = None
        dt_str = dt_raw.replace("_", ":")

    return {"filename": filename, "name": name, "dt": dt, "dt_str": dt_str}


UNKNOWN_TS_RE_8 = re.compile(r"^(?P<d>\d{8})_(?P<t>\d{6})$")
UNKNOWN_TS_RE_6 = re.compile(r"^(?P<d>\d{6})_(?P<t>\d{6})$")


def parse_unknown_session_name(folder_name: str):
    for regex, fmt, out_fmt in (
        (UNKNOWN_TS_RE_8, "%Y%m%d_%H%M%S", "%d/%m/%Y %H:%M:%S"),
        (UNKNOWN_TS_RE_6, "%y%m%d_%H%M%S", "%d/%m/%Y %H:%M:%S"),
    ):
        m = regex.match(folder_name)
        if not m:
            continue
        raw = f"{m.group('d')}_{m.group('t')}"
        try:
            dt = datetime.strptime(raw, fmt)
            return dt, dt.strftime(out_fmt)
        except Exception:
            return None, folder_name
    return None, folder_name


def iter_image_files(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in exts:
            files.append(fn)
    files.sort()
    return files


def is_allowed_image_filename(filename: str) -> bool:
    ext = os.path.splitext((filename or "").lower())[1]
    return ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp")


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize_person_name(name: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", (name or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 1
    while True:
        cand = f"{base}_{n}{ext}"
        if not os.path.exists(cand):
            return cand
        n += 1


def load_npz_features(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    key = "features" if "features" in data else (keys[0] if keys else None)
    if key is None:
        return None
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    return arr


def append_features_for_person(person: str, new_features: list):
    if not new_features:
        return 0

    os.makedirs(KNOWN_DIR, exist_ok=True)
    npz_path = os.path.join(KNOWN_DIR, f"{person}.npz")
    new_stack = np.stack(new_features, axis=0).astype(np.float32)

    if os.path.isfile(npz_path):
        old = load_npz_features(npz_path)
        if old is not None:
            merged = np.concatenate([old, new_stack], axis=0)
        else:
            merged = new_stack
    else:
        merged = new_stack

    np.savez_compressed(npz_path, features=merged)
    return int(new_stack.shape[0])


def resolve_face_extract_session_dir(session: str):
    session = (session or "").strip()
    if not session:
        return None
    d = os.path.abspath(os.path.join(FACE_EXTRACT_TMP_DIR, session))
    root = os.path.abspath(FACE_EXTRACT_TMP_DIR)
    if os.path.commonpath([d, root]) != root:
        return None
    return d


def load_face_extract_manifest(session_dir: str):
    path = os.path.join(session_dir, "manifest.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_face_extract_manifest(session_dir: str, manifest: dict):
    path = os.path.join(session_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)



def load_known_embeddings():
    known_dir = KNOWN_DIR
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


def list_unknown_sessions():
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    sessions = []
    for folder in os.listdir(UNKNOWN_DIR):
        full = os.path.join(UNKNOWN_DIR, folder)
        if not os.path.isdir(full):
            continue
        files = iter_image_files(full)
        print(f"Session '{folder}': found {len(files)} image files.")
        dt, dt_str = parse_unknown_session_name(folder)
        sessions.append({
            "folder": folder,
            "count": len(files),
            "preview": files[0] if files else None,
            "dt": dt,
            "dt_str": dt_str,
        })
    sessions.sort(
        key=lambda x: (x["dt"] is not None, x["dt"] or datetime.min, x["folder"]),
        reverse=True,
    )
    return sessions


def resolve_unknown_session_dir(session: str):
    session = (session or "").strip()
    if not session:
        return None
    session_dir = os.path.abspath(os.path.join(UNKNOWN_DIR, session))
    unknown_root = os.path.abspath(UNKNOWN_DIR)
    if os.path.commonpath([session_dir, unknown_root]) != unknown_root:
        return None
    return session_dir


def move_unknown_session_images_to_known(session_dir: str, person: str, session: str):
    """
    Verplaatst snapshots van unknown/<session>/ naar known/<person>/.
    Bestandsnamen krijgen een session-prefix om collisions te vermijden.
    """
    person_dir = os.path.join(KNOWN_DIR, person)
    os.makedirs(person_dir, exist_ok=True)

    files = iter_image_files(session_dir)
    moved = 0
    for idx, fn in enumerate(files, start=1):
        src = os.path.join(session_dir, fn)
        ext = os.path.splitext(fn)[1].lower() or ".jpg"
        base = f"{session}_{idx:04d}"
        dst = os.path.join(person_dir, f"{base}{ext}")

        # Unieke naam forceren als bestand al bestaat.
        n = 1
        while os.path.exists(dst):
            dst = os.path.join(person_dir, f"{base}_{n}{ext}")
            n += 1

        shutil.move(src, dst)
        moved += 1

    return person_dir, moved

# ---- Herkenning procesbeheer ----
RECOGNITION_SCRIPT = os.path.join(BASE_DIR, "nl_launch.py")  # hetzelfde pad als jouw upload/bronbestand :contentReference[oaicite:1]{index=1}
_recognition_proc = None
_log_lines = deque(maxlen=400)
_log_lock = threading.Lock()

def _append_log(line: str):
    with _log_lock:
        _log_lines.append(line.rstrip("\n"))

def _reader_thread(proc: subprocess.Popen):
    try:
        for line in proc.stdout:
            if not line:
                break
            _append_log(line)
    except Exception as e:
        _append_log(f"[LOG-ERROR] {e}")
    finally:
        _append_log("[INFO] Herkenning gestopt.")

def recognition_running() -> bool:
    global _recognition_proc
    return _recognition_proc is not None and _recognition_proc.poll() is None

def start_recognition():
    global _recognition_proc
    if recognition_running():
        return

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(KNOWN_DIR, exist_ok=True)

    # Belangrijk:
    # - nl_launch.py is headless en kan (optioneel) input vragen voor onbekenden.
    #   Deze web-integratie start hem vooral voor "bekenden herkennen" + snapshots opslaan.
    cmd = [
        "python3", RECOGNITION_SCRIPT,
        "--known", KNOWN_DIR,
        # optioneel: zet speak uit als je geen audio wil vanuit deze service:
        # "--no_tts",
        # optioneel: camera settings
        # "--cam", "0",
        # "--width", "640", "--height", "480", "--fps", "15",
    ]

    _append_log("[INFO] Start herkenning…")
    _append_log("[INFO] CMD: " + " ".join(cmd))

    _recognition_proc = subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    t = threading.Thread(target=_reader_thread, args=(_recognition_proc,), daemon=True)
    t.start()

def stop_recognition():
    global _recognition_proc
    if not recognition_running():
        _recognition_proc = None
        return
    _append_log("[INFO] Stop signaal gestuurd…")
    try:
        _recognition_proc.terminate()
    except Exception:
        pass

# ---- Helpers: bekende personen uit known/*.npz ----
def list_known_people():
    os.makedirs(KNOWN_DIR, exist_ok=True)
    people = []
    for fn in os.listdir(KNOWN_DIR):
        if fn.lower().endswith(".npz"):
            people.append(os.path.splitext(fn)[0])
    people.sort(key=lambda s: s.lower())
    return people


def resolve_known_person_dir(person: str):
    person = (person or "").strip()
    if not person:
        return None, None
    person_dir = os.path.abspath(os.path.join(KNOWN_DIR, person))
    known_root = os.path.abspath(KNOWN_DIR)
    if os.path.commonpath([person_dir, known_root]) != known_root:
        return None, None
    npz_path = os.path.join(KNOWN_DIR, f"{person}.npz")
    return person_dir, npz_path


def list_known_people_with_photos():
    items = []
    for person in list_known_people():
        person_dir = os.path.join(KNOWN_DIR, person)
        files = iter_image_files(person_dir) if os.path.isdir(person_dir) else []
        items.append({
            "name": person,
            "count": len(files),
            "preview": files[0] if files else None,
        })
    return items


def list_known_photo_dirs():
    os.makedirs(KNOWN_DIR, exist_ok=True)
    dirs = []
    for name in os.listdir(KNOWN_DIR):
        full = os.path.join(KNOWN_DIR, name)
        if os.path.isdir(full):
            dirs.append(name)
    dirs.sort(key=lambda s: s.lower())
    return dirs


_camera = None
_camera_lock = threading.Lock()
_stream_enabled = False
_stream_lock = threading.Lock()
_face_lock = threading.Lock()
_face_detector = None
_face_recognizer = None
_known_cache = {}
_known_cache_at = 0.0

def get_camera(cam_index=0):
    global _camera
    with _camera_lock:
        if _camera is None or not _camera.isOpened():
            _camera = cv2.VideoCapture(cam_index)
            _camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return _camera

def release_camera():
    global _camera
    with _camera_lock:
        if _camera is not None:
            try:
                _camera.release()
            except Exception:
                pass
        _camera = None

def is_stream_enabled():
    with _stream_lock:
        return _stream_enabled

def set_stream_enabled(val: bool):
    global _stream_enabled
    with _stream_lock:
        _stream_enabled = val

def gen_frames():
    # Wacht tot stream aan staat (of stop meteen)
    while not is_stream_enabled():
        time.sleep(0.1)

    cam = get_camera(0)

    while True:
        # Als iemand op "stop" drukt: stop generator + release camera
        if not is_stream_enabled():
            release_camera()
            return  # <-- beëindigt de HTTP stream netjes

        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        out = frame.copy()
        try:
            known = get_known_embeddings_cached(refresh_sec=5.0)
            with _face_lock:
                detector, recognizer = _get_face_models()
                h, w = out.shape[:2]
                detector.setInputSize((w, h))
                _, faces = detector.detect(out)
                if faces is not None and len(faces) > 0:
                    for i, face in enumerate(faces, start=1):
                        x, y, fw, fh = face[:4].astype(int)
                        label = "Onbekend"
                        color = (0, 0, 255)
                        score_txt = ""

                        try:
                            aligned = recognizer.alignCrop(out, face)
                            feat = recognizer.feature(aligned).astype(np.float32)
                            best_name, best_score, second_score = _best_match(recognizer, feat, known) if known else (None, -1.0, -1.0)
                            confident = (
                                best_name is not None
                                and (best_score >= 0.70)
                                and ((best_score - second_score) >= 0.06)
                            )
                            if confident:
                                label = best_name
                                color = (0, 255, 0)
                            if best_score >= 0:
                                score_txt = f" {best_score:.2f}"
                        except Exception:
                            pass

                        cv2.rectangle(out, (x, y), (x + fw, y + fh), color, 2)
                        cv2.putText(
                            out,
                            f"{label}{score_txt}",
                            (max(0, x), max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )
        except Exception:
            pass

        ok, buffer = cv2.imencode(".jpg", out)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


def capture_known_person_from_camera(person: str, cam_index: int = 0):
    person = sanitize_person_name(person)
    if not person:
        raise RuntimeError("Naam is ongeldig of leeg.")

    cam = get_camera(cam_index)
    ok, frame = cam.read()
    if not ok or frame is None:
        raise RuntimeError("Kan geen frame lezen van camera.")

    with _face_lock:
        detector, recognizer = _get_face_models()
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        face = _largest_face(faces)
        if face is None:
            raise RuntimeError("Geen gezicht gedetecteerd.")

        x, y, fw, fh = face[:4].astype(int)
        pad_w = int(fw * 0.15)
        pad_h = int(fh * 0.15)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + fw + pad_w)
        y2 = min(h, y + fh + pad_h)

        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            raise RuntimeError("Kon geen geldige crop maken.")

        feat = None
        try:
            aligned = recognizer.alignCrop(frame, face)
            feat = recognizer.feature(aligned).astype(np.float32)
        except Exception:
            feat = None

    person_dir = os.path.join(KNOWN_DIR, person)
    os.makedirs(person_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = unique_path(os.path.join(person_dir, f"{ts}_camera.jpg"))
    cv2.imwrite(out_path, crop)

    added = 0
    if feat is not None and feat.ndim == 1 and feat.shape[0] > 0:
        added = append_features_for_person(person, [feat])

    return out_path, added


def _largest_face(faces: np.ndarray):
    if faces is None or len(faces) == 0:
        return None
    areas = faces[:, 2] * faces[:, 3]
    return faces[int(np.argmax(areas))]


def _get_face_models():
    global _face_detector, _face_recognizer
    if _face_detector is not None and _face_recognizer is not None:
        return _face_detector, _face_recognizer

    if not os.path.exists(YUNET_PATH):
        raise RuntimeError(f"Model ontbreekt: {YUNET_PATH}")
    if not os.path.exists(SFACE_PATH):
        raise RuntimeError(f"Model ontbreekt: {SFACE_PATH}")

    _face_detector = cv2.FaceDetectorYN.create(
        YUNET_PATH, "", (320, 320), 0.9, 0.3, 5000
    )
    _face_recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
    return _face_detector, _face_recognizer


def get_known_embeddings_cached(refresh_sec: float = 5.0):
    global _known_cache, _known_cache_at
    now = time.time()
    if _known_cache and (now - _known_cache_at) < refresh_sec:
        return _known_cache
    _known_cache = load_known_embeddings()
    _known_cache_at = now
    return _known_cache


def _best_match(recognizer, feat: np.ndarray, known: dict):
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


def _annotate_face(img, face_row, label: str, color):
    x, y, w, h = face_row[:4].astype(int)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        img,
        label,
        (max(0, x), max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def annotate_uploaded_image(img: np.ndarray, threshold: float = 0.50, margin: float = 0.06, min_face: int = 80):
    known = load_known_embeddings()
    print(f"Annotating image with {len(known)} known people loaded.")
    if not known:
        raise RuntimeError("Geen bekende personen in known/. Voeg eerst .npz profielen toe.")

    with _face_lock:
        detector, recognizer = _get_face_models()
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)

        annotated = img.copy()
        results = []
        if faces is None or len(faces) == 0:
            return annotated, results

        for idx, face in enumerate(faces):
            x, y, fw, fh = face[:4].astype(int)

            if fw < min_face:
                label = f"SMALL ({fw}px)"
                _annotate_face(annotated, face, label, (0, 255, 255))
                results.append({
                    "idx": idx + 1,
                    "label": "TOO_SMALL",
                    "score": "",
                    "x": x, "y": y, "w": fw, "h": fh,
                })
                continue

            try:
                aligned = recognizer.alignCrop(img, face)
                feat = recognizer.feature(aligned).astype(np.float32)
            except Exception:
                _annotate_face(annotated, face, "FEATURE_FAIL", (0, 0, 255))
                results.append({
                    "idx": idx + 1,
                    "label": "FEATURE_FAIL",
                    "score": "",
                    "x": x, "y": y, "w": fw, "h": fh,
                })
                continue

            best_name, best_score, second_score = _best_match(recognizer, feat, known)
            confident = (best_score >= threshold) and ((best_score - second_score) >= margin)

            if confident and best_name:
                label = f"{best_name} ({best_score:.2f})"
                _annotate_face(annotated, face, label, (0, 255, 0))
                results.append({
                    "idx": idx + 1,
                    "label": best_name,
                    "score": f"{best_score:.4f}",
                    "x": x, "y": y, "w": fw, "h": fh,
                })
            else:
                label = f"UNKNOWN ({best_score:.2f})"
                _annotate_face(annotated, face, label, (0, 0, 255))
                results.append({
                    "idx": idx + 1,
                    "label": "UNKNOWN",
                    "score": f"{best_score:.4f}",
                    "x": x, "y": y, "w": fw, "h": fh,
                })

    return annotated, results


def extract_features_from_unknown_folder(folder_path: str, min_face: int = 80):
    files = iter_image_files(folder_path)
    feats = []
    skipped = 0

    with _face_lock:
        detector, recognizer = _get_face_models()
        for fn in files:
            img_path = os.path.join(folder_path, fn)
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            face = _largest_face(faces)
            if face is None:
                skipped += 1
                continue

            fw = int(face[2])
            if fw < min_face:
                skipped += 1
                continue

            try:
                aligned = recognizer.alignCrop(img, face)
                feat = recognizer.feature(aligned).astype(np.float32)
            except Exception:
                skipped += 1
                continue

            feats.append(feat)

    return feats, len(files), skipped





# ---- Routes ----
@app.route("/")
def index():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    files = [f for f in os.listdir(SNAPSHOT_DIR) if f.lower().endswith((".jpg", ".jpeg"))]
    items = [parse_snapshot_filename(f) for f in files]
    items.sort(key=lambda x: (x["dt"] is not None, x["dt"] or datetime.min, x["filename"]), reverse=True)
    return render_template("index.html", items=items)

@app.route("/personen")
def personen():
    return render_template(
        "personen.html",
        running=recognition_running(),
        known_people=list_known_people()
    )

@app.route("/personen/start", methods=["POST"])
def personen_start():
    start_recognition()
    return redirect(url_for("personen"))

@app.route("/personen/stop", methods=["POST"])
def personen_stop():
    stop_recognition()
    return redirect(url_for("personen"))

@app.route("/api/personen/status")
def api_personen_status():
    return jsonify({
        "running": recognition_running(),
        "known_count": len(list_known_people())
    })

@app.route("/api/personen/log")
def api_personen_log():
    with _log_lock:
        return jsonify({"lines": list(_log_lines)})

@app.route("/snapshots/<path:filename>")
def snapshot_file(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)
    
@app.route("/camera")
def camera_page():
    return render_template(
        "camera.html",
        stream_on=is_stream_enabled(),
        msg=request.args.get("msg", ""),
        level=request.args.get("level", "info"),
    )

@app.route("/annotate-photo", methods=["GET", "POST"])
def annotate_photo_page():
    msg = ""
    level = "info"
    image_b64 = ""
    results = []
    filename = ""

    if request.method == "POST":
        up = request.files.get("photo")
        if up is None or not up.filename:
            msg = "Selecteer een foto."
            level = "error"
        elif not is_allowed_image_filename(up.filename):
            msg = "Bestandstype niet ondersteund. Gebruik jpg/jpeg/png/bmp/webp."
            level = "error"
        else:
            filename = os.path.basename(up.filename)
            data = up.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                msg = "Kon de afbeelding niet lezen."
                level = "error"
            else:
                try:
                    annotated, results = annotate_uploaded_image(img)
                    ok, enc = cv2.imencode(".jpg", annotated)
                    if not ok:
                        raise RuntimeError("Annotatie kon niet als JPG worden opgeslagen.")
                    image_b64 = base64.b64encode(enc.tobytes()).decode("ascii")
                    msg = f"Klaar: {len(results)} gezicht(en) verwerkt."
                    level = "ok"
                except Exception as e:
                    msg = f"Annotatie mislukt: {e}"
                    level = "error"

    return render_template(
        "annotate_photo.html",
        msg=msg,
        level=level,
        image_b64=image_b64,
        results=results,
        filename=filename,
    )


@app.route("/faces-extract", methods=["GET", "POST"])
def faces_extract_page():
    msg = request.args.get("msg", "")
    level = request.args.get("level", "info")
    manifest = None
    session = ""

    if request.method == "POST":
        uploads = request.files.getlist("photos")
        if not uploads:
            one = request.files.get("photo")
            if one is not None:
                uploads = [one]

        valid_uploads = [u for u in uploads if u is not None and u.filename]
        if not valid_uploads:
            msg = "Selecteer minstens 1 foto."
            level = "error"
        else:
            try:
                with _face_lock:
                    detector, recognizer = _get_face_models()
                    known = load_known_embeddings()
                    session = uuid.uuid4().hex
                    session_dir = os.path.join(FACE_EXTRACT_TMP_DIR, session)
                    os.makedirs(session_dir, exist_ok=True)

                    entries = []
                    source_files = []
                    face_id = 0

                    for up in valid_uploads:
                        src_name = os.path.basename(up.filename or "")
                        if not is_allowed_image_filename(src_name):
                            continue

                        data = up.read()
                        arr = np.frombuffer(data, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is None:
                            continue

                        source_files.append(src_name)
                        h, w = img.shape[:2]
                        detector.setInputSize((w, h))
                        _, faces = detector.detect(img)
                        if faces is None or len(faces) == 0:
                            continue

                        # Sorteer links->rechts voor voorspelbare volgorde.
                        face_rows = sorted(list(faces), key=lambda r: float(r[0]))
                        for face in face_rows:
                            x, y, fw, fh = face[:4].astype(int)
                            pad_w = int(fw * 0.15)
                            pad_h = int(fh * 0.15)
                            x1 = max(0, x - pad_w)
                            y1 = max(0, y - pad_h)
                            x2 = min(w, x + fw + pad_w)
                            y2 = min(h, y + fh + pad_h)

                            crop = img[y1:y2, x1:x2]
                            if crop is None or crop.size == 0:
                                continue

                            face_id += 1
                            img_name = f"face_{face_id:04d}.jpg"
                            img_path = os.path.join(session_dir, img_name)
                            cv2.imwrite(img_path, crop)

                            feat_name = None
                            suggested_name = ""
                            suggested_score = ""
                            try:
                                aligned = recognizer.alignCrop(img, face)
                                feat = recognizer.feature(aligned).astype(np.float32)
                                feat_name = f"face_{face_id:04d}.npy"
                                np.save(os.path.join(session_dir, feat_name), feat)
                                best_name, best_score, _ = _best_match(recognizer, feat, known) if known else (None, -1.0, -1.0)
                                if best_name:
                                    suggested_name = best_name
                                    suggested_score = f"{best_score:.4f}"
                            except Exception:
                                feat_name = None

                            entries.append({
                                "id": face_id,
                                "image": img_name,
                                "feature": feat_name,
                                "box": [int(x), int(y), int(fw), int(fh)],
                                "source_filename": src_name,
                                "suggested_name": suggested_name,
                                "suggested_score": suggested_score,
                            })

                    if not entries:
                        shutil.rmtree(session_dir, ignore_errors=True)
                        msg = "Geen bruikbare gezichten gevonden in de geselecteerde foto('s)."
                        level = "error"
                    else:
                        manifest = {
                            "session": session,
                            "source_files": source_files,
                            "entries": entries,
                        }
                        save_face_extract_manifest(session_dir, manifest)
                        msg = (
                            f"{len(entries)} gezicht(en) gevonden in "
                            f"{len(source_files)} foto('s). Vul per gezicht een naam in."
                        )
                        level = "ok"
            except Exception as e:
                msg = f"Verwerken mislukt: {e}"
                level = "error"

    return render_template(
        "faces_extract.html",
        msg=msg,
        level=level,
        manifest=manifest,
        session=session,
    )


@app.route("/faces-extract/tmp/<session>/<path:filename>")
def faces_extract_tmp_file(session, filename):
    session_dir = resolve_face_extract_session_dir(session)
    if session_dir is None or not os.path.isdir(session_dir):
        return ("", 404)
    return send_from_directory(session_dir, filename)


@app.route("/faces-extract/save", methods=["POST"])
def faces_extract_save():
    session = (request.form.get("session") or "").strip()
    session_dir = resolve_face_extract_session_dir(session)
    if session_dir is None or not os.path.isdir(session_dir):
        return redirect(url_for("faces_extract_page", level="error", msg="Sessie niet gevonden of verlopen."))

    manifest = load_face_extract_manifest(session_dir)
    if not manifest or not manifest.get("entries"):
        shutil.rmtree(session_dir, ignore_errors=True)
        return redirect(url_for("faces_extract_page", level="error", msg="Geen sessiegegevens gevonden."))

    by_person_features = {}
    saved_photos = 0
    saved_faces = 0
    skipped = 0

    try:
        for e in manifest["entries"]:
            rid = str(e.get("id"))
            raw_name = (request.form.get(f"name_{rid}") or "").strip()
            person = sanitize_person_name(raw_name)
            if not person:
                skipped += 1
                continue

            person_dir = os.path.join(KNOWN_DIR, person)
            os.makedirs(person_dir, exist_ok=True)

            src_img = os.path.join(session_dir, e.get("image", ""))
            if os.path.isfile(src_img):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                dst_img = os.path.join(person_dir, f"{ts}_{rid}.jpg")
                dst_img = unique_path(dst_img)
                shutil.copy2(src_img, dst_img)
                saved_photos += 1

            feat_file = e.get("feature")
            if feat_file:
                feat_path = os.path.join(session_dir, feat_file)
                if os.path.isfile(feat_path):
                    feat = np.load(feat_path).astype(np.float32)
                    if feat.ndim == 1 and feat.shape[0] > 0:
                        by_person_features.setdefault(person, []).append(feat)
                        saved_faces += 1

        for person, feats in by_person_features.items():
            append_features_for_person(person, feats)
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)

    msg = f"Opslaan klaar: {saved_photos} foto('s), {saved_faces} feature(s), {skipped} overgeslagen."
    return redirect(url_for("faces_extract_page", level="ok", msg=msg))


@app.route("/faces")
def unknown_page():
    return render_template(
        "unknown.html",
        people=list_known_people_with_photos(),
        sessions=list_unknown_sessions(),
        msg=request.args.get("msg", ""),
        level=request.args.get("level", "info"),
    )


@app.route("/unknown")
def unknown_legacy_redirect():
    return redirect(url_for("unknown_page"))


@app.route("/unknown/person/<person>")
def known_person_page(person):
    person_dir, npz_path = resolve_known_person_dir(person)
    if person_dir is None or not os.path.isfile(npz_path):
        return redirect(url_for("unknown_page", level="error", msg="Persoon bestaat niet (meer)."))

    files = iter_image_files(person_dir) if os.path.isdir(person_dir) else []
    return render_template(
        "known_person.html",
        person=person,
        files=files,
    )


@app.route("/known/<person>/<path:filename>")
def known_person_file(person, filename):
    person_dir, npz_path = resolve_known_person_dir(person)
    if person_dir is None or not os.path.isfile(npz_path):
        return ("", 404)
    return send_from_directory(person_dir, filename)


@app.route("/known/delete", methods=["POST"])
def known_delete():
    person = (request.form.get("person") or "").strip()
    person_dir, npz_path = resolve_known_person_dir(person)

    if person_dir is None:
        return redirect(url_for("unknown_page", level="error", msg="Ongeldige persoonsnaam."))

    exists_npz = os.path.isfile(npz_path)
    exists_dir = os.path.isdir(person_dir)
    if not exists_npz and not exists_dir:
        return redirect(url_for("unknown_page", level="error", msg=f"Persoon {person} bestaat niet (meer)."))

    try:
        if exists_npz:
            os.remove(npz_path)
        if exists_dir:
            shutil.rmtree(person_dir, ignore_errors=True)
    except Exception as e:
        return redirect(url_for("unknown_page", level="error", msg=f"Verwijderen van {person} mislukt: {e}"))

    return redirect(url_for("unknown_page", level="ok", msg=f"Persoon {person} verwijderd."))


@app.route("/known/reprocess", methods=["POST"])
def known_reprocess():
    person_dirs = list_known_photo_dirs()
    if not person_dirs:
        return redirect(url_for("unknown_page", level="error", msg="Geen mappen gevonden in known/."))

    processed = 0
    skipped = 0
    total_features = 0

    for person in person_dirs:
        folder = os.path.join(KNOWN_DIR, person)
        try:
            feats, total, _ = extract_features_from_unknown_folder(folder, min_face=80)
        except Exception:
            skipped += 1
            continue

        if total == 0 or len(feats) == 0:
            skipped += 1
            continue

        out_path = os.path.join(KNOWN_DIR, f"{person}.npz")
        np.savez_compressed(out_path, features=np.stack(feats, axis=0))
        processed += 1
        total_features += len(feats)

    msg = (
        f"Gezichten verwerkt: {processed} map(pen), {skipped} overgeslagen, "
        f"{total_features} feature(s) opgeslagen."
    )
    level = "ok" if processed > 0 else "error"
    return redirect(url_for("unknown_page", level=level, msg=msg))


@app.route("/unknown/enroll", methods=["POST"])
def unknown_enroll():
    session = (request.form.get("session") or "").strip()
    raw_name = (request.form.get("name") or "").strip()
    person = sanitize_person_name(raw_name)

    if not session:
        return redirect(url_for("unknown_page", level="error", msg="Geen map geselecteerd."))
    if not person:
        return redirect(url_for("unknown_page", level="error", msg="Naam is ongeldig of leeg."))

    session_dir = resolve_unknown_session_dir(session)
    if session_dir is None:
        return redirect(url_for("unknown_page", level="error", msg="Ongeldige mapnaam."))
    if not os.path.isdir(session_dir):
        return redirect(url_for("unknown_page", level="error", msg="Map bestaat niet meer."))

    try:
        feats, total, skipped = extract_features_from_unknown_folder(session_dir, min_face=80)
    except Exception as e:
        return redirect(url_for("unknown_page", level="error", msg=f"Fout bij verwerken: {e}"))

    if len(feats) < 3:
        return redirect(
            url_for(
                "unknown_page",
                level="error",
                msg=f"Te weinig bruikbare gezichten in {session} ({len(feats)}/{total}). Minimaal 3 nodig.",
            )
        )

    os.makedirs(KNOWN_DIR, exist_ok=True)
    out_path = os.path.join(KNOWN_DIR, f"{person}.npz")
    overwritten = os.path.exists(out_path)
    np.savez_compressed(out_path, features=np.stack(feats, axis=0))

    try:
        person_dir, moved_count = move_unknown_session_images_to_known(session_dir, person, session)
        shutil.rmtree(session_dir, ignore_errors=True)
    except Exception as e:
        return redirect(
            url_for(
                "unknown_page",
                level="error",
                msg=f".npz opgeslagen, maar foto's verplaatsen/verwijderen mislukte: {e}",
            )
        )

    msg = (
        f"{person} opgeslagen met {len(feats)} features (geskipt: {skipped}, totaal: {total}). "
        f"{moved_count} foto's verplaatst naar {person_dir}. Unknown map verwijderd."
    )
    if overwritten:
        msg = f"{person} overschreven. " + msg
    return redirect(url_for("unknown_page", level="ok", msg=msg))


@app.route("/unknown/delete", methods=["POST"])
def unknown_delete():
    session = (request.form.get("session") or "").strip()
    session_dir = resolve_unknown_session_dir(session)

    if session_dir is None:
        return redirect(url_for("unknown_page", level="error", msg="Ongeldige mapnaam."))
    if not os.path.isdir(session_dir):
        return redirect(url_for("unknown_page", level="error", msg="Map bestaat niet meer."))

    try:
        shutil.rmtree(session_dir)
    except Exception as e:
        return redirect(url_for("unknown_page", level="error", msg=f"Verwijderen mislukt: {e}"))

    return redirect(url_for("unknown_page", level="ok", msg=f"Map {session} verwijderd."))


@app.route("/unknown/upload-enroll", methods=["POST"])
def unknown_upload_enroll():
    raw_name = (request.form.get("name") or "").strip()
    person = sanitize_person_name(raw_name)
    if not person:
        return redirect(url_for("unknown_page", level="error", msg="Naam is ongeldig of leeg."))

    uploaded = request.files.getlist("photos")
    if not uploaded:
        return redirect(url_for("unknown_page", level="error", msg="Geen foto's geupload."))

    tmp_dir = os.path.join(BASE_DIR, "_tmp_uploads", uuid.uuid4().hex)
    os.makedirs(tmp_dir, exist_ok=True)
    saved = 0

    try:
        for f in uploaded:
            if not f or not f.filename:
                continue

            rel = f.filename.replace("\\", "/")
            rel = os.path.basename(rel)
            if not is_allowed_image_filename(rel):
                continue

            safe_name = sanitize_person_name(os.path.splitext(rel)[0]) or f"img_{saved+1:04d}"
            ext = os.path.splitext(rel)[1].lower()
            out_name = f"{saved+1:04d}_{safe_name}{ext}"
            out_path = os.path.join(tmp_dir, out_name)
            f.save(out_path)
            saved += 1

        if saved == 0:
            return redirect(url_for("unknown_page", level="error", msg="Geen geldige image-bestanden gevonden."))

        feats, total, skipped = extract_features_from_unknown_folder(tmp_dir, min_face=80)
    except Exception as e:
        return redirect(url_for("unknown_page", level="error", msg=f"Upload verwerken mislukt: {e}"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if len(feats) < 3:
        return redirect(
            url_for(
                "unknown_page",
                level="error",
                msg=f"Te weinig bruikbare gezichten uit upload ({len(feats)}/{total}). Minimaal 3 nodig.",
            )
        )

    os.makedirs(KNOWN_DIR, exist_ok=True)
    out_path = os.path.join(KNOWN_DIR, f"{person}.npz")
    overwritten = os.path.exists(out_path)
    np.savez_compressed(out_path, features=np.stack(feats, axis=0))

    msg = f"{person} opgeslagen vanuit upload met {len(feats)} features (geskipt: {skipped}, totaal: {total})."
    if overwritten:
        msg = f"{person} overschreven. " + msg
    return redirect(url_for("unknown_page", level="ok", msg=msg))

@app.route("/camera/start", methods=["POST"])
def camera_start():
    set_stream_enabled(True)
    return redirect(url_for("camera_page"))

@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    set_stream_enabled(False)
    release_camera()
    return redirect(url_for("camera_page"))


@app.route("/camera/capture", methods=["POST"])
def camera_capture():
    raw_name = (request.form.get("name") or "").strip()
    person = sanitize_person_name(raw_name)
    if not person:
        return redirect(url_for("camera_page", level="error", msg="Naam is ongeldig of leeg."))

    try:
        out_path, added = capture_known_person_from_camera(person, cam_index=0)
    except Exception as e:
        return redirect(url_for("camera_page", level="error", msg=f"Capture mislukt: {e}"))

    msg = f"Foto opgeslagen voor {person}: {out_path}"
    if added > 0:
        msg += f" | {added} feature toegevoegd aan {person}.npz"
    return redirect(url_for("camera_page", level="ok", msg=msg))

@app.route("/video_feed")
def video_feed():
    # Als stream niet aan staat: geen feed (voorkomt eindeloos reconnecten)
    if not is_stream_enabled():
        return ("", 204)  # No Content
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
