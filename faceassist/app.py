from flask import Flask, render_template, send_from_directory, jsonify, redirect, url_for, Response, request
import os
import re
import shutil
from datetime import datetime
import subprocess
import threading
from collections import deque
import cv2
import time
import numpy as np


app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
KNOWN_DIR = os.path.join(BASE_DIR, "known")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown")
MODELS_DIR = os.path.join(BASE_DIR, "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

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


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


def sanitize_person_name(name: str) -> str:
    cleaned = SAFE_NAME_RE.sub("_", (name or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned


def list_unknown_sessions():
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    sessions = []
    for folder in os.listdir(UNKNOWN_DIR):
        full = os.path.join(UNKNOWN_DIR, folder)
        if not os.path.isdir(full):
            continue
        files = iter_image_files(full)
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


_camera = None
_camera_lock = threading.Lock()
_stream_enabled = False
_stream_lock = threading.Lock()
_face_lock = threading.Lock()
_face_detector = None
_face_recognizer = None

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

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


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
    return render_template("camera.html", stream_on=is_stream_enabled())


@app.route("/unknown")
def unknown_page():
    return render_template(
        "unknown.html",
        sessions=list_unknown_sessions(),
        msg=request.args.get("msg", ""),
        level=request.args.get("level", "info"),
    )


@app.route("/unknown/session/<session>")
def unknown_session_page(session):
    session_dir = resolve_unknown_session_dir(session)
    if session_dir is None or not os.path.isdir(session_dir):
        return redirect(url_for("unknown_page", level="error", msg="Map bestaat niet (meer)."))

    _, dt_str = parse_unknown_session_name(session)
    files = iter_image_files(session_dir)
    return render_template(
        "unknown_session.html",
        session=session,
        dt_str=dt_str,
        files=files,
    )


@app.route("/unknown/<session>/<path:filename>")
def unknown_file(session, filename):
    return send_from_directory(os.path.join(UNKNOWN_DIR, session), filename)


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

    msg = f"{person} opgeslagen met {len(feats)} features (geskipt: {skipped}, totaal: {total})."
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

@app.route("/camera/start", methods=["POST"])
def camera_start():
    set_stream_enabled(True)
    return redirect(url_for("camera_page"))

@app.route("/camera/stop", methods=["POST"])
def camera_stop():
    set_stream_enabled(False)
    release_camera()
    return redirect(url_for("camera_page"))

@app.route("/video_feed")
def video_feed():
    # Als stream niet aan staat: geen feed (voorkomt eindeloos reconnecten)
    if not is_stream_enabled():
        return ("", 204)  # No Content
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
