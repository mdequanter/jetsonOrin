from flask import Flask, render_template, send_from_directory, jsonify, redirect, url_for, Response
import os
import re
from datetime import datetime
import subprocess
import threading
from collections import deque
import cv2
import time


app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
KNOWN_DIR = os.path.join(BASE_DIR, "known")

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


# ---- Camera streaming (MJPEG) ----
_camera = None
_camera_lock = threading.Lock()

def get_camera(cam_index=0):
    global _camera
    with _camera_lock:
        if _camera is None or not _camera.isOpened():
            _camera = cv2.VideoCapture(cam_index)
            # Optioneel: zet resolutie
            _camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return _camera

def gen_frames():
    cam = get_camera(0)
    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.05)
            continue

        # JPEG encode
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()

        # MJPEG chunk
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")




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
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
