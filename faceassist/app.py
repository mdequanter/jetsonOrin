from flask import Flask, render_template, send_from_directory
import os
import re
from datetime import datetime

app = Flask(__name__)

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")

# Verwacht: Naam_yyyy_mm_dd_hh_mm_ss.jpg
FILENAME_RE = re.compile(r"^(?P<name>.+)_(?P<dt>\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})\.jpe?g$", re.IGNORECASE)

def parse_snapshot_filename(filename: str):
    m = FILENAME_RE.match(filename)
    if not m:
        return {
            "filename": filename,
            "name": "Onbekend",
            "dt": None,
            "dt_str": "Onbekende datum"
        }

    name = m.group("name")
    dt_raw = m.group("dt")
    try:
        dt = datetime.strptime(dt_raw, "%Y_%m_%d_%H_%M_%S")
        dt_str = dt.strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        dt = None
        dt_str = dt_raw.replace("_", ":")

    return {
        "filename": filename,
        "name": name,
        "dt": dt,
        "dt_str": dt_str
    }

@app.route("/")
def index():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    files = [
        f for f in os.listdir(SNAPSHOT_DIR)
        if f.lower().endswith((".jpg", ".jpeg"))
    ]

    items = [parse_snapshot_filename(f) for f in files]

    # Sorteer: nieuwste eerst (fallback: alfabetisch)
    items.sort(key=lambda x: (x["dt"] is not None, x["dt"] or datetime.min, x["filename"]), reverse=True)

    return render_template("index.html", items=items)

@app.route("/snapshots/<path:filename>")
def snapshot_file(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)

if __name__ == "__main__":
    # Voor development:
    app.run(host="0.0.0.0", port=5000, debug=True)
