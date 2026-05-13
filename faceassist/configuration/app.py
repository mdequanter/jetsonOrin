from flask import Flask, redirect, render_template, request, url_for
from collections import deque
import json
import os
import shutil
import subprocess
import sys
import threading
import time


APP_DIR = os.path.dirname(os.path.abspath(__file__))
FACEASSIST_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
SETTINGS_PATH = os.path.join(FACEASSIST_DIR, "settings.json")
KNOWN_DIR = os.path.join(FACEASSIST_DIR, "known")
RECOGNITION_SCRIPT = os.path.join(FACEASSIST_DIR, "nl_launchv2.py")

SERVICE_NAME = os.environ.get("FACEASSIST_SERVICE", "faceassist.service")
CONFIG_HOST = os.environ.get("CONFIGURATION_HOST", "0.0.0.0")
CONFIG_PORT = int(os.environ.get("CONFIGURATION_PORT", "5050"))
DEFAULT_DROIDCAM_URL = os.environ.get("MOBILE_VIEW_DROIDCAM_URL", "http://192.168.0.55:4747/video")

app = Flask(__name__)

_recognition_proc = None
_recognition_source = "local"
_recognition_lock = threading.Lock()
_log_lines = deque(maxlen=400)
_log_lock = threading.Lock()


def _coerce_voice_volume(value, default_value=100):
    try:
        volume = int(value)
    except Exception:
        volume = int(default_value)
    return max(0, min(100, volume))


def _default_voice_volume():
    return _coerce_voice_volume(os.environ.get("VOICE_VOLUME", "100"), 100)


def _default_settings():
    return {
        "droidcam_url": DEFAULT_DROIDCAM_URL,
        "voice_volume": _default_voice_volume(),
    }


def load_settings():
    defaults = _default_settings()
    if not os.path.isfile(SETTINGS_PATH):
        return defaults

    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings = json.load(f)
    except Exception:
        return defaults

    if not isinstance(settings, dict):
        return defaults

    merged = dict(defaults)
    merged.update(settings)
    merged["droidcam_url"] = str(merged.get("droidcam_url", defaults["droidcam_url"])).strip() or defaults["droidcam_url"]
    settings["voice_volume"] = _coerce_voice_volume(
        settings.get("voice_volume", _default_voice_volume()),
        _default_voice_volume(),
    )
    merged["voice_volume"] = settings["voice_volume"]
    return merged


def save_voice_volume(volume):
    settings = load_settings()
    settings["voice_volume"] = _coerce_voice_volume(
        volume,
        settings.get("voice_volume", _default_voice_volume()),
    )

    tmp_path = f"{SETTINGS_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, SETTINGS_PATH)
    return settings["voice_volume"]


def _run_command(cmd, timeout=12):
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return False, f"{cmd[0]} not found"
    except subprocess.TimeoutExpired:
        return False, "command timed out"
    except Exception as exc:
        return False, str(exc)

    output = (result.stdout or result.stderr or "").strip()
    if result.returncode == 0:
        return True, output
    return False, output or f"exit code {result.returncode}"


def _run_first_success(cmds, timeout=12):
    errors = []
    for cmd in cmds:
        ok, message = _run_command(cmd, timeout=timeout)
        if ok:
            return True, message
        errors.append(f"{' '.join(cmd)}: {message}")
    return False, " | ".join(errors)


def _systemctl_action_commands(action):
    return [
        ["sudo", "-n", "systemctl", action, SERVICE_NAME],
        ["systemctl", action, SERVICE_NAME],
    ]


def _system_action_commands(action):
    return [
        ["sudo", "-n", "systemctl", action],
        ["systemctl", action],
    ]


def _run_system_action_later(cmds):
    def _worker():
        time.sleep(1.0)
        _run_first_success(cmds, timeout=30)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def _append_log(line):
    with _log_lock:
        _log_lines.append(str(line).rstrip("\n"))


def get_recognition_log():
    with _log_lock:
        return list(_log_lines)


def _reader_thread(proc):
    try:
        for line in proc.stdout:
            if not line:
                break
            _append_log(line)
    except Exception as exc:
        _append_log(f"[LOG-ERROR] {exc}")
    finally:
        global _recognition_proc
        _append_log("[INFO] nl_launchv2.py stopped.")
        with _recognition_lock:
            if _recognition_proc is proc:
                _recognition_proc = None


def _recognition_running_unlocked():
    return _recognition_proc is not None and _recognition_proc.poll() is None


def recognition_running():
    with _recognition_lock:
        return _recognition_running_unlocked()


def get_recognition_source():
    return _recognition_source


def _normalize_droidcam_url(url):
    value = (url or "").strip()
    if not value:
        return ""
    if value.endswith("/video") or value.endswith("/mjpegfeed"):
        return value
    if value.endswith("/"):
        return value + "video"
    return value + "/video"


def start_recognition(source="local"):
    global _recognition_proc, _recognition_source

    source = (source or "").strip().lower()
    if source not in ("local", "droidcam"):
        source = "local"

    with _recognition_lock:
        if _recognition_running_unlocked():
            return False, "nl_launchv2.py is already running."

        if not os.path.isfile(RECOGNITION_SCRIPT):
            return False, f"Script not found: {RECOGNITION_SCRIPT}"

        settings = load_settings()
        os.makedirs(KNOWN_DIR, exist_ok=True)

        cmd = [
            sys.executable,
            RECOGNITION_SCRIPT,
            "--known",
            KNOWN_DIR,
            "--voice_volume",
            str(settings["voice_volume"]),
        ]

        if source == "droidcam":
            droidcam_url = _normalize_droidcam_url(settings.get("droidcam_url", ""))
            if not droidcam_url:
                return False, "DroidCam URL is empty."
            cmd.extend(["--cam_url", droidcam_url])

        _recognition_source = source
        _append_log("[INFO] Starting nl_launchv2.py.")
        _append_log(f"[INFO] Source: {'DroidCam' if source == 'droidcam' else 'Local camera'}")
        _append_log("[INFO] CMD: " + " ".join(cmd))

        try:
            _recognition_proc = subprocess.Popen(
                cmd,
                cwd=FACEASSIST_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            _recognition_proc = None
            _append_log(f"[ERROR] Start failed: {exc}")
            return False, str(exc)

        thread = threading.Thread(target=_reader_thread, args=(_recognition_proc,), daemon=True)
        thread.start()
        return True, "nl_launchv2.py started."


def stop_recognition():
    global _recognition_proc
    with _recognition_lock:
        if not _recognition_running_unlocked():
            _recognition_proc = None
            return False, "nl_launchv2.py is not running."

        try:
            _append_log("[INFO] Stop signal sent to nl_launchv2.py.")
            _recognition_proc.terminate()
        except Exception as exc:
            _append_log(f"[ERROR] Stop failed: {exc}")
            return False, str(exc)

        return True, "Stop signal sent to nl_launchv2.py."


def service_status():
    status = {
        "service_name": SERVICE_NAME,
        "systemctl_available": shutil.which("systemctl") is not None,
        "load_state": "unknown",
        "active_state": "unknown",
        "sub_state": "unknown",
        "unit_file_state": "unknown",
        "main_pid": "",
        "fragment_path": "",
        "error": "",
    }

    ok, output = _run_command(
        [
            "systemctl",
            "show",
            SERVICE_NAME,
            "--property=LoadState,ActiveState,SubState,UnitFileState,MainPID,FragmentPath",
            "--no-page",
        ],
        timeout=5,
    )
    if not ok:
        status["error"] = output
        return status

    keys = {
        "LoadState": "load_state",
        "ActiveState": "active_state",
        "SubState": "sub_state",
        "UnitFileState": "unit_file_state",
        "MainPID": "main_pid",
        "FragmentPath": "fragment_path",
    }
    for line in output.splitlines():
        key, sep, value = line.partition("=")
        if sep and key in keys:
            status[keys[key]] = value.strip() or "unknown"

    return status


def _redirect_with(message, level="info"):
    return redirect(url_for("control_page", msg=message, level=level))


@app.route("/")
def control_page():
    settings = load_settings()
    return render_template(
        "control.html",
        title="Face Assist Configuration",
        msg=request.args.get("msg", ""),
        level=request.args.get("level", "info"),
        settings=settings,
        status=service_status(),
        recognition_running=recognition_running(),
        recognition_source=get_recognition_source(),
        recognition_script=RECOGNITION_SCRIPT,
        recognition_log=get_recognition_log(),
        settings_path=SETTINGS_PATH,
    )


@app.route("/service/start", methods=["POST"])
def start_service():
    ok, message = _run_first_success(_systemctl_action_commands("start"), timeout=30)
    if ok:
        return _redirect_with(f"{SERVICE_NAME} start requested.", "ok")
    return _redirect_with(f"Start failed: {message}", "error")


@app.route("/service/stop", methods=["POST"])
def stop_service():
    ok, message = _run_first_success(_systemctl_action_commands("stop"), timeout=30)
    if ok:
        return _redirect_with(f"{SERVICE_NAME} stop requested.", "ok")
    return _redirect_with(f"Stop failed: {message}", "error")


@app.route("/nl-launch/start", methods=["POST"])
def start_nl_launch():
    source = (request.form.get("source") or get_recognition_source()).strip().lower()
    ok, message = start_recognition(source=source)
    return _redirect_with(message, "ok" if ok else "error")


@app.route("/nl-launch/stop", methods=["POST"])
def stop_nl_launch():
    ok, message = stop_recognition()
    return _redirect_with(message, "ok" if ok else "info")


@app.route("/api/nl-launch/log")
def api_nl_launch_log():
    return {"lines": get_recognition_log()}


@app.route("/volume", methods=["POST"])
def set_volume():
    try:
        volume = save_voice_volume(request.form.get("voice_volume"))
    except Exception as exc:
        return _redirect_with(f"Volume save failed: {exc}", "error")
    return _redirect_with(f"Voice volume saved at {volume}.", "ok")


@app.route("/reboot", methods=["POST"])
def reboot_system():
    _run_system_action_later(_system_action_commands("reboot"))
    return _redirect_with("Jetson reboot requested.", "ok")


if __name__ == "__main__":
    app.run(host=CONFIG_HOST, port=CONFIG_PORT, debug=False)
