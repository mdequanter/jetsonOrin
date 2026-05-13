from flask import Flask, redirect, render_template, request, url_for
import json
import os
import shutil
import subprocess
import threading
import time


APP_DIR = os.path.dirname(os.path.abspath(__file__))
FACEASSIST_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
SETTINGS_PATH = os.path.join(FACEASSIST_DIR, "settings.json")

SERVICE_NAME = os.environ.get("FACEASSIST_SERVICE", "faceassist.service")
CONFIG_HOST = os.environ.get("CONFIGURATION_HOST", "0.0.0.0")
CONFIG_PORT = int(os.environ.get("CONFIGURATION_PORT", "5050"))

app = Flask(__name__)


def _coerce_voice_volume(value, default_value=100):
    try:
        volume = int(value)
    except Exception:
        volume = int(default_value)
    return max(0, min(100, volume))


def _default_voice_volume():
    return _coerce_voice_volume(os.environ.get("VOICE_VOLUME", "100"), 100)


def load_settings():
    if not os.path.isfile(SETTINGS_PATH):
        return {"voice_volume": _default_voice_volume()}

    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            settings = json.load(f)
    except Exception:
        return {"voice_volume": _default_voice_volume()}

    if not isinstance(settings, dict):
        return {"voice_volume": _default_voice_volume()}

    settings["voice_volume"] = _coerce_voice_volume(
        settings.get("voice_volume", _default_voice_volume()),
        _default_voice_volume(),
    )
    return settings


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


def _main_app_url():
    host = (request.host or "localhost").split(":", 1)[0]
    return f"http://{host}:5000/"


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
        main_app_url=_main_app_url(),
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
