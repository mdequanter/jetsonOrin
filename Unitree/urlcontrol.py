"""
Webbediening voor de Unitree Go2.

Zelfde commando's als keyboard2.py, maar bediend via een webpagina die
geoptimaliseerd is voor een smartphone. Start dit script op de Jetson en
surf met je gsm naar http://<ip-van-de-jetson>:8080/

    python3 urlcontrol.py
"""

import asyncio
import json
import logging
import random
import threading
import time

from flask import Flask, jsonify, render_template_string

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD, VUI_COLOR

logging.basicConfig(level=logging.FATAL)

ROBOT_IP = "192.168.12.1"
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080

MOVE_SPEED = 0.5
TURN_SPEED = 1
EULER_STEP = 0.1
EULER_LIMIT = 0.4
BRIGHTNESS_LVL = 1
DISCO_COLOURS = [VUI_COLOR.RED, VUI_COLOR.YELLOW, VUI_COLOR.GREEN,
                 VUI_COLOR.CYAN, VUI_COLOR.BLUE, VUI_COLOR.PURPLE]


# ---------------------------------------------------------------- robot state

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


class RobotController:
    """Stuurt commando's naar de Go2 vanuit de Flask threads."""

    def __init__(self):
        self.conn = None
        self.loop = None
        self.connected = False
        self.euler_roll = 0.0
        self.euler_pitch = 0.0
        self.euler_yaw = 0.0
        self._lock = threading.Lock()
        self._disco_thread = None

    # -- laag niveau ---------------------------------------------------------

    def publish(self, topic, payload):
        """Publiceer een bericht op de datachannel (thread-safe)."""
        if not self.connected:
            raise RuntimeError("Geen verbinding met de robot")
        return asyncio.run_coroutine_threadsafe(
            self.conn.datachannel.pub_sub.publish_request_new(topic, payload),
            self.loop
        )

    def sport(self, api_id, parameter=None):
        payload = {"api_id": api_id}
        if parameter is not None:
            payload["parameter"] = parameter
        return self.publish(RTC_TOPIC["SPORT_MOD"], payload)

    def vui(self, api_id, parameter):
        return self.publish(RTC_TOPIC["VUI"], {"api_id": api_id, "parameter": parameter})

    # -- bewegen -------------------------------------------------------------

    def move(self, x=0, y=0, z=0):
        return self.sport(SPORT_CMD["Move"], {"x": x, "y": y, "z": z})

    def send_euler(self):
        return self.sport(SPORT_CMD["Euler"], {
            "x": self.euler_roll,
            "y": self.euler_pitch,
            "z": self.euler_yaw,
        })

    def nudge_euler(self, roll=0.0, pitch=0.0, yaw=0.0):
        with self._lock:
            self.euler_roll = clamp(self.euler_roll + roll, -EULER_LIMIT, EULER_LIMIT)
            self.euler_pitch = clamp(self.euler_pitch + pitch, -EULER_LIMIT, EULER_LIMIT)
            self.euler_yaw = clamp(self.euler_yaw + yaw, -EULER_LIMIT, EULER_LIMIT)
        return self.send_euler()

    def reset_euler(self):
        with self._lock:
            self.euler_roll = 0.0
            self.euler_pitch = 0.0
            self.euler_yaw = 0.0
        return self.send_euler()

    # -- licht ---------------------------------------------------------------

    def colour(self, colour):
        return self.vui(1007, {"color": colour})

    def disco(self):
        """Laat de kleuren wisselen in een aparte thread, zodat de webpagina
        niet blijft wachten."""
        if self._disco_thread and self._disco_thread.is_alive():
            return

        def run():
            for _ in range(30):
                try:
                    self.colour(random.choice(DISCO_COLOURS))
                except Exception:
                    break
                time.sleep(0.2)

        self._disco_thread = threading.Thread(target=run, daemon=True)
        self._disco_thread.start()

    # -- noodstop ------------------------------------------------------------

    def emergency_stop(self):
        self.move(x=0, y=0, z=0)
        self.reset_euler()


robot = RobotController()


# -------------------------------------------------------------- commandotabel

COMMANDS = {
    # de basis
    "forward":      lambda: robot.move(x=MOVE_SPEED),
    "backward":     lambda: robot.move(x=-MOVE_SPEED),
    "turn_left":    lambda: robot.move(z=TURN_SPEED),
    "turn_right":   lambda: robot.move(z=-TURN_SPEED),
    "strafe_left":  lambda: robot.move(y=MOVE_SPEED),
    "strafe_right": lambda: robot.move(y=-MOVE_SPEED),
    "stop":         lambda: robot.move(x=0, y=0, z=0),

    "stand_up":     lambda: robot.sport(SPORT_CMD["RecoveryStand"], {"data": False}),
    "stand_down":   lambda: robot.sport(SPORT_CMD["StandDown"]),

    # pose / euler
    "roll_left":    lambda: robot.nudge_euler(roll=EULER_STEP),
    "roll_right":   lambda: robot.nudge_euler(roll=-EULER_STEP),
    "pitch_front":  lambda: robot.nudge_euler(pitch=EULER_STEP),
    "pitch_back":   lambda: robot.nudge_euler(pitch=-EULER_STEP),
    "yaw_left":     lambda: robot.nudge_euler(yaw=EULER_STEP),
    "yaw_right":    lambda: robot.nudge_euler(yaw=-EULER_STEP),
    "pose_reset":   lambda: robot.reset_euler(),

    # extra bewegingen
    "hello":        lambda: robot.sport(SPORT_CMD["Hello"]),
    "stretch":      lambda: robot.sport(SPORT_CMD["Stretch"], {"data": False}),
    "sit":          lambda: robot.sport(1009, {"data": False}),
    "rise_sit":     lambda: robot.sport(1010, {"data": False}),
    "scrape":       lambda: robot.sport(1029, {"data": False}),
    "front_jump":   lambda: robot.sport(1031, {"data": False}),

    # licht
    "torch_on":     lambda: robot.vui(1005, {"brightness": BRIGHTNESS_LVL}),
    "torch_off":    lambda: robot.vui(1005, {"brightness": 0}),
    "white":        lambda: robot.colour(VUI_COLOR.WHITE),
    "red":          lambda: robot.colour(VUI_COLOR.RED),
    "yellow":       lambda: robot.colour(VUI_COLOR.YELLOW),
    "green":        lambda: robot.colour(VUI_COLOR.GREEN),
    "cyan":         lambda: robot.colour(VUI_COLOR.CYAN),
    "blue":         lambda: robot.colour(VUI_COLOR.BLUE),
    "purple":       lambda: robot.colour(VUI_COLOR.PURPLE),
    "disco":        lambda: robot.disco(),
    "flash":        lambda: robot.vui(1007, {"color": VUI_COLOR.WHITE, "flash_cycle": 500}),

    # noodstop
    "estop":        lambda: robot.emergency_stop(),
}


# ------------------------------------------------------------------ webpagina

HTML_PAGE = """
<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
  <meta name="theme-color" content="#111418">
  <meta name="mobile-web-app-capable" content="yes">
  <title>Go2 bediening</title>
  <style>
    :root {
      --bg: #111418;
      --panel: #1c2128;
      --panel-2: #262d36;
      --line: #333b45;
      --text: #f2f4f7;
      --muted: #9aa4b2;
      --accent: #2f81f7;
      --danger: #e5484d;
      --ok: #2ea043;
    }
    * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
    body {
      margin: 0;
      padding: 12px 12px calc(12px + env(safe-area-inset-bottom));
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      font-size: 15px;
      user-select: none;
      -webkit-user-select: none;
      touch-action: manipulation;
      overscroll-behavior: none;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 12px;
    }
    h1 { font-size: 17px; margin: 0; }
    .status {
      font-size: 12px;
      color: var(--muted);
      display: flex;
      align-items: center;
      gap: 6px;
      min-width: 0;
    }
    #msg { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .dot {
      width: 9px; height: 9px; border-radius: 50%;
      background: var(--muted);
      flex: none;
    }
    .dot.ok { background: var(--ok); }
    .dot.err { background: var(--danger); }

    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      margin-bottom: 12px;
    }
    h2 {
      font-size: 12px;
      letter-spacing: .08em;
      text-transform: uppercase;
      color: var(--muted);
      margin: 0 0 10px;
    }

    button {
      font: inherit;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 6px;
      min-height: 58px;
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 3px;
      cursor: pointer;
      transition: transform .05s ease, background .1s ease;
    }
    button .ico { font-size: 20px; line-height: 1; }
    button .lbl { font-size: 12px; color: var(--muted); text-align: center; }
    button:active { background: var(--accent); transform: scale(.96); }
    button:active .lbl { color: #fff; }

    .grid { display: grid; gap: 8px; }
    .cols-2 { grid-template-columns: repeat(2, 1fr); }
    .cols-3 { grid-template-columns: repeat(3, 1fr); }
    .cols-4 { grid-template-columns: repeat(4, 1fr); }

    .dpad { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .dpad button { min-height: 66px; }
    .dpad .stop { background: #3a2224; border-color: #5a2c30; }

    .estop {
      background: var(--danger);
      border-color: var(--danger);
      min-height: 62px;
      font-weight: 700;
      letter-spacing: .04em;
    }
    .estop .lbl { color: #ffe9ea; font-size: 15px; font-weight: 700; }
    .estop:active { background: #ff5a5f; }

    .swatch { min-height: 52px; }
    .swatch .ico { font-size: 22px; }

    .hint { font-size: 11px; color: var(--muted); margin: 8px 2px 0; }
    footer { text-align: center; color: var(--muted); font-size: 11px; padding: 4px 0 8px; }
  </style>
</head>
<body>

<header>
  <h1>🐕 Go2 bediening</h1>
  <div class="status"><span class="dot" id="dot"></span><span id="msg">klaar</span></div>
</header>

<section>
  <h2>Noodstop</h2>
  <button class="estop" data-cmd="estop"><span class="lbl">⛔ EMERGENCY STOP</span></button>
</section>

<section>
  <h2>Bewegen — houd ingedrukt</h2>
  <div class="dpad">
    <button data-cmd="strafe_left" data-hold><span class="ico">⬅️</span><span class="lbl">zijwaarts</span></button>
    <button data-cmd="forward" data-hold><span class="ico">⬆️</span><span class="lbl">vooruit</span></button>
    <button data-cmd="strafe_right" data-hold><span class="ico">➡️</span><span class="lbl">zijwaarts</span></button>

    <button data-cmd="turn_left" data-hold><span class="ico">↩️</span><span class="lbl">draai links</span></button>
    <button class="stop" data-cmd="stop"><span class="ico">⏹️</span><span class="lbl">stop</span></button>
    <button data-cmd="turn_right" data-hold><span class="ico">↪️</span><span class="lbl">draai rechts</span></button>

    <div></div>
    <button data-cmd="backward" data-hold><span class="ico">⬇️</span><span class="lbl">achteruit</span></button>
    <div></div>
  </div>
  <p class="hint">De robot beweegt zolang je de knop ingedrukt houdt.</p>
</section>

<section>
  <h2>Houding</h2>
  <div class="grid cols-2">
    <button data-cmd="stand_up"><span class="ico">🦮</span><span class="lbl">opstaan</span></button>
    <button data-cmd="stand_down"><span class="ico">🛌</span><span class="lbl">neerliggen</span></button>
  </div>
</section>

<section>
  <h2>Pose (euler)</h2>
  <div class="grid cols-3">
    <button data-cmd="roll_left"><span class="ico">↺</span><span class="lbl">roll links</span></button>
    <button data-cmd="pitch_front"><span class="ico">⤵️</span><span class="lbl">pitch voor</span></button>
    <button data-cmd="yaw_left"><span class="ico">◀</span><span class="lbl">yaw links</span></button>

    <button data-cmd="roll_right"><span class="ico">↻</span><span class="lbl">roll rechts</span></button>
    <button data-cmd="pitch_back"><span class="ico">⤴️</span><span class="lbl">pitch achter</span></button>
    <button data-cmd="yaw_right"><span class="ico">▶</span><span class="lbl">yaw rechts</span></button>
  </div>
  <div class="grid" style="margin-top:8px">
    <button data-cmd="pose_reset"><span class="ico">🎯</span><span class="lbl">pose reset</span></button>
  </div>
</section>

<section>
  <h2>Bewegingen</h2>
  <div class="grid cols-3">
    <button data-cmd="hello"><span class="ico">👋</span><span class="lbl">hello</span></button>
    <button data-cmd="stretch"><span class="ico">🤸</span><span class="lbl">stretch</span></button>
    <button data-cmd="sit"><span class="ico">🪑</span><span class="lbl">zitten</span></button>
    <button data-cmd="rise_sit"><span class="ico">🚶</span><span class="lbl">rechtstaan</span></button>
    <button data-cmd="scrape"><span class="ico">🐾</span><span class="lbl">scrape</span></button>
    <button data-cmd="front_jump"><span class="ico">🤾</span><span class="lbl">front jump</span></button>
  </div>
</section>

<section>
  <h2>Zaklamp</h2>
  <div class="grid cols-2">
    <button data-cmd="torch_on"><span class="ico">🔦</span><span class="lbl">aan</span></button>
    <button data-cmd="torch_off"><span class="ico">🌑</span><span class="lbl">uit</span></button>
  </div>
</section>

<section>
  <h2>Kleuren</h2>
  <div class="grid cols-4">
    <button class="swatch" data-cmd="white"><span class="ico">🤍</span></button>
    <button class="swatch" data-cmd="red"><span class="ico">❤️</span></button>
    <button class="swatch" data-cmd="yellow"><span class="ico">💛</span></button>
    <button class="swatch" data-cmd="green"><span class="ico">💚</span></button>
    <button class="swatch" data-cmd="cyan"><span class="ico">🩵</span></button>
    <button class="swatch" data-cmd="blue"><span class="ico">💙</span></button>
    <button class="swatch" data-cmd="purple"><span class="ico">💜</span></button>
    <button class="swatch" data-cmd="disco"><span class="ico">🪩</span></button>
  </div>
  <div class="grid" style="margin-top:8px">
    <button data-cmd="flash"><span class="ico">⚡</span><span class="lbl">flash</span></button>
  </div>
</section>

<footer>Robot: {{ robot_ip }}</footer>

<script>
const dot = document.getElementById('dot');
const msg = document.getElementById('msg');
const HOLD_INTERVAL = 150;   // ms tussen twee move-commando's tijdens indrukken

function setStatus(text, state) {
  msg.textContent = text;
  dot.className = 'dot' + (state ? ' ' + state : '');
}

async function send(cmd) {
  try {
    const res = await fetch('/cmd/' + cmd, { method: 'POST' });
    const data = await res.json();
    setStatus(data.ok ? cmd : (data.error || 'fout'), data.ok ? 'ok' : 'err');
  } catch (err) {
    setStatus('geen verbinding', 'err');
  }
}

// gewone knoppen: één commando per tik
document.querySelectorAll('button[data-cmd]:not([data-hold])').forEach(btn => {
  btn.addEventListener('click', () => send(btn.dataset.cmd));
});

// houd-knoppen: herhaal het commando tot je loslaat, daarna stoppen
document.querySelectorAll('button[data-hold]').forEach(btn => {
  let timer = null;

  const start = (ev) => {
    ev.preventDefault();
    if (timer) return;
    send(btn.dataset.cmd);
    timer = setInterval(() => send(btn.dataset.cmd), HOLD_INTERVAL);
  };

  const end = () => {
    if (!timer) return;
    clearInterval(timer);
    timer = null;
    send('stop');
  };

  btn.addEventListener('pointerdown', start);
  btn.addEventListener('pointerup', end);
  btn.addEventListener('pointercancel', end);
  btn.addEventListener('pointerleave', end);
  btn.addEventListener('contextmenu', ev => ev.preventDefault());
});

// stoppen wanneer de pagina naar de achtergrond gaat of je wegsurft
document.addEventListener('visibilitychange', () => {
  if (document.hidden) send('stop');
});
window.addEventListener('pagehide', () => send('stop'));

// verbindingstoestand ophalen
async function poll() {
  try {
    const res = await fetch('/status');
    const data = await res.json();
    if (!data.connected) setStatus('robot niet verbonden', 'err');
  } catch (err) {
    setStatus('server onbereikbaar', 'err');
  }
}
poll();
setInterval(poll, 5000);
</script>
</body>
</html>
"""


app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(HTML_PAGE, robot_ip=ROBOT_IP)


@app.route("/status")
def status():
    return jsonify({
        "connected": robot.connected,
        "euler": {
            "roll": round(robot.euler_roll, 2),
            "pitch": round(robot.euler_pitch, 2),
            "yaw": round(robot.euler_yaw, 2),
        },
    })


@app.route("/commands")
def commands():
    """Handig om te zien welke commando's er bestaan."""
    return jsonify(sorted(COMMANDS))


@app.route("/cmd/<command>", methods=["POST", "GET"])
def cmd(command):
    action = COMMANDS.get(command)
    if action is None:
        return jsonify({"ok": False, "error": "onbekend commando: " + command}), 404

    if not robot.connected:
        return jsonify({"ok": False, "error": "geen verbinding met de robot"}), 503

    try:
        action()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    return jsonify({"ok": True, "command": command})


# ---------------------------------------------------------------------- opstart

def run_asyncio_loop(loop, ready):
    asyncio.set_event_loop(loop)

    async def setup():
        conn = UnitreeWebRTCConnection(
            WebRTCConnectionMethod.LocalSTA,
            ip=ROBOT_IP
        )
        await conn.connect()

        # Zorg dat de robot in normale modus staat
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1001}
        )
        data = json.loads(response["data"]["data"])
        if data["name"] != "normal":
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": "normal"}}
            )
            await asyncio.sleep(5)

        robot.conn = conn
        robot.loop = loop
        robot.connected = True
        print("Verbonden met de Go2", flush=True)

    try:
        loop.run_until_complete(setup())
    except Exception as exc:
        print("Verbinden mislukt: " + str(exc), flush=True)
    finally:
        ready.set()

    loop.run_forever()


def main():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    asyncio_thread = threading.Thread(
        target=run_asyncio_loop, args=(loop, ready), daemon=True)
    asyncio_thread.start()

    # Wacht tot de verbindingspoging klaar is, zodat de eerste kliks werken
    ready.wait(timeout=30)

    print("Open op je smartphone: http://<ip-van-deze-pc>:%d/" % WEB_PORT, flush=True)

    try:
        app.run(host=WEB_HOST, port=WEB_PORT, threaded=True, use_reloader=False)
    finally:
        loop.call_soon_threadsafe(loop.stop)


if __name__ == "__main__":
    main()
