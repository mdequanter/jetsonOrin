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
import math
import random
import subprocess
import threading
import time

from flask import Flask, jsonify, render_template_string, request

try:
    import cv2
    import numpy as np
except ImportError:           # zonder opencv/numpy werkt enkel de handbediening
    cv2 = None
    np = None

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD, VUI_COLOR

logging.basicConfig(level=logging.FATAL)

ROBOT_IP = "192.168.12.1"
WEB_HOST = "0.0.0.0"
WEB_PORT = 8080

# De Jetson afsluiten vanaf de webpagina. Dit vraagt rootrechten: draai het
# script als root, of geef de gebruiker een sudoregel zonder wachtwoord:
#   jetson ALL=(ALL) NOPASSWD: /sbin/shutdown
SHUTDOWN_COMMAND = ["sudo", "-n", "shutdown", "-h", "now"]

MOVE_SPEED = 0.5
TURN_SPEED = 1
BRIGHTNESS_LVL = 1

# Heading-bediening: 90 = recht vooruit, meer = naar rechts, minder = naar links
HEADING_FORWARD = 90.0
HEADING_MOVE_SPEED = 0.2      # traag vooruit terwijl de robot draait
HEADING_FORWARD_TIME = 1.0    # s rechtdoor stappen als de heading precies 90 is
HEADING_MAX_TURN = 180.0      # nooit meer dan een halve draai in één commando
TURN_INTERVAL = 0.1           # s tussen twee move-commando's tijdens het draaien
TURN_CALIBRATION = 1.0        # verhoog als de robot te weinig draait, verlaag als te veel

# Camera + segmentatie
USE_CAMERA = True             # zet op False om zonder camera/YOLO te draaien
MODEL_PATH = "/home/jetson/jetsonOrin/signaling/models/thuis.pt"
DETECTION_CONFIDENCE = 0.3
SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ALLOWED_PATH_LABELS = {"path", "path-oxod"}
FRAME_INTERVAL = 0.2          # s tussen twee frames die we bijhouden (5 fps volstaat)
SEGMENTATION_INTERVAL = 0.5   # s tussen twee berekeningen
HEADING_MAX_AGE = 3.0         # s waarna we een heading als verouderd beschouwen

# Het pad volgen zolang de vooruitknop ingedrukt blijft
FOLLOW_DEADBAND = 3.0         # graden verschil waarbinnen we niet bijsturen
FOLLOW_FULL_TURN = 45.0       # graden verschil waarbij we op volle draaisnelheid zitten
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
        self._disco_thread = None
        self._heading_lock = threading.Lock()

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

    def follow_path(self):
        """Vooruit stappen en meteen bijsturen naar de heading die de camera
        ziet, zodat de robot het pad volgt zolang dit commando binnenkomt.

        Ziet de camera niets bruikbaars, dan stappen we gewoon rechtdoor."""
        heading = segmentation.current_heading()
        if heading is None:
            return self.move(x=MOVE_SPEED)

        error = heading - HEADING_FORWARD
        if abs(error) < FOLLOW_DEADBAND:
            z = 0.0
        else:
            # Hoe verder van 90, hoe harder we draaien; positieve z is links
            z = -clamp(error / FOLLOW_FULL_TURN, -1.0, 1.0) * TURN_SPEED

        return self.move(x=MOVE_SPEED, z=z)

    # -- heading -------------------------------------------------------------

    def follow_heading(self, degrees):
        """Volg een heading: draai het gegeven aantal graden (positief =
        rechts, negatief = links) terwijl de robot traag vooruit gaat. Is de
        heading precies recht vooruit (0 graden verschil), dan stapt de robot
        HEADING_FORWARD_TIME seconden rechtdoor.

        De Go2 beweegt zolang hij move-commando's krijgt, dus we blijven
        herhalen tot de berekende tijd voorbij is en sturen daarna een stop.

        Geeft de hoek en de duur terug, of None als er al een beweging bezig
        is."""
        if not self._heading_lock.acquire(blocking=False):
            return None

        try:
            degrees = clamp(degrees, -HEADING_MAX_TURN, HEADING_MAX_TURN)

            if degrees == 0:
                # Recht vooruit: gewoon een seconde stappen, niet draaien
                duration = HEADING_FORWARD_TIME
                z = 0.0
            else:
                duration = math.radians(abs(degrees)) / TURN_SPEED * TURN_CALIBRATION
                # In de Go2 is een positieve z een draai naar links
                z = -TURN_SPEED if degrees > 0 else TURN_SPEED

            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                self.move(x=HEADING_MOVE_SPEED, z=z)
                time.sleep(TURN_INTERVAL)
            self.move(x=0, y=0, z=0)

            return {
                "degrees": round(degrees, 1),
                "forward": HEADING_MOVE_SPEED,
                "duration": round(duration, 2),
            }
        finally:
            self._heading_lock.release()

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


robot = RobotController()


# ---------------------------------------------------------- camera + segmentatie

def get_allowed_mask_indices(result, model_names):
    """Houd enkel de maskers over die een pad voorstellen."""
    if result.boxes is None or result.boxes.cls is None:
        return []

    allowed_indices = []
    class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()
    for index, class_id in enumerate(class_ids):
        label = str(model_names.get(class_id, "")).strip().lower()
        if label in ALLOWED_PATH_LABELS:
            allowed_indices.append(index)
    return allowed_indices


def compute_heading_to_point(frame, target_x, target_y):
    """Hoek van het midden onderaan het beeld naar een punt, in de conventie van
    deze pagina: 90 is recht vooruit, meer dan 90 is naar rechts.

    arctan2 telt tegen de klok in, dus een punt rechts van het midden geeft daar
    minder dan 90 graden (zo staat het ook in segmentAndControlGo2.py). Daarom
    spiegelen we het resultaat rond 90."""
    h, w = frame.shape[:2]
    start_x = w // 2
    start_y = h
    dx = target_x - start_x
    dy = start_y - target_y
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return 2 * HEADING_FORWARD - angle


def compute_heading(frame, model):
    """Zoek het pad in het beeld en geef de heading ernaartoe.

    Per masker van een pad kijken we op een aantal hoogtes waar het pad zit, en
    we mikken op het gemiddelde van die punten. Wordt er niets gevonden, dan
    geven we recht vooruit terug."""
    h, w = frame.shape[:2]
    result = model(frame, conf=DETECTION_CONFIDENCE, verbose=False)[0]
    model_names = getattr(model, "names", {})
    midpoints = []

    if result.masks is None or len(result.masks.data) == 0:
        return HEADING_FORWARD

    for mask_index in get_allowed_mask_indices(result, model_names):
        if mask_index >= len(result.masks.data):
            continue

        mask = result.masks.data[mask_index].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        for row_ratio in SCAN_HEIGHTS:
            y = int(h * row_ratio)
            if y >= h:
                continue
            filled_x = np.where(mask[y, :] > 0)[0]
            if len(filled_x) > 0:
                midpoints.append((int(np.mean(filled_x)), y))

    if not midpoints:
        return HEADING_FORWARD

    avg_x = int(np.mean([point[0] for point in midpoints]))
    target_y = min(point[1] for point in midpoints)
    return compute_heading_to_point(frame, avg_x, target_y)


class Segmentation:
    """Berekent voortdurend de heading uit de camerabeelden van de robot.

    Het model wordt in een achtergrondthread geladen, zodat de webpagina meteen
    bruikbaar is en niet op YOLO moet wachten."""

    def __init__(self):
        self._frame = None
        self._frame_lock = threading.Lock()
        self._last_frame_at = 0.0
        self.heading = None
        self.updated_at = 0.0
        self.status = "uit" if not USE_CAMERA else "wachten op beeld"
        self.error = None

    # -- beelden binnenkrijgen ----------------------------------------------

    def put_frame(self, image):
        """Bewaar enkel het laatste beeld; oudere beelden zijn toch achterhaald."""
        now = time.monotonic()
        if now - self._last_frame_at < FRAME_INTERVAL:
            return
        self._last_frame_at = now
        with self._frame_lock:
            self._frame = image

    def take_frame(self):
        with self._frame_lock:
            image, self._frame = self._frame, None
        return image

    # -- berekening ----------------------------------------------------------

    def start(self):
        if not USE_CAMERA:
            return
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        self.status = "model laden"
        try:
            if cv2 is None or np is None:
                raise RuntimeError("opencv of numpy ontbreekt")
            # Ultralytics importeren en het model laden duurt een tiental seconden
            from ultralytics import YOLO
            model = YOLO(MODEL_PATH, verbose=False)
        except Exception as exc:
            self.status = "fout"
            self.error = str(exc)
            print("Segmentatie niet beschikbaar: " + str(exc), flush=True)
            return

        self.status = "wachten op beeld"
        print("Model geladen: " + MODEL_PATH, flush=True)

        while True:
            image = self.take_frame()
            if image is None:
                time.sleep(0.05)
                continue

            try:
                heading = compute_heading(image, model)
            except Exception as exc:
                self.status = "fout"
                self.error = str(exc)
                time.sleep(SEGMENTATION_INTERVAL)
                continue

            self.heading = clamp(float(heading), 0.0, 180.0)
            self.updated_at = time.monotonic()
            self.status = "actief"
            self.error = None
            time.sleep(SEGMENTATION_INTERVAL)

    # -- uitlezen ------------------------------------------------------------

    def current_heading(self):
        """De laatste heading, of None als er nog geen of enkel een verouderde is."""
        if self.heading is None or not self.updated_at:
            return None
        if time.monotonic() - self.updated_at > HEADING_MAX_AGE:
            return None
        return self.heading

    def snapshot(self):
        """Toestand voor de webpagina."""
        heading = self.current_heading()
        age = time.monotonic() - self.updated_at if self.updated_at else None

        return {
            "status": self.status,
            "error": self.error,
            "heading": round(heading, 1) if heading is not None else None,
            "age": round(age, 1) if age is not None else None,
        }


segmentation = Segmentation()


# -------------------------------------------------------------- commandotabel

COMMANDS = {
    # de basis
    "forward":      lambda: robot.move(x=MOVE_SPEED),
    "follow":       lambda: robot.follow_path(),
    "backward":     lambda: robot.move(x=-MOVE_SPEED),
    "turn_left":    lambda: robot.move(z=TURN_SPEED),
    "turn_right":   lambda: robot.move(z=-TURN_SPEED),
    "strafe_left":  lambda: robot.move(y=MOVE_SPEED),
    "strafe_right": lambda: robot.move(y=-MOVE_SPEED),
    "stop":         lambda: robot.move(x=0, y=0, z=0),

    "stand_up":     lambda: robot.sport(SPORT_CMD["RecoveryStand"], {"data": False}),
    "stand_down":   lambda: robot.sport(SPORT_CMD["StandDown"]),

    # extra bewegingen
    "hello":        lambda: robot.sport(SPORT_CMD["Hello"]),
    "stretch":      lambda: robot.sport(SPORT_CMD["Stretch"], {"data": False}),
    "sit":          lambda: robot.sport(1009, {"data": False}),
    "rise_sit":     lambda: robot.sport(1010, {"data": False}),
    "scrape":       lambda: robot.sport(1029, {"data": False}),

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

    .danger {
      background: #3a2224;
      border-color: #6b2f33;
      color: #ff9ea1;
    }
    .danger .lbl { color: #ff9ea1; }
    .danger:active { background: var(--danger); }
    .danger:active .ico, .danger:active .lbl { color: #fff; }

    .swatch { min-height: 52px; }
    .swatch .ico { font-size: 22px; }

    .row { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: stretch; }
    .row button { width: auto; padding: 12px 22px; }
    input[type=number] {
      font: inherit;
      user-select: text;
      -webkit-user-select: text;
      font-size: 20px;
      text-align: center;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      min-height: 58px;
      width: 100%;
    }

    .check {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }
    .check input { width: 22px; height: 22px; accent-color: var(--accent); }
    .check b { color: var(--text); font-variant-numeric: tabular-nums; }

    .hint { font-size: 11px; color: var(--muted); margin: 8px 2px 0; }
    .hint code { background: var(--panel-2); padding: 1px 5px; border-radius: 5px; }
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
    <button data-cmd="forward" data-follow data-hold><span class="ico">⬆️</span><span class="lbl">vooruit</span></button>
    <button data-cmd="strafe_right" data-hold><span class="ico">➡️</span><span class="lbl">zijwaarts</span></button>

    <button data-cmd="turn_left" data-hold><span class="ico">↩️</span><span class="lbl">draai links</span></button>
    <button class="stop" data-cmd="stop"><span class="ico">⏹️</span><span class="lbl">stop</span></button>
    <button data-cmd="turn_right" data-hold><span class="ico">↪️</span><span class="lbl">draai rechts</span></button>

    <div></div>
    <button data-cmd="backward" data-hold><span class="ico">⬇️</span><span class="lbl">achteruit</span></button>
    <div></div>
  </div>
  <p class="hint">De robot beweegt zolang je de knop ingedrukt houdt. Staat
     "volg de camera" aan, dan blijft hij tijdens het vooruit stappen het pad volgen.</p>
</section>

<section>
  <h2>Heading</h2>
  <div class="row">
    <input id="heading" type="number" inputmode="decimal" step="1" value="90">
    <button id="heading-go"><span class="ico">🧭</span><span class="lbl">draai</span></button>
  </div>
  <label class="check">
    <input type="checkbox" id="auto-heading" checked>
    <span>volg de camera <b id="camera-heading">—</b></span>
  </label>
  <p class="hint">90 = één seconde rechtdoor stappen. Meer draait naar rechts,
     minder naar links, telkens traag vooruit al draaiend.
     Werkt ook via de URL: <code>/heading/?heading=101</code><br>
     Met "volg de camera" aan wordt dit veld bijgewerkt met wat de camera ziet
     en stuurt de vooruitknop zelf bij naar 90.</p>
</section>

<section>
  <h2>Houding</h2>
  <div class="grid cols-2">
    <button data-cmd="stand_up"><span class="ico">🦮</span><span class="lbl">opstaan</span></button>
    <button data-cmd="stand_down"><span class="ico">🛌</span><span class="lbl">neerliggen</span></button>
  </div>
</section>

<section>
  <h2>Bewegingen</h2>
  <div class="grid cols-3">
    <button data-cmd="hello"><span class="ico">👋</span><span class="lbl">hello</span></button>
    <button data-cmd="stretch"><span class="ico">🤸</span><span class="lbl">stretch</span></button>
    <button data-cmd="scrape"><span class="ico">🐾</span><span class="lbl">scrape</span></button>
    <button data-cmd="sit"><span class="ico">🪑</span><span class="lbl">zitten</span></button>
    <button data-cmd="rise_sit"><span class="ico">🚶</span><span class="lbl">rechtstaan</span></button>
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

<section>
  <h2>Systeem</h2>
  <button class="danger" id="shutdown"><span class="ico">⏻</span><span class="lbl">Jetson-Orin afsluiten</span></button>
  <p class="hint">Zet de Jetson uit. Daarna is deze pagina niet meer bereikbaar
     en moet je de Jetson met de hand opstarten.</p>
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

// heading: draai naar een hoek (90 = vooruit)
document.getElementById('heading-go').addEventListener('click', async () => {
  const value = document.getElementById('heading').value;
  setStatus('draaien naar ' + value + '...');
  try {
    const res = await fetch('/heading/?heading=' + encodeURIComponent(value), { method: 'POST' });
    const data = await res.json();
    if (data.ok) {
      setStatus(data.turned + '° ' + data.direction, 'ok');
    } else {
      setStatus(data.error || 'fout', 'err');
    }
  } catch (err) {
    setStatus('geen verbinding', 'err');
  }
});

// gewone knoppen: één commando per tik
document.querySelectorAll('button[data-cmd]:not([data-hold])').forEach(btn => {
  btn.addEventListener('click', () => send(btn.dataset.cmd));
});

// Jetson afsluiten, met bevestiging want dit legt alles stil
document.getElementById('shutdown').addEventListener('click', async () => {
  if (!confirm('De Jetson-Orin nu afsluiten?\n\nDe robot stopt en deze pagina valt weg.')) return;

  setStatus('afsluiten...');
  try {
    const res = await fetch('/shutdown', { method: 'POST' });
    const data = await res.json();
    setStatus(data.ok ? 'Jetson wordt afgesloten' : (data.error || 'fout'), data.ok ? 'ok' : 'err');
  } catch (err) {
    // Bij een snelle shutdown kan het antwoord wegvallen, dat is normaal
    setStatus('Jetson wordt afgesloten', 'ok');
  }
});

// Staat "volg de camera" aan, dan stuurt de vooruitknop het pad achterna
function commandFor(btn) {
  if (btn.dataset.follow !== undefined && autoHeading.checked) return 'follow';
  return btn.dataset.cmd;
}

// houd-knoppen: herhaal het commando tot je loslaat, daarna stoppen
document.querySelectorAll('button[data-hold]').forEach(btn => {
  let timer = null;

  const start = (ev) => {
    ev.preventDefault();
    if (timer) return;
    send(commandFor(btn));
    timer = setInterval(() => send(commandFor(btn)), HOLD_INTERVAL);
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

// verbindingstoestand + heading van de camera ophalen
const headingInput = document.getElementById('heading');
const autoHeading = document.getElementById('auto-heading');
const cameraHeading = document.getElementById('camera-heading');
const followButton = document.querySelector('button[data-follow]');

// Toon op de knop zelf of hij gewoon vooruit gaat of het pad volgt
function showForwardMode() {
  followButton.querySelector('.lbl').textContent =
    autoHeading.checked ? 'volg pad' : 'vooruit';
}
autoHeading.addEventListener('change', showForwardMode);
showForwardMode();

function showCamera(cam) {
  if (!cam) return;

  if (cam.heading === null || cam.heading === undefined) {
    cameraHeading.textContent = '(' + cam.status + ')';
    return;
  }

  cameraHeading.textContent = cam.heading.toFixed(1) + '°';

  // Niet overschrijven terwijl je zelf een waarde aan het typen bent
  if (autoHeading.checked && document.activeElement !== headingInput) {
    headingInput.value = cam.heading;
  }
}

async function poll() {
  try {
    const res = await fetch('/status');
    const data = await res.json();
    if (!data.connected) setStatus('robot niet verbonden', 'err');
    showCamera(data.camera);
  } catch (err) {
    setStatus('server onbereikbaar', 'err');
  }
}
poll();
setInterval(poll, 1000);
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
        "camera": segmentation.snapshot(),
    })


@app.route("/heading/", methods=["GET", "POST"])
@app.route("/heading", methods=["GET", "POST"])
def heading():
    """Draai de robot naar een heading, bv. http://<ip>:8080/heading/?heading=101

    Bij precies 90 stapt de robot een seconde rechtdoor. Meer dan 90 draait
    naar rechts, minder naar links: het verschil met 90 is de hoek die de robot
    draait, en ondertussen gaat hij traag vooruit."""
    raw = request.args.get("heading")
    if raw is None:
        return jsonify({"ok": False, "error": "parameter 'heading' ontbreekt"}), 400

    try:
        value = float(raw)
    except ValueError:
        return jsonify({"ok": False, "error": "heading moet een getal zijn: " + raw}), 400

    if not robot.connected:
        return jsonify({"ok": False, "error": "geen verbinding met de robot"}), 503

    error = value - HEADING_FORWARD

    try:
        result = robot.follow_heading(error)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    if result is None:
        return jsonify({"ok": False, "error": "er is al een beweging bezig"}), 409

    return jsonify({
        "ok": True,
        "heading": value,
        "turned": result["degrees"],
        "direction": "rechts" if result["degrees"] > 0 else ("links" if result["degrees"] < 0 else "vooruit"),
        "forward": result["forward"],
        "duration": result["duration"],
    })


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Zet de Jetson uit. Daarna is de bediening uiteraard weg."""
    try:
        # De robot niet laten doorlopen terwijl de besturing verdwijnt
        robot.move(x=0, y=0, z=0)
    except Exception:
        pass

    try:
        result = subprocess.run(SHUTDOWN_COMMAND, capture_output=True,
                                text=True, timeout=10)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    if result.returncode != 0:
        melding = (result.stderr or result.stdout or "").strip()
        return jsonify({
            "ok": False,
            "error": melding or ("afsluiten gaf foutcode %d" % result.returncode),
        }), 500

    print("Jetson wordt afgesloten", flush=True)
    return jsonify({"ok": True, "message": "Jetson wordt afgesloten"})


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

    async def recv_camera_stream(track):
        while True:
            frame = await track.recv()
            segmentation.put_frame(frame.to_ndarray(format="bgr24"))

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

        # Camerabeelden binnenhalen voor de segmentatie
        if USE_CAMERA:
            conn.video.switchVideoChannel(True)
            conn.video.add_track_callback(recv_camera_stream)
            print("Camerastream gestart", flush=True)

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

    # Het YOLO-model laden duurt even, dus dat gebeurt in de achtergrond
    segmentation.start()

    # Wacht tot de verbindingspoging klaar is, zodat de eerste kliks werken
    ready.wait(timeout=30)

    print("Open op je smartphone: http://<ip-van-deze-pc>:%d/" % WEB_PORT, flush=True)

    try:
        app.run(host=WEB_HOST, port=WEB_PORT, threaded=True, use_reloader=False)
    finally:
        loop.call_soon_threadsafe(loop.stop)


if __name__ == "__main__":
    main()
