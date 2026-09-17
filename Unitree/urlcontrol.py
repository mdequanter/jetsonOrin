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
import os
import random
import threading
import time

from flask import Flask, Response, jsonify, render_template_string, request

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
MODEL_PATH = "/home/jetson/jetsonOrin/signaling/models/unrealsim.pt"
DETECTION_CONFIDENCE = 0.8    # startwaarde, op de webpagina aanpasbaar
CONFIDENCE_CHOICES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
MODEL_DIR = os.path.dirname(MODEL_PATH)   # hier zoeken we de andere .pt-bestanden
ALLOWED_PATH_LABELS = {"path", "path-oxod"}
FRAME_INTERVAL = 0.2          # s tussen twee frames die we bijhouden (5 fps volstaat)
SEGMENTATION_INTERVAL = 0.3   # s tussen twee berekeningen
HEADING_MAX_AGE = 3.0         # s waarna we een heading als verouderd beschouwen

# Live beeld op de webpagina (/video)
STREAM_SIZE = (640, 480)      # het beeld wordt hierin gepast, met zwarte randen
STREAM_QUALITY = 70           # JPEG-kwaliteit van de stream
OVERLAY_MAX_AGE = 2.0         # s dat we het getekende beeld blijven tonen
STREAM_IDLE_TIMEOUT = 1.0     # s wachten op een nieuw beeld voor we iets sturen
MASK_COLOR = (0, 255, 0)      # BGR: het pad
MASK_ALPHA = 0.35             # hoe hard het masker het beeld inkleurt
MIDPOINT_COLOR = (0, 200, 255)
HEADING_COLOR = (0, 255, 255)

# ArUco: we tonen enkel de grootste marker in beeld
ARUCO_DICTIONARY = "DICT_4X4_50"
ARUCO_COLOR = (255, 0, 255)   # BGR: magenta kader
ARUCO_LABEL_SCALE = 2.1       # groot genoeg om van wat verder af te lezen
ARUCO_LABEL_THICKNESS = 3

# IJkpunten om de afstand te schatten: (oppervlakte in pixels, afstand in meter)
ARUCO_CALIBRATION = [(2100, 1.00), (25000, 0.30)]

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

def create_aruco_detector(dictionary_name):
    """Maak een detector die werkt met zowel de oude als de nieuwe OpenCV-API
    (zelfde opzet als in arucoContro.py)."""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("deze OpenCV heeft geen cv2.aruco (opencv-contrib-python)")

    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise RuntimeError("onbekende ArUco-dictionary: " + dictionary_name)

    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    else:
        dictionary = cv2.aruco.Dictionary_get(dictionary_id)

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return lambda image: detector.detectMarkers(image)

    return lambda image: cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)


_aruco_detector = None
_aruco_tried = False


def detect_largest_marker(frame):
    """Zoek de ArUco-markers in het beeld en houd enkel de grootste over.

    De grootste is doorgaans ook de dichtste, en één marker tegelijk houdt het
    beeld rustig. Geeft een dict met de id, de hoekpunten en de oppervlakte
    terug, of None als er niets in beeld is."""
    global _aruco_detector, _aruco_tried

    if not _aruco_tried:
        _aruco_tried = True
        try:
            _aruco_detector = create_aruco_detector(ARUCO_DICTIONARY)
        except Exception as exc:
            print("ArUco niet beschikbaar: " + str(exc), flush=True)

    if _aruco_detector is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = _aruco_detector(gray)
    if ids is None or len(ids) == 0:
        return None

    best = None
    for marker_id, marker_corners in zip(ids, corners):
        points = np.round(marker_corners.reshape((4, 2))).astype(np.int32)
        area = float(cv2.contourArea(points))
        if best is None or area > best["area"]:
            best = {"id": int(marker_id[0]), "points": points, "area": area}
    return best


def marker_distance(area):
    """Schat de afstand tot de marker uit zijn oppervlakte in pixels.

    Een marker die twee keer zo ver staat is half zo breed, en beslaat dus een
    kwart van de oppervlakte. De zijde (de wortel van de oppervlakte) is met
    andere woorden omgekeerd evenredig met de afstand, dus interpoleren we
    rechtlijnig in 1/wortel(oppervlakte) door de twee ijkpunten.

    Geeft meters terug, of None als er niets te rekenen valt."""
    (area_a, distance_a), (area_b, distance_b) = ARUCO_CALIBRATION
    if area <= 0 or area_a <= 0 or area_b <= 0 or area_a == area_b:
        return None

    x = 1.0 / math.sqrt(area)
    x_a, x_b = 1.0 / math.sqrt(area_a), 1.0 / math.sqrt(area_b)
    slope = (distance_a - distance_b) / (x_a - x_b)
    return max(0.0, distance_b + slope * (x - x_b))


def draw_marker(frame, marker):
    """Teken een kader rond de marker, met zijn nummer en oppervlakte erboven."""
    h, w = frame.shape[:2]
    points = marker["points"]
    cv2.polylines(frame, [points.reshape((-1, 1, 2))], True, ARUCO_COLOR, 3, cv2.LINE_AA)

    distance = marker_distance(marker["area"])
    label = "aruco %d" % marker["id"]
    if distance is not None:
        label += " - %.2f m" % distance
    (text_w, text_h), _baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, ARUCO_LABEL_SCALE, ARUCO_LABEL_THICKNESS)

    # Net boven het kader, maar altijd binnen het beeld
    x = clamp(int(points[:, 0].min()), 5, max(5, w - text_w - 5))
    y = clamp(int(points[:, 1].min()) - 10, text_h + 5, h - 5)

    cv2.putText(frame, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                ARUCO_LABEL_SCALE, ARUCO_COLOR, ARUCO_LABEL_THICKNESS, cv2.LINE_AA)


def available_models():
    """De .pt-bestanden die naast het ingestelde model staan."""
    try:
        names = [name for name in os.listdir(MODEL_DIR or ".")
                 if name.lower().endswith(".pt")]
    except OSError:
        names = []
    return sorted(names)


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


def largest_allowed_mask(result, model_names, shape):
    """Zoek van alle maskers die een pad voorstellen het grootste.

    We werken bewust met één vlak: een tweede pad, een zijweggetje of een
    los stukje berm zou de heading anders doen verspringen.

    Geeft de index en het masker op beeldformaat terug, of (None, None)."""
    h, w = shape
    best_index, best_mask, best_area = None, None, 0

    for index in get_allowed_mask_indices(result, model_names):
        if index >= len(result.masks.data):
            continue

        mask = result.masks.data[index].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        area = int(np.count_nonzero(mask))
        if area > best_area:
            best_index, best_mask, best_area = index, mask, area

    return best_index, best_mask


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


def draw_path_overlay(frame, result, mask_index, midpoints, heading):
    """Teken het grootste pad, de meetpunten en de heading op het beeld."""
    h, w = frame.shape[:2]
    masks_xy = getattr(result.masks, "xy", None) if result.masks is not None else None
    polygon = None

    if masks_xy is not None and mask_index is not None and mask_index < len(masks_xy):
        polygon = masks_xy[mask_index]

    if polygon is not None and len(polygon) >= 3:
        points = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))
        points[:, 0, 0] = np.clip(points[:, 0, 0], 0, w - 1)
        points[:, 0, 1] = np.clip(points[:, 0, 1], 0, h - 1)

        shaded = frame.copy()
        cv2.fillPoly(shaded, [points], MASK_COLOR)
        cv2.addWeighted(shaded, MASK_ALPHA, frame, 1.0 - MASK_ALPHA, 0, dst=frame)
        cv2.polylines(frame, [points], True, MASK_COLOR, 2, cv2.LINE_AA)

    for x, y in midpoints:
        cv2.circle(frame, (x, y), 4, MIDPOINT_COLOR, -1, cv2.LINE_AA)

    if midpoints:
        avg_x = int(np.mean([point[0] for point in midpoints]))
        target_y = min(point[1] for point in midpoints)
        cv2.arrowedLine(frame, (w // 2, h - 1), (avg_x, target_y),
                        HEADING_COLOR, 3, cv2.LINE_AA, tipLength=0.08)

    text = "geen pad" if heading is None else "heading: %.1f" % heading
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                HEADING_COLOR, 2, cv2.LINE_AA)


def segment_frame(frame, model, confidence):
    """Zoek het pad in het beeld en geef de heading ernaartoe, samen met een
    kopie van het beeld waarop het masker getekend staat (voor de live stream).

    We houden enkel het grootste pad over.  Daarvan kijken we op een aantal
    hoogtes waar het zit, en we mikken op het gemiddelde van die punten. Wordt
    er niets gevonden, dan geven we recht vooruit terug.  Hoe lager de
    confidence, hoe sneller het model iets een pad noemt."""
    h, w = frame.shape[:2]
    result = model(frame, conf=confidence, verbose=False)[0]
    model_names = getattr(model, "names", {})
    midpoints = []
    mask_index, mask = None, None

    if result.masks is not None and len(result.masks.data) > 0:
        mask_index, mask = largest_allowed_mask(result, model_names, (h, w))

    if mask is not None:
        for row_ratio in SCAN_HEIGHTS:
            y = int(h * row_ratio)
            if y >= h:
                continue
            filled_x = np.where(mask[y, :] > 0)[0]
            if len(filled_x) > 0:
                midpoints.append((int(np.mean(filled_x)), y))

    if midpoints:
        avg_x = int(np.mean([point[0] for point in midpoints]))
        target_y = min(point[1] for point in midpoints)
        heading = compute_heading_to_point(frame, avg_x, target_y)
    else:
        heading = HEADING_FORWARD

    overlay = frame.copy()
    draw_path_overlay(overlay, result, mask_index, midpoints,
                      heading if midpoints else None)
    return heading, overlay


def fit_to_stream_size(image):
    """Schaal het beeld naar STREAM_SIZE en vul de rest zwart op, zodat de
    stream altijd even groot is en de verhoudingen kloppen."""
    width, height = STREAM_SIZE
    h, w = image.shape[:2]
    if (w, h) == STREAM_SIZE:
        return image

    scale = min(width / w, height / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - new_w) // 2, (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def placeholder_frame(text):
    """Zwart beeld met een boodschap, zolang er geen camerabeeld binnenkomt."""
    width, height = STREAM_SIZE
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(image, str(text)[:40], (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (160, 160, 160), 2, cv2.LINE_AA)
    return image


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
        self.confidence = DETECTION_CONFIDENCE
        self.model_name = os.path.basename(MODEL_PATH)
        self.marker = None

        # Voor de live stream op /video
        self._stream_cond = threading.Condition()
        self._stream_image = None
        self._stream_seq = 0
        self._overlay_at = 0.0

    # -- beelden binnenkrijgen ----------------------------------------------

    def put_frame(self, image):
        """Bewaar enkel het laatste beeld; oudere beelden zijn toch achterhaald."""
        self._publish_stream(image, annotated=False)

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

    # -- live stream ---------------------------------------------------------

    def _publish_stream(self, image, annotated):
        """Zet een beeld klaar voor /video.

        Zolang het model beelden aflevert tonen we die met het masker erop; het
        kale camerabeeld dient enkel als terugval, bijvoorbeeld terwijl YOLO nog
        aan het laden is of wanneer de segmentatie stilvalt."""
        now = time.monotonic()
        with self._stream_cond:
            if annotated:
                self._overlay_at = now
            elif now - self._overlay_at < OVERLAY_MAX_AGE:
                return
            self._stream_image = image
            self._stream_seq += 1
            self._stream_cond.notify_all()

    def stream_frames(self):
        """Blijf JPEG's opleveren voor de MJPEG-stream van /video."""
        last_seq = -1
        while True:
            with self._stream_cond:
                if self._stream_seq == last_seq:
                    self._stream_cond.wait(timeout=STREAM_IDLE_TIMEOUT)
                image = None
                if self._stream_seq != last_seq:
                    image, last_seq = self._stream_image, self._stream_seq

            # Niets nieuws? Toch iets sturen: zo merken we dat de browser weg is
            if image is None:
                image = placeholder_frame(self.error or self.status)

            ok, buffer = cv2.imencode(
                ".jpg", fit_to_stream_size(image),
                [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_QUALITY])
            if ok:
                yield buffer.tobytes()

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
            # Ultralytics importeren duurt een tiental seconden
            from ultralytics import YOLO
        except Exception as exc:
            self.status = "fout"
            self.error = str(exc)
            print("Segmentatie niet beschikbaar: " + str(exc), flush=True)
            return

        model = None
        loaded_name = None

        while True:
            # Een ander model gekozen op de webpagina? Dan laden we dat nu
            wanted = self.model_name
            if wanted != loaded_name:
                self.status = "model laden"
                loaded_name = wanted
                try:
                    model = YOLO(os.path.join(MODEL_DIR, wanted), verbose=False)
                    self.status = "wachten op beeld"
                    self.error = None
                    print("Model geladen: " + wanted, flush=True)
                except Exception as exc:
                    model = None
                    self.status = "fout"
                    self.error = str(exc)
                    print("Model laden mislukt: " + str(exc), flush=True)

            # Zonder model wachten we tot er een ander gekozen wordt
            if model is None:
                time.sleep(SEGMENTATION_INTERVAL)
                continue

            image = self.take_frame()
            if image is None:
                time.sleep(0.05)
                continue

            try:
                heading, overlay = segment_frame(image, model, self.confidence)

                # De grootste ArUco-marker krijgt een kader op hetzelfde beeld
                marker = detect_largest_marker(image)
                if marker is not None:
                    draw_marker(overlay, marker)
                    distance = marker_distance(marker["area"])
                    self.marker = {
                        "id": marker["id"],
                        "distance": round(distance, 2) if distance is not None else None,
                        "area": int(round(marker["area"])),
                        "size": int(round(math.sqrt(marker["area"]))),
                    }
                else:
                    self.marker = None
            except Exception as exc:
                self.status = "fout"
                self.error = str(exc)
                time.sleep(SEGMENTATION_INTERVAL)
                continue

            self._publish_stream(overlay, annotated=True)
            self.heading = clamp(float(heading), 0.0, 180.0)
            self.updated_at = time.monotonic()
            self.status = "actief"
            self.error = None
            time.sleep(SEGMENTATION_INTERVAL)

    # -- uitlezen ------------------------------------------------------------

    def model_choices(self):
        """Alle modellen om uit te kiezen, met het huidige er zeker bij."""
        return sorted(set(available_models()) | {self.model_name})

    def set_model(self, name):
        """Kies een ander .pt-bestand. De achtergrondthread laadt het daarna.

        Enkel bestandsnamen uit MODEL_DIR zijn toegelaten, zo kan er via de URL
        geen ander bestand van de Jetson geopend worden."""
        name = os.path.basename(str(name))
        if name not in available_models():
            raise ValueError("onbekend model: " + name)

        if name != self.model_name:
            self.model_name = name
            # De oude heading hoort bij het oude model, dus die gooien we weg
            self.heading = None
            self.updated_at = 0.0
        return self.model_name

    def set_confidence(self, value):
        """Zet de drempel waarboven het model een masker meetelt."""
        value = clamp(float(value), min(CONFIDENCE_CHOICES), max(CONFIDENCE_CHOICES))
        self.confidence = round(value, 2)
        return self.confidence

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
            "confidence": round(self.confidence, 2),
            "model": self.model_name,
            "marker": self.marker,
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
    .check input[type=checkbox] { width: 22px; height: 22px; accent-color: var(--accent); }
    .check select {
      max-width: 60%;
      font: inherit;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 8px;
      min-height: 34px;
      margin-left: auto;
    }
    .check b { color: var(--text); font-variant-numeric: tabular-nums; }

    .camera {
      position: relative;
      width: 100%;
      max-width: 640px;        /* 640x480, op een gsm schaalt het mee naar beneden */
      aspect-ratio: 4 / 3;
      margin: 0 auto;
      background: #000;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
    }
    .camera img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .camera .badge {
      position: absolute;
      left: 8px; bottom: 8px;
      background: rgba(17, 20, 24, .75);
      border-radius: 8px;
      padding: 3px 8px;
      font-size: 11px;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }

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
  <h2>Camera</h2>
  <div class="camera">
    <img id="cam" alt="camerabeeld">
    <span class="badge" id="cam-badge">—</span>
  </div>
  <label class="check">
    <input type="checkbox" id="show-camera" checked>
    <span>toon het beeld</span>
  </label>
  <label class="check">
    <span>model</span>
    <select id="model">
      {% for name in model_choices %}
      <option value="{{ name }}"{% if name == model %} selected{% endif %}>{{ name }}</option>
      {% endfor %}
    </select>
  </label>
  <label class="check">
    <span>confidence</span>
    <select id="confidence">
      {% for value in confidence_choices %}
      <option value="{{ value }}"{% if value == confidence %} selected{% endif %}>{{ "%.1f"|format(value) }}</option>
      {% endfor %}
    </select>
  </label>
  <p class="hint">Het groene vlak is het grootste pad dat het model herkent, de stippen zijn
     de meetpunten en de pijl wijst naar de heading. Ligt er een ArUco-marker in
     beeld, dan krijgt de grootste een magenta kader met zijn nummer en de
     geschatte afstand, berekend uit zijn oppervlakte
     ({{ calibration }}). Zet het beeld uit als de
     verbinding traag wordt. Los te bekijken via <code>/video</code>.<br>
     Een lagere confidence laat het model sneller een pad zien (maar ook meer
     verkeerde), een hogere enkel wat het zeker weet.
     Werkt ook via de URL: <code>/confidence/?value=0.4</code><br>
     De keuzelijst toont de <code>.pt</code>-bestanden uit de modelmap. Een ander
     model laden duurt een tiental seconden, ondertussen staat de status op
     "model laden". Werkt ook via <code>/model/?name=denham.pt</code></p>
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

// live beeld met het masker erop: gewoon een MJPEG-stream in een <img>
const camImg = document.getElementById('cam');
const camBadge = document.getElementById('cam-badge');
const showCameraBox = document.getElementById('show-camera');

function startStream() {
  if (!showCameraBox.checked || document.hidden) return;
  if (camImg.getAttribute('src')) return;
  camImg.src = '/video?t=' + Date.now();   // nieuwe url, anders hergebruikt de browser de oude stream
}

function stopStream() {
  if (!camImg.getAttribute('src')) return;
  camImg.removeAttribute('src');           // sluit de verbinding met /video
}

showCameraBox.addEventListener('change', () => {
  if (showCameraBox.checked) startStream(); else stopStream();
});

// De stream niet laten doorlopen als de pagina toch niet zichtbaar is
document.addEventListener('visibilitychange', () => {
  if (document.hidden) stopStream(); else startStream();
});
// het model kiezen: de .pt-bestanden uit de modelmap
const modelBox = document.getElementById('model');

modelBox.addEventListener('change', async () => {
  setStatus('model laden...');
  try {
    const res = await fetch('/model/?name=' + encodeURIComponent(modelBox.value), { method: 'POST' });
    const data = await res.json();
    setStatus(data.ok ? data.model : (data.error || 'fout'), data.ok ? 'ok' : 'err');
  } catch (err) {
    setStatus('geen verbinding', 'err');
  }
});

// de drempel waarboven het model een masker meetelt
const confidenceBox = document.getElementById('confidence');

confidenceBox.addEventListener('change', async () => {
  const value = confidenceBox.value;
  try {
    const res = await fetch('/confidence/?value=' + encodeURIComponent(value), { method: 'POST' });
    const data = await res.json();
    setStatus(data.ok ? 'confidence ' + data.confidence : (data.error || 'fout'),
              data.ok ? 'ok' : 'err');
  } catch (err) {
    setStatus('geen verbinding', 'err');
  }
});

camImg.addEventListener('error', () => {
  stopStream();
  setTimeout(startStream, 2000);        // opnieuw proberen, bv. na een herstart
});
startStream();

function showCamera(cam) {
  if (!cam) return;

  let badge = cam.heading === null || cam.heading === undefined
    ? cam.status
    : cam.status + ' · ' + cam.heading.toFixed(1) + '°';
  if (cam.marker) {
    badge += ' · aruco ' + cam.marker.id;
    if (cam.marker.distance !== null) badge += ' op ' + cam.marker.distance.toFixed(2) + ' m';
  }
  camBadge.textContent = badge;

  // De keuzelijsten gelijk houden met de server, bv. na een herstart of een URL
  if (cam.confidence !== undefined && document.activeElement !== confidenceBox) {
    confidenceBox.value = cam.confidence.toFixed(1);
  }
  if (cam.model && document.activeElement !== modelBox && modelBox.value !== cam.model) {
    modelBox.value = cam.model;
  }

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
    return render_template_string(
        HTML_PAGE,
        robot_ip=ROBOT_IP,
        confidence_choices=CONFIDENCE_CHOICES,
        confidence=segmentation.confidence,
        model_choices=segmentation.model_choices(),
        model=segmentation.model_name,
        calibration=" en ".join("%d px² = %.2f m" % (area, distance)
                                for area, distance in ARUCO_CALIBRATION),
    )


@app.route("/status")
def status():
    return jsonify({
        "connected": robot.connected,
        "camera": segmentation.snapshot(),
    })


@app.route("/video")
def video():
    """Live camerabeeld van de robot met het masker van het model erop.

    Het is een MJPEG-stream, zo kan de webpagina er gewoon een <img> van maken."""
    if not USE_CAMERA or cv2 is None:
        return jsonify({"ok": False, "error": "camera staat uit"}), 404

    def frames():
        for jpeg in segmentation.stream_frames():
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" +
                   jpeg + b"\r\n")

    return Response(
        frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


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


@app.route("/confidence/", methods=["GET", "POST"])
@app.route("/confidence", methods=["GET", "POST"])
def confidence():
    """Lees of zet de confidence van de segmentatie, bv. /confidence/?value=0.4

    Zonder parameter geeft dit gewoon de huidige waarde terug."""
    raw = request.args.get("value")
    if raw is None:
        return jsonify({"ok": True, "confidence": segmentation.confidence,
                        "choices": CONFIDENCE_CHOICES})

    try:
        value = float(raw)
    except ValueError:
        return jsonify({"ok": False, "error": "confidence moet een getal zijn: " + raw}), 400

    return jsonify({"ok": True, "confidence": segmentation.set_confidence(value)})


@app.route("/model/", methods=["GET", "POST"])
@app.route("/model", methods=["GET", "POST"])
def model():
    """Lees of kies het model, bv. /model/?name=laerbeekbos.pt

    De keuze gaat over de .pt-bestanden die in MODEL_DIR staan. Het laden
    gebeurt in de achtergrondthread en duurt een tiental seconden; ondertussen
    staat de status op "model laden"."""
    raw = request.args.get("name")
    if raw is None:
        return jsonify({"ok": True, "model": segmentation.model_name,
                        "choices": segmentation.model_choices(),
                        "directory": MODEL_DIR})

    try:
        name = segmentation.set_model(raw)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc),
                        "choices": segmentation.model_choices()}), 400

    return jsonify({"ok": True, "model": name})


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
