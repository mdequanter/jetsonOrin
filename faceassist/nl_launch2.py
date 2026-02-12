#!/usr/bin/env python3
"""
AANPASSINGEN (zoals gevraagd)
- Geen input() meer voor ja/nee en naam.
- Bij vragen: meteen luisteren (spraak).
- 1 beep wanneer er mag worden gesproken.
- 2 beeps nadat het antwoord is vastgesteld.
- STT via standaard Whisper (PyTorch) i.p.v. faster-whisper.
"""

import os
import time
import argparse
import urllib.request
import numpy as np
import cv2
import multiprocessing as mp
import signal
import subprocess
import queue as pyqueue
import sys
import json
from datetime import datetime

# NEW: audio + whisper
import sounddevice as sd
import threading
import whisper

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# -----------------------------
# Voice/STT instellingen (pas aan indien nodig)
# -----------------------------
STT_DEVICE_ID = 0        # jouw webcam mic (of 29/33 als ALSA moeilijk doet)
STT_CHANNELS = 2
STT_TARGET_SR = 16000

# Beeps
BEEP_OUT_SR = 44100
BEEP_VOL = 0.35

def beep_once(freq=1200, dur=0.18):
    t = np.linspace(0, dur, int(BEEP_OUT_SR * dur), False)
    tone = (BEEP_VOL * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sd.play(tone, BEEP_OUT_SR)
    sd.wait()

def beep_twice():
    beep_once(900, 0.12)
    time.sleep(0.06)
    beep_once(1200, 0.12)

def pick_input_samplerate(device_id: int, channels: int) -> int:
    candidates = [48000, 16000, 44100, 32000, 24000, 8000]
    for sr in candidates:
        try:
            sd.check_input_settings(device=device_id, channels=channels, samplerate=sr)
            return sr
        except Exception:
            pass
    raise RuntimeError(
        f"No supported samplerate found for device {device_id}. "
        f"Tried {candidates}. Check ALSA/Pulse device."
    )

def resample_linear(x: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr:
        return x.astype(np.float32, copy=False)
    n_out = int(len(x) * out_sr / in_sr)
    xp = np.linspace(0.0, 1.0, len(x), endpoint=False)
    xnew = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(xnew, xp, x).astype(np.float32)

def record_until_silence(
    device_id: int,
    channels: int,
    in_sr: int,
    max_seconds: float = 4.0,
    min_seconds: float = 0.6,
    silence_threshold: float = 0.010,   # RMS drempel (tunen!)
    silence_seconds: float = 0.7,
    block_ms: int = 50
) -> np.ndarray:
    """
    Neemt audio op tot stilte lang genoeg duurt of max_seconds bereikt is.
    Return: mono float32 @ in_sr
    """
    block = int(in_sr * (block_ms / 1000.0))
    need_silence_blocks = max(1, int((silence_seconds * 1000) / block_ms))

    frames = []
    silent_blocks = 0
    started = False
    t0 = time.time()

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))) + 1e-12)

    with sd.InputStream(
        device=device_id,
        channels=channels,
        samplerate=in_sr,
        dtype="float32",
        blocksize=block
    ) as stream:
        while True:
            data, _overflowed = stream.read(block)
            mono = data.mean(axis=1).astype(np.float32)

            frames.append(mono)

            elapsed = time.time() - t0
            if elapsed >= max_seconds:
                break

            level = rms(mono)
            if level >= silence_threshold:
                started = True
                silent_blocks = 0
            else:
                if started:
                    silent_blocks += 1

            if started and elapsed >= min_seconds and silent_blocks >= need_silence_blocks:
                break

    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    return audio

def stt_listen_once(whisper_model, device_id, channels, in_sr, prompt_tts=None, tts_queue=None,
                    max_seconds=4.0, silence_threshold=0.010) -> str:
    """
    - (optioneel) TTS prompt
    - 1 beep: start spreken
    - opnemen tot stilte
    - whisper transcribe NL
    - 2 beeps: antwoord vastgesteld
    - return tekst
    """
    if prompt_tts and tts_queue is not None:
        try:
            tts_queue.put_nowait(prompt_tts)
        except Exception:
            pass

    # start-beep (mag spreken)
    beep_once(1200, 0.18)

    audio = record_until_silence(
        device_id=device_id,
        channels=channels,
        in_sr=in_sr,
        max_seconds=max_seconds,
        silence_threshold=silence_threshold
    )

    # resample -> 16k
    audio_16k = resample_linear(audio, in_sr, STT_TARGET_SR)

    # Whisper expects float32 numpy @ 16k
    result = whisper_model.transcribe(
        audio_16k,
        language="nl",
        fp16=False,
        temperature=0.0,
        condition_on_previous_text=False
    )
    text = (result.get("text") or "").strip()

    # 2 beeps: antwoord vastgesteld
    beep_twice()

    return text

# -----------------------------
# Helpers
# -----------------------------

def download_if_missing(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    print(f"[INFO] Downloaden: {os.path.basename(path)} ...", flush=True)
    urllib.request.urlretrieve(url, path)
    print(f"[OK] Opgeslagen naar {path}", flush=True)

def load_known(known_dir: str):
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

def largest_face(faces: np.ndarray):
    if faces is None or len(faces) == 0:
        return None
    areas = faces[:, 2] * faces[:, 3]
    return faces[int(np.argmax(areas))]

def best_match(recognizer, feat, known: dict):
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

def face_direction_nl(x: int, w_face: int, frame_w: int) -> str:
    cx = x + (w_face // 2)
    if cx < frame_w / 3:
        return "is links van je"
    elif cx > 2 * frame_w / 3:
        return "is rechts van je"
    return "staat voor je"

def open_camera_linux(cam_index: int, width: int, height: int, fps: int):
    dev = f"/dev/video{cam_index}"
    gst_pipeline = (
        f"v4l2src device={dev} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! videoconvert ! appsink drop=true sync=false max-buffers=1"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[INFO] Camera geopend via GStreamer.", flush=True)
        return cap
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        print("[INFO] Camera geopend via V4L2 (OpenCV).", flush=True)
        return cap
    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        print("[INFO] Camera geopend via standaard backend (OpenCV).", flush=True)
        return cap
    return cap

def str2bool(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on", "ja", "j")

def sanitize_name(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = name.replace("..", ".")
    return name

# -----------------------------
# Foto snapshot (JPG)
# -----------------------------
def save_person_snapshot(frame, name: str, out_dir: str = "snapshots") -> str:
    os.makedirs(out_dir, exist_ok=True)
    safe_name = sanitize_name(name) if name else "Onbekend"
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = os.path.join(out_dir, f"{safe_name}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path

# -----------------------------
# Piper TTS
# -----------------------------
def read_piper_sample_rate(model_path: str, default_rate: int = 22050) -> int:
    json_path = model_path + ".json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("sample_rate", "audio.sample_rate", "audio_sample_rate"):
            if key in data and isinstance(data[key], int):
                return int(data[key])
        if isinstance(data.get("audio"), dict) and isinstance(data["audio"].get("sample_rate"), int):
            return int(data["audio"]["sample_rate"])
    except Exception:
        pass
    return default_rate

def piper_say(text: str, model_path: str, sample_rate: int, length_scale: float = 1.0):
    p1 = subprocess.Popen(
        ["piper", "--model", model_path, "--output_raw", "--length_scale", str(length_scale)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    p2 = subprocess.Popen(
        ["aplay", "-r", str(sample_rate), "-f", "S16_LE", "-t", "raw", "-"],
        stdin=p1.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if p1.stdin is not None:
            p1.stdin.write((text + "\n").encode("utf-8"))
            p1.stdin.close()
    except Exception:
        pass
    if p1.stdout is not None:
        p1.stdout.close()
    try:
        p2.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p2.kill()
    try:
        p1.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p1.kill()

def tts_worker_loop(tts_queue: mp.Queue, stop_event: mp.Event, args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        subprocess.run(["piper", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        print("[WAARSCHUWING] 'piper' niet gevonden in PATH.", flush=True)
        return

    model_path = os.path.expanduser(args.piper_model)
    if not os.path.exists(model_path):
        print(f"[WAARSCHUWING] Piper model niet gevonden: {model_path}", flush=True)
        return

    sample_rate = args.piper_rate
    if args.piper_rate_auto:
        sample_rate = read_piper_sample_rate(model_path, default_rate=args.piper_rate)

    while not stop_event.is_set():
        try:
            msg = tts_queue.get(timeout=0.1)
        except pyqueue.Empty:
            continue
        if msg is None:
            break
        text = str(msg).strip()
        if not text:
            continue
        try:
            piper_say(text, model_path=model_path, sample_rate=sample_rate, length_scale=args.piper_length_scale)
        except Exception:
            pass

def tts_enqueue(tts_queue, text: str):
    if tts_queue is None:
        return
    try:
        tts_queue.put_nowait(text)
    except pyqueue.Full:
        pass

# -----------------------------
# Snapshot opslag (features)
# -----------------------------
def save_snapshot(frame, out_dir: str, tag: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = sanitize_name(tag) if tag else "onbekend"
    path = os.path.join(out_dir, f"{ts}_{safe_tag}.jpg")
    cv2.imwrite(path, frame)
    return path

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # Camera + detectie
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--infer_every", type=int, default=2)

    ap.add_argument("--min_face", type=int, default=50)
    ap.add_argument("--score_th", type=float, default=0.9)
    ap.add_argument("--nms_th", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=5000)

    # Herkenning
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--margin", type=float, default=0.06)

    # Entry / leave
    ap.add_argument("--lost_timeout", type=float, default=1.0)
    ap.add_argument("--enter_confirm_frames", type=int, default=3)
    ap.add_argument("--reannounce_after", type=float, default=6.0)

    # Onbekend gedrag
    ap.add_argument("--unknown_seconds", type=float, default=5.0)
    ap.add_argument("--unknown_confirm_frames", type=int, default=5)
    ap.add_argument("--cooldown_after_unknown", type=float, default=10.0)

    ap.add_argument("--unknown_capture_interval", type=float, default=0.5)
    ap.add_argument("--unknown_max_snaps", type=int, default=60)

    # Opslag
    ap.add_argument("--known", type=str, default="known")
    ap.add_argument("--min_save_samples", type=int, default=8)

    ap.add_argument("--unknown_photos", type=str, default="unknown_photos")
    ap.add_argument("--save_unknown_snapshot", action="store_true")

    # Piper TTS
    ap.add_argument("--no_tts", action="store_true")
    ap.add_argument("--speak", type=str, default="True")
    ap.add_argument("--piper_model", type=str, default="~/jetsonOrin/voices/nl_BE-nathalie-medium.onnx")
    ap.add_argument("--piper_rate", type=int, default=22050)
    ap.add_argument("--piper_rate_auto", action="store_true")
    ap.add_argument("--piper_length_scale", type=float, default=1.0)
    ap.add_argument("--tts_queue_size", type=int, default=20)

    # NEW: Whisper model keuze
    ap.add_argument("--whisper_model", type=str, default="base", help="tiny/base/small/...")

    args = ap.parse_args()

    yunet_path = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join("models", "face_recognition_sface_2021dec.onnx")
    download_if_missing(YUNET_URL, yunet_path)
    download_if_missing(SFACE_URL, sface_path)

    os.makedirs(args.known, exist_ok=True)

    # TTS
    stop_event = mp.Event()
    tts_queue = None
    tts_proc = None
    speak_enabled = (not args.no_tts) and str2bool(args.speak)
    if speak_enabled:
        tts_queue = mp.Queue(maxsize=args.tts_queue_size)
        tts_proc = mp.Process(target=tts_worker_loop, args=(tts_queue, stop_event, args), daemon=True)
        tts_proc.start()
        tts_enqueue(tts_queue, "Gezichtsherkenning is gestart.")

    # NEW: init Whisper + audio input SR
    stt_in_sr = pick_input_samplerate(STT_DEVICE_ID, STT_CHANNELS)
    print(f"[INFO] STT input: device={STT_DEVICE_ID}, channels={STT_CHANNELS}, sr={stt_in_sr}", flush=True)
    whisper_model = whisper.load_model(args.whisper_model)

    # Camera
    cap = open_camera_linux(args.cam, args.width, args.height, args.fps)
    if not cap.isOpened():
        print("[FOUT] Kan camera niet openen.", flush=True)
        stop_event.set()
        if tts_queue is not None:
            try: tts_queue.put_nowait(None)
            except Exception: pass
        return

    ok, frame = cap.read()
    if not ok or frame is None:
        print("[FOUT] Kan eerste frame niet lezen.", flush=True)
        cap.release()
        return

    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h), args.score_th, args.nms_th, args.topk)
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    known = load_known(args.known)
    if known:
        print("[INFO] Bekend:", ", ".join(sorted(known.keys())), flush=True)
        if speak_enabled:
            tts_enqueue(tts_queue, f"{len(known)} personen geladen.")
    else:
        print(f"[WAARSCHUWING] Geen bekende identiteiten in '{args.known}'.", flush=True)
        if speak_enabled:
            tts_enqueue(tts_queue, "Ik ken nog niemand. Ik kan nieuwe personen opslaan wanneer ik ze zie.")

    # Entry/leave state
    present = False
    present_name = None
    last_seen = 0.0

    consec_needed = args.enter_confirm_frames
    consec_count = 0
    candidate_name = None
    last_announced_at = {}

    # Unknown state
    unknown_consec = 0
    unknown_started_at = None
    last_unknown_handled_at = 0.0

    unknown_feats = []
    unknown_last_cap = 0.0
    unknown_last_frame = None

    last_person_photo_at = {}
    person_photo_cooldown = 3.0

    frame_id = 0
    print("[INFO] Headless actief. Ctrl+C om te stoppen.", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue

            frame_id += 1
            if frame_id % args.infer_every != 0:
                continue

            now = time.time()

            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            face = largest_face(faces)

            if face is None:
                if present and (now - last_seen) >= args.lost_timeout:
                    print(f"[INFO] {present_name} is uit beeld.", flush=True)
                    present = False
                    present_name = None
                    consec_count = 0
                    candidate_name = None

                unknown_consec = 0
                unknown_started_at = None
                unknown_feats = []
                unknown_last_cap = 0.0
                unknown_last_frame = None
                continue

            x, y, fw, fh = face[:4].astype(int)
            if fw < args.min_face:
                if present and (now - last_seen) >= args.lost_timeout:
                    print(f"[INFO] {present_name} is uit beeld.", flush=True)
                    present = False
                    present_name = None

                unknown_consec = 0
                unknown_started_at = None
                unknown_feats = []
                unknown_last_cap = 0.0
                unknown_last_frame = None
                consec_count = 0
                candidate_name = None
                continue

            richting = face_direction_nl(x, fw, w)

            aligned = recognizer.alignCrop(frame, face)
            feat = recognizer.feature(aligned).astype(np.float32)

            best_name, best_score, second_score = best_match(recognizer, feat, known) if known else (None, -1.0, -1.0)
            confident = (best_name is not None) and (best_score >= args.threshold) and ((best_score - second_score) >= args.margin)

            # -------------------------
            # ONBEKEND
            # -------------------------
            if not confident:
                if present and (now - last_seen) >= args.lost_timeout:
                    print(f"[INFO] {present_name} is uit beeld.", flush=True)
                    present = False
                    present_name = None

                if (now - last_unknown_handled_at) < args.cooldown_after_unknown:
                    unknown_consec = 0
                    unknown_started_at = None
                    unknown_feats = []
                    unknown_last_cap = 0.0
                    unknown_last_frame = None
                    continue

                unknown_consec += 1
                if unknown_consec < args.unknown_confirm_frames:
                    continue

                if unknown_started_at is None:
                    unknown_started_at = now
                    unknown_feats = []
                    unknown_last_cap = 0.0
                    unknown_last_frame = None
                    print("[INFO] Onbekend gezicht gedetecteerd. Snapshots verzamelen...", flush=True)
                    if speak_enabled:
                        tts_enqueue(tts_queue, "Ik zie iemand die ik niet herken.")

                    name_for_photo = "Onbekend"
                    last_t = last_person_photo_at.get(name_for_photo, 0.0)
                    if (now - last_t) >= person_photo_cooldown:
                        p = save_person_snapshot(frame, name_for_photo, out_dir="snapshots")
                        last_person_photo_at[name_for_photo] = now
                        print("[OK] Snapshot opgeslagen:", p, flush=True)

                unknown_last_frame = frame

                if (now - unknown_last_cap) >= args.unknown_capture_interval:
                    try:
                        aligned_u = recognizer.alignCrop(frame, face)
                        feat_u = recognizer.feature(aligned_u).astype(np.float32)
                        unknown_feats.append(feat_u)
                        unknown_last_cap = now

                        if len(unknown_feats) > args.unknown_max_snaps:
                            unknown_feats = unknown_feats[-args.unknown_max_snaps:]

                        if len(unknown_feats) % 10 == 0:
                            print(f"[INFO] Onbekende snapshots: {len(unknown_feats)}", flush=True)
                    except Exception:
                        pass

                if (now - unknown_started_at) >= args.unknown_seconds:
                    print("[INFO] Nieuwe persoon gedetecteerd. Vraag om op te slaan (spraak)...", flush=True)

                    # VRAAG 1: ja/nee via spraak
                    ans_spoken = stt_listen_once(
                        whisper_model,
                        STT_DEVICE_ID, STT_CHANNELS, stt_in_sr,
                        prompt_tts="Nieuwe persoon gedetecteerd. Wil je deze persoon opslaan? Zeg ja of nee.",
                        tts_queue=tts_queue,
                        max_seconds=3.0,
                        silence_threshold=0.006
                    ).lower()
                    print("[STT] ja/nee:", ans_spoken, flush=True)

                    save_it = any(ans_spoken.startswith(x) for x in ("ja", "jawel", "yes", "j", "y"))
                    if save_it:
                        # VRAAG 2: naam via spraak
                        name_spoken = stt_listen_once(
                            whisper_model,
                            STT_DEVICE_ID, STT_CHANNELS, stt_in_sr,
                            prompt_tts="Wat is de naam van de persoon? Zeg nu de naam.",
                            tts_queue=tts_queue,
                            max_seconds=4.0,
                            silence_threshold=0.010
                        )
                        name = sanitize_name(name_spoken)
                        print("[STT] naam:", name, flush=True)

                        if not name:
                            print("[INFO] Geen naam herkend. Niet opslaan.", flush=True)
                            if speak_enabled:
                                tts_enqueue(tts_queue, "Ik heb geen naam verstaan. Ik sla niets op.")
                        else:
                            if len(unknown_feats) >= args.min_save_samples:
                                out_path = os.path.join(args.known, f"{name}.npz")
                                np.savez_compressed(out_path, features=np.stack(unknown_feats, axis=0))
                                print("[OK] Opgeslagen:", out_path, flush=True)

                                if speak_enabled:
                                    tts_enqueue(tts_queue, f"Opgeslagen. {name} is toegevoegd.")

                                known = load_known(args.known)

                                if args.save_unknown_snapshot and unknown_last_frame is not None:
                                    tagged_path = save_snapshot(unknown_last_frame, args.unknown_photos, name)
                                    print("[OK] Foto opgeslagen:", tagged_path, flush=True)
                            else:
                                print(f"[WAARSCHUWING] Te weinig snapshots ({len(unknown_feats)}). Niet opslaan.", flush=True)
                                if speak_enabled:
                                    tts_enqueue(tts_queue, "Ik kon niet genoeg snapshots nemen. Ik sla niets op.")
                    else:
                        print("[INFO] Gebruiker kiest om niet op te slaan.", flush=True)
                        if speak_enabled:
                            tts_enqueue(tts_queue, "Oké. Ik sla niets op.")

                    last_unknown_handled_at = time.time()
                    unknown_consec = 0
                    unknown_started_at = None
                    unknown_feats = []
                    unknown_last_cap = 0.0
                    unknown_last_frame = None

                continue

            # -------------------------
            # BEKEND
            # -------------------------
            last_seen = now

            unknown_consec = 0
            unknown_started_at = None
            unknown_feats = []
            unknown_last_cap = 0.0
            unknown_last_frame = None

            if present and best_name == present_name:
                continue

            if candidate_name == best_name:
                consec_count += 1
            else:
                candidate_name = best_name
                consec_count = 1

            if consec_count < consec_needed:
                continue

            last_spoke = last_announced_at.get(candidate_name, 0.0)
            if (now - last_spoke) < args.reannounce_after:
                present = True
                present_name = candidate_name
                consec_count = 0
                candidate_name = None
                continue

            present = True
            present_name = candidate_name
            last_announced_at[present_name] = now

            last_t = last_person_photo_at.get(present_name, 0.0)
            if (now - last_t) >= person_photo_cooldown:
                p = save_person_snapshot(frame, present_name, out_dir="snapshots")
                last_person_photo_at[present_name] = now
                print("[OK] Snapshot opgeslagen:", p, flush=True)

            print(f"[INFO] BINNEN: {present_name} {richting} (score={best_score:.2f}, tweede={second_score:.2f})", flush=True)
            if speak_enabled:
                tts_enqueue(tts_queue, f"{present_name} {richting}")

            consec_count = 0
            candidate_name = None

    except KeyboardInterrupt:
        print("\n[INFO] Stoppen...", flush=True)

    finally:
        cap.release()
        stop_event.set()
        if tts_queue is not None:
            try:
                tts_queue.put_nowait(None)
            except Exception:
                pass
        if tts_proc is not None:
            tts_proc.join(timeout=1.0)
            if tts_proc.is_alive():
                tts_proc.terminate()
                tts_proc.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
