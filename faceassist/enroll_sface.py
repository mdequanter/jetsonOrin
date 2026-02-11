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


YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

naam = input("Wat is jouw naam? ")


def download_if_missing(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    print(f"[INFO] Downloading {os.path.basename(path)} ...", flush=True)
    urllib.request.urlretrieve(url, path)
    print(f"[OK] Saved to {path}", flush=True)


def largest_face(faces: np.ndarray):
    if faces is None or len(faces) == 0:
        return None
    areas = faces[:, 2] * faces[:, 3]
    return faces[int(np.argmax(areas))]


# -----------------------------
# Async TTS Process
# -----------------------------

def espeak_say(text: str, rate: int, voice: str, pitch: int, amp: int):
    cmd = ["espeak", "-v", voice, "-s", str(rate), "-p", str(pitch), "-a", str(amp)]
    subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False
    )


def tts_worker_loop(tts_queue: mp.Queue, stop_event: mp.Event, args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while not stop_event.is_set():
        try:
            msg = tts_queue.get(timeout=0.1)
        except pyqueue.Empty:
            continue
        if msg is None:
            break
        espeak_say(msg, args.tts_rate, args.espeak_voice, args.espeak_pitch, args.espeak_amp)


def tts_enqueue(tts_queue, text):
    if tts_queue is None:
        return
    try:
        tts_queue.put_nowait(text)
    except pyqueue.Full:
        pass


def sanitize_name(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = name.replace("..", ".")
    return name


def ask_input(prompt: str) -> str:
    # ensure prompt shows immediately
    sys.stdout.write(prompt)
    sys.stdout.flush()
    return input()


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="known")
    ap.add_argument("--min_face", type=int, default=80)
    ap.add_argument("--capture_interval", type=float, default=0.5)

    # TTS options
    ap.add_argument("--tts_rate", type=int, default=175)
    ap.add_argument("--espeak_voice", type=str, default="nl")
    ap.add_argument("--espeak_pitch", type=int, default=50)
    ap.add_argument("--espeak_amp", type=int, default=100)
    ap.add_argument("--tts_queue_size", type=int, default=10)

    args = ap.parse_args()

    yunet_path = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join("models", "face_recognition_sface_2021dec.onnx")
    download_if_missing(YUNET_URL, yunet_path)
    download_if_missing(SFACE_URL, sface_path)
    os.makedirs(args.outdir, exist_ok=True)

    # Start TTS
    stop_event = mp.Event()
    tts_queue = mp.Queue(maxsize=args.tts_queue_size)
    tts_proc = mp.Process(target=tts_worker_loop, args=(tts_queue, stop_event, args), daemon=True)
    tts_proc.start()

    # Camera
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    ok, frame = cap.read()
    if not ok or frame is None:
        print("[ERROR] Camera werkt niet.", flush=True)
        return

    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h))
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    tts_enqueue(tts_queue, "Systeem gestart. Ik wacht op een gezicht. Druk Control C om te stoppen.")
    #print("[INFO] Running. Ctrl+C om te stoppen.", flush=True)

    try:
        while True:
            # ---------------------------
            # 1) Wacht op een groot genoeg gezicht
            # ---------------------------
            face = None
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    face = None
                    break

                detector.setInputSize((w, h))
                _, faces = detector.detect(frame)
                face = largest_face(faces)

                if face is not None:
                    x, y, fw, fh = face[:4].astype(int)
                    if fw >= args.min_face:
                        break  # face good enough

                time.sleep(0.05)

            if face is None:
                continue

            # ---------------------------
            # 2) Pauzeer capture en vraag naam (input)
            # ---------------------------
            tts_enqueue(tts_queue, "Gezicht gedetecteerd.")
            tts_enqueue(tts_queue, "Typ nu de naam in de terminal en druk op Enter.")

            name = input("Naam: ")

            if not name:
                tts_enqueue(tts_queue, "Geen naam ingegeven. Ik wacht opnieuw op een gezicht.")
                #print("[INFO] Geen naam. Terug naar wachten.\n", flush=True)
                time.sleep(0.3)
                continue

            tts_enqueue(tts_queue, f"Oké {name}. Ik neem nu voorbeelden op.")
            print(f"[INFO] Capturing samples for: {name}", flush=True)

            # ---------------------------
            # 3) Samples opnemen
            # ---------------------------
            features = []
            last_capture = 0.0

            while len(features) < args.samples:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                detector.setInputSize((w, h))
                _, faces = detector.detect(frame)
                face = largest_face(faces)

                if face is None:
                    tts_enqueue(tts_queue, "Gezicht kwijt. Kijk naar de camera.")
                    time.sleep(0.4)
                    continue

                x, y, fw, fh = face[:4].astype(int)
                if fw < args.min_face:
                    tts_enqueue(tts_queue, "Kom iets dichter bij de camera.")
                    time.sleep(0.6)
                    continue

                now = time.time()
                if now - last_capture >= args.capture_interval:
                    aligned = recognizer.alignCrop(frame, face)
                    feat = recognizer.feature(aligned).astype(np.float32)
                    features.append(feat)
                    last_capture = now

                    if len(features) % 5 == 0:
                        tts_enqueue(tts_queue, f"{len(features)} voorbeelden opgenomen.")
                        print(f"[INFO] {len(features)}/{args.samples}", flush=True)

            tts_enqueue(tts_queue, "Klaar met opnemen.")
            print("[INFO] Capture done.", flush=True)

            # ---------------------------
            # 4) Vraag opslaan (input)
            # ---------------------------
            tts_enqueue(tts_queue, "Mag ik deze persoon opslaan? Typ j of n en druk op Enter.")

            ans = input("Opslaan? (j/n): ").strip().lower()

            if ans.startswith("j") and len(features) >= 8:
                out_path = os.path.join(args.outdir, f"{name}.npz")
                np.savez_compressed(out_path, features=np.stack(features, axis=0))
                tts_enqueue(tts_queue, f"{name} is opgeslagen.")
                print("[OK] Saved:", out_path, flush=True)
            else:
                tts_enqueue(tts_queue, "Oké. Ik sla niets op.")
                print("[INFO] Not saved.", flush=True)

            tts_enqueue(tts_queue, "Ik wacht opnieuw op een gezicht.")

    except KeyboardInterrupt:
        print("\n[INFO] Stoppen...", flush=True)

    finally:
        cap.release()
        stop_event.set()
        try:
            tts_queue.put_nowait(None)
        except Exception:
            pass
        tts_proc.join(timeout=1.0)
        if tts_proc.is_alive():
            tts_proc.terminate()
            tts_proc.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
