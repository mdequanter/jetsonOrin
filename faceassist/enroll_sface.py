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


YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


# -----------------------------
# Download models if needed
# -----------------------------

def download_if_missing(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    print(f"[INFO] Downloading {os.path.basename(path)} ...")
    urllib.request.urlretrieve(url, path)
    print(f"[OK] Saved to {path}")


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

        #print("[TTS]", msg)
        espeak_say(msg, args.tts_rate, args.espeak_voice,
                   args.espeak_pitch, args.espeak_amp)


def tts_enqueue(tts_queue, text):
    if tts_queue is None:
        return
    try:
        tts_queue.put_nowait(text)
    except pyqueue.Full:
        pass


# -----------------------------
# Helpers
# -----------------------------

def safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError):
        return ""


def sanitize_name(name: str) -> str:
    return name.strip().replace("/", "_").replace("\\", "_")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="known")
    ap.add_argument("--min_face", type=int, default=80)

    # TTS options
    ap.add_argument("--tts_rate", type=int, default=175)
    ap.add_argument("--espeak_voice", type=str, default="nl")
    ap.add_argument("--espeak_pitch", type=int, default=50)
    ap.add_argument("--espeak_amp", type=int, default=100)
    ap.add_argument("--tts_queue_size", type=int, default=10)

    args = ap.parse_args()

    # Download models
    yunet_path = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join("models", "face_recognition_sface_2021dec.onnx")

    download_if_missing(YUNET_URL, yunet_path)
    download_if_missing(SFACE_URL, sface_path)

    os.makedirs(args.outdir, exist_ok=True)

    # Start TTS
    stop_event = mp.Event()
    tts_queue = mp.Queue(maxsize=args.tts_queue_size)

    tts_proc = mp.Process(
        target=tts_worker_loop,
        args=(tts_queue, stop_event, args),
        daemon=True
    )
    tts_proc.start()

    # Camera
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Camera werkt niet.")
        return

    h, w = frame.shape[:2]

    detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h))
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    tts_enqueue(tts_queue, "Systeem gestart. Ik wacht op een gezicht.")

    try:
        while True:

            # ---------------------------
            # 1. Wacht op gezicht
            # ---------------------------
            face = None
            while face is None:
                ok, frame = cap.read()
                if not ok:
                    break

                detector.setInputSize((w, h))
                _, faces = detector.detect(frame)
                face = largest_face(faces)

                time.sleep(0.1)

            if face is None:
                continue

            x, y, fw, fh = face[:4].astype(int)

            if fw < args.min_face:
                tts_enqueue(tts_queue, "Kom iets dichter bij de camera.")
                time.sleep(2)
                continue

            # ---------------------------
            # 2. Naam vragen
            # ---------------------------
            tts_enqueue(tts_queue, "Gezicht gedetecteerd.")
            tts_enqueue(tts_queue, "Wat is de naam van deze persoon? Typ de naam.")

            name = sanitize_name(safe_input("\nNaam: "))

            if not name:
                tts_enqueue(tts_queue, "Geen naam ingegeven. Ik wacht opnieuw.")
                continue

            tts_enqueue(tts_queue, f"Oké {name}. Ik neem nu foto's op.")

            # ---------------------------
            # 3. Samples opnemen
            # ---------------------------
            features = []
            last_capture = 0

            while len(features) < args.samples:
                ok, frame = cap.read()
                if not ok:
                    break

                detector.setInputSize((w, h))
                _, faces = detector.detect(frame)
                face = largest_face(faces)

                if face is None:
                    tts_enqueue(tts_queue, "Gezicht kwijt. Kijk naar de camera.")
                    time.sleep(1)
                    continue

                now = time.time()
                if now - last_capture > 0.5:
                    aligned = recognizer.alignCrop(frame, face)
                    feat = recognizer.feature(aligned).astype(np.float32)
                    features.append(feat)
                    last_capture = now

                    if len(features) % 5 == 0:
                        tts_enqueue(tts_queue, f"{len(features)} voorbeelden opgenomen.")

            tts_enqueue(tts_queue, "Klaar met opnemen.")

            # ---------------------------
            # 4. Opslaan vragen
            # ---------------------------
            tts_enqueue(tts_queue, "Mag ik deze persoon opslaan? Typ j of n.")

            ans = safe_input("\nOpslaan? (j/n): ").strip().lower()

            if ans.startswith("j"):
                out_path = os.path.join(args.outdir, f"{name}.npz")
                np.savez_compressed(out_path, features=np.stack(features))
                tts_enqueue(tts_queue, f"{name} is opgeslagen.")
                print("[OK] Saved:", out_path)

            else:
                tts_enqueue(tts_queue, "Oké. Ik sla niets op.")

            tts_enqueue(tts_queue, "Ik wacht opnieuw op een gezicht.")

    except KeyboardInterrupt:
        print("\n[INFO] Stoppen...")

    finally:
        cap.release()
        stop_event.set()
        tts_queue.put(None)
        tts_proc.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
