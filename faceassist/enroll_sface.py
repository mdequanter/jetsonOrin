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


def sanitize_name(name: str) -> str:
    name = name.strip().replace("/", "_").replace("\\", "_")
    name = name.replace("..", ".")
    return name


# -----------------------------
# TTS worker (espeak)
# -----------------------------

def espeak_say(text: str, rate: int, voice: str, pitch: int, amp: int):
    cmd = ["espeak", "-v", voice, "-s", str(rate), "-p", str(pitch), "-a", str(amp)]
    subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
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
        try:
            espeak_say(msg, args.tts_rate, args.espeak_voice, args.espeak_pitch, args.espeak_amp)
        except Exception:
            pass


def tts_enqueue(tts_queue: mp.Queue, text: str):
    if tts_queue is None:
        return
    try:
        tts_queue.put_nowait(text)
    except pyqueue.Full:
        pass


# -----------------------------
# Vision worker (camera + detect + capture)
# -----------------------------

def vision_worker_loop(cmd_q: mp.Queue, resp_q: mp.Queue, stop_event: mp.Event, args, yunet_path: str, sface_path: str):
    """
    Commands from cmd_q:
      ("WAIT_FOR_FACE",) -> start looking; when found send ("FACE_DETECTED", fw, fh)
      ("CAPTURE", samples, min_face, capture_interval) -> capture features; send ("CAPTURE_DONE", features_np)
      ("RESET",) -> clear internal state
      ("STOP",) -> exit
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    ok, frame = cap.read()
    if not ok or frame is None:
        resp_q.put(("ERROR", "Cannot read from camera"))
        cap.release()
        return

    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h), args.score_th, args.nms_th, args.topk)
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    mode = None  # None / "WAIT" / "CAPTURE"
    need_samples = 0
    min_face = args.min_face
    capture_interval = args.capture_interval
    features = []
    last_capture = 0.0

    while not stop_event.is_set():
        # handle commands (non-blocking)
        try:
            cmd = cmd_q.get_nowait()
        except pyqueue.Empty:
            cmd = None

        if cmd is not None:
            op = cmd[0]

            if op == "STOP":
                break

            if op == "RESET":
                mode = None
                features = []
                last_capture = 0.0
                continue

            if op == "WAIT_FOR_FACE":
                mode = "WAIT"
                features = []
                last_capture = 0.0
                continue

            if op == "CAPTURE":
                _, need_samples, min_face, capture_interval = cmd
                mode = "CAPTURE"
                features = []
                last_capture = 0.0
                continue

        # do work depending on mode
        if mode is None:
            time.sleep(0.05)
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            resp_q.put(("ERROR", "Frame grab failed"))
            time.sleep(0.2)
            continue

        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        face = largest_face(faces)

        if mode == "WAIT":
            if face is None:
                time.sleep(0.05)
                continue

            x, y, fw, fh = face[:4].astype(int)
            if fw < min_face:
                # still too small; keep waiting
                time.sleep(0.05)
                continue

            resp_q.put(("FACE_DETECTED", fw, fh))
            mode = None  # stop until main tells what next
            continue

        if mode == "CAPTURE":
            if face is None:
                # tell main occasionally? keep it simple: just keep trying
                time.sleep(0.05)
                continue

            x, y, fw, fh = face[:4].astype(int)
            if fw < min_face:
                time.sleep(0.05)
                continue

            now = time.time()
            if now - last_capture >= capture_interval:
                aligned = recognizer.alignCrop(frame, face)
                feat = recognizer.feature(aligned).astype(np.float32)
                features.append(feat)
                last_capture = now

                # progress ping every 5
                if len(features) % 5 == 0:
                    resp_q.put(("PROGRESS", len(features), need_samples))

            if len(features) >= need_samples:
                feats_np = np.stack(features, axis=0)
                resp_q.put(("CAPTURE_DONE", feats_np))
                mode = None
                continue

    cap.release()


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

    # detector params
    ap.add_argument("--score_th", type=float, default=0.9)
    ap.add_argument("--nms_th", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=5000)

    # TTS options
    ap.add_argument("--tts_rate", type=int, default=175)
    ap.add_argument("--espeak_voice", type=str, default="nl")
    ap.add_argument("--espeak_pitch", type=int, default=50)
    ap.add_argument("--espeak_amp", type=int, default=100)
    ap.add_argument("--tts_queue_size", type=int, default=20)

    args = ap.parse_args()

    # models
    yunet_path = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join("models", "face_recognition_sface_2021dec.onnx")
    download_if_missing(YUNET_URL, yunet_path)
    download_if_missing(SFACE_URL, sface_path)
    os.makedirs(args.outdir, exist_ok=True)

    # TTS process
    stop_event = mp.Event()
    tts_queue = mp.Queue(maxsize=args.tts_queue_size)
    tts_proc = mp.Process(target=tts_worker_loop, args=(tts_queue, stop_event, args), daemon=True)
    tts_proc.start()

    # Vision process
    cmd_q = mp.Queue()
    resp_q = mp.Queue()
    vision_proc = mp.Process(
        target=vision_worker_loop,
        args=(cmd_q, resp_q, stop_event, args, yunet_path, sface_path),
        daemon=True
    )
    vision_proc.start()

    tts_enqueue(tts_queue, "Systeem gestart. Ik wacht op een gezicht. Druk Control C om te stoppen.")
    print("[INFO] Running. Ctrl+C om te stoppen.", flush=True)

    try:
        while True:
            # 1) wait for face
            cmd_q.put(("WAIT_FOR_FACE",))
            tts_enqueue(tts_queue, "Ik wacht op een gezicht.")

            # wait for FACE_DETECTED
            while True:
                msg = resp_q.get()
                if msg[0] == "ERROR":
                    print("[ERROR]", msg[1], flush=True)
                    tts_enqueue(tts_queue, "Er is een fout met de camera.")
                    time.sleep(1.0)
                    break

                if msg[0] == "FACE_DETECTED":
                    tts_enqueue(tts_queue, "Gezicht gedetecteerd.")
                    break

            # 2) ask name (input works because camera is in another process)
            tts_enqueue(tts_queue, "Typ nu de naam in de terminal en druk op Enter.")
            name = sanitize_name(input("Naam: ").strip())

            if not name:
                tts_enqueue(tts_queue, "Geen naam ingegeven. Ik begin opnieuw.")
                cmd_q.put(("RESET",))
                continue

            # 3) capture samples
            tts_enqueue(tts_queue, f"Oké {name}. Ik neem nu voorbeelden op.")
            cmd_q.put(("CAPTURE", args.samples, args.min_face, args.capture_interval))

            feats_np = None
            while True:
                msg = resp_q.get()
                if msg[0] == "PROGRESS":
                    _, n, total = msg
                    tts_enqueue(tts_queue, f"{n} van {total}.")
                elif msg[0] == "CAPTURE_DONE":
                    feats_np = msg[1]
                    tts_enqueue(tts_queue, "Klaar met opnemen.")
                    break
                elif msg[0] == "ERROR":
                    print("[ERROR]", msg[1], flush=True)
                    tts_enqueue(tts_queue, "Er is een fout opgetreden tijdens opnemen.")
                    feats_np = None
                    break

            if feats_np is None or feats_np.shape[0] < 8:
                tts_enqueue(tts_queue, "Te weinig voorbeelden. Ik begin opnieuw.")
                cmd_q.put(("RESET",))
                continue

            # 4) confirm save
            tts_enqueue(tts_queue, "Mag ik deze persoon opslaan? Typ j of n en druk op Enter.")
            ans = input("Opslaan? (j/n): ").strip().lower()

            if ans.startswith("j"):
                out_path = os.path.join(args.outdir, f"{name}.npz")
                np.savez_compressed(out_path, features=feats_np)
                print("[OK] Saved:", out_path, flush=True)
                tts_enqueue(tts_queue, f"{name} is opgeslagen.")
            else:
                tts_enqueue(tts_queue, "Oké. Ik sla niets op.")

            cmd_q.put(("RESET",))

    except KeyboardInterrupt:
        print("\n[INFO] Stoppen...", flush=True)

    finally:
        stop_event.set()
        try:
            cmd_q.put(("STOP",))
        except Exception:
            pass

        try:
            tts_queue.put_nowait(None)
        except Exception:
            pass

        vision_proc.join(timeout=1.0)
        if vision_proc.is_alive():
            vision_proc.terminate()
            vision_proc.join()

        tts_proc.join(timeout=1.0)
        if tts_proc.is_alive():
            tts_proc.terminate()
            tts_proc.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
