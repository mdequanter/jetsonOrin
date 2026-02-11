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
import select
import termios
import tty


YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


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


# -----------------------------
# Non-blocking keyboard input
# -----------------------------

class RawStdin:
    """
    Put stdin in raw mode so we can read single keys without blocking.
    """
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = None

    def __enter__(self):
        if not sys.stdin.isatty():
            return self
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def read_key_nonblocking(timeout=0.0):
    """
    Return a single character if available, else None.
    """
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if r:
        ch = sys.stdin.read(1)
        return ch
    return None


def collect_name_via_keys(tts_queue, max_len=32, timeout_s=30):
    """
    Collect a name using typed keys. ENTER confirms, ESC cancels.
    Backspace supported.
    """
    tts_enqueue(tts_queue, "Typ de naam op het toetsenbord. Druk op Enter om te bevestigen. Escape om te annuleren.")
    print("\n[INPUT] Typ naam (Enter=OK, Esc=Annuleer): ", end="", flush=True)

    buf = ""
    start = time.time()

    while True:
        if timeout_s and (time.time() - start) > timeout_s:
            print("\n[INPUT] Timeout.")
            tts_enqueue(tts_queue, "Geen invoer. Timeout.")
            return ""

        ch = read_key_nonblocking(timeout=0.1)
        if ch is None:
            continue

        # ESC cancels
        if ch == "\x1b":
            print("\n[INPUT] Geannuleerd.")
            tts_enqueue(tts_queue, "Geannuleerd.")
            return ""

        # ENTER confirms
        if ch in ("\r", "\n"):
            name = sanitize_name(buf)
            print()  # newline
            return name

        # Backspace (Linux terminals: \x7f)
        if ch in ("\x7f", "\b"):
            if len(buf) > 0:
                buf = buf[:-1]
                # remove last char on screen
                print("\b \b", end="", flush=True)
            continue

        # Accept simple printable chars
        if ch.isprintable() and ch not in ("\t",):
            if len(buf) < max_len:
                buf += ch
                print(ch, end="", flush=True)


def confirm_yes_no_via_keys(tts_queue, question_tts, timeout_s=20):
    """
    Ask J/N using single key press.
    Returns True for yes, False for no/timeout/cancel.
    """
    tts_enqueue(tts_queue, question_tts + " Druk op J voor ja, of N voor nee.")
    print("\n[INPUT] (J/N): ", end="", flush=True)

    start = time.time()
    while True:
        if timeout_s and (time.time() - start) > timeout_s:
            print("\n[INPUT] Timeout -> nee.")
            tts_enqueue(tts_queue, "Geen antwoord. Ik neem nee.")
            return False

        ch = read_key_nonblocking(timeout=0.1)
        if ch is None:
            continue
        c = ch.lower()

        if c == "j":
            print("j")
            return True
        if c == "n":
            print("n")
            return False
        if ch == "\x1b":  # ESC
            print("\n[INPUT] Geannuleerd -> nee.")
            tts_enqueue(tts_queue, "Geannuleerd.")
            return False


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
        print("[ERROR] Camera werkt niet.")
        return

    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h))
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    tts_enqueue(tts_queue, "Systeem gestart. Ik wacht op een gezicht. Druk Control C om te stoppen.")

    try:
        with RawStdin():  # raw keyboard input mode
            while True:
                # 1) wait for face
                face = None
                while face is None:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    detector.setInputSize((w, h))
                    _, faces = detector.detect(frame)
                    face = largest_face(faces)
                    time.sleep(0.05)

                if face is None:
                    continue

                x, y, fw, fh = face[:4].astype(int)
                if fw < args.min_face:
                    tts_enqueue(tts_queue, "Kom iets dichter bij de camera.")
                    time.sleep(1.0)
                    continue

                # 2) ask name (keys, no blocking input())
                tts_enqueue(tts_queue, "Gezicht gedetecteerd.")
                name = collect_name_via_keys(tts_queue, timeout_s=30)

                if not name:
                    tts_enqueue(tts_queue, "Geen naam. Ik wacht opnieuw op een gezicht.")
                    time.sleep(0.5)
                    continue

                tts_enqueue(tts_queue, f"Oké {name}. Ik neem nu voorbeelden op.")

                # 3) capture features
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
                        time.sleep(0.5)
                        continue

                    x, y, fw, fh = face[:4].astype(int)
                    if fw < args.min_face:
                        tts_enqueue(tts_queue, "Kom iets dichter bij de camera.")
                        time.sleep(0.8)
                        continue

                    now = time.time()
                    if now - last_capture >= args.capture_interval:
                        aligned = recognizer.alignCrop(frame, face)
                        feat = recognizer.feature(aligned).astype(np.float32)
                        features.append(feat)
                        last_capture = now

                        if len(features) % 5 == 0:
                            tts_enqueue(tts_queue, f"{len(features)} voorbeelden opgenomen.")

                tts_enqueue(tts_queue, "Klaar met opnemen.")

                # 4) confirm save via single key
                do_save = confirm_yes_no_via_keys(tts_queue, f"Mag ik {name} opslaan?", timeout_s=20)
                if do_save and len(features) >= 8:
                    out_path = os.path.join(args.outdir, f"{name}.npz")
                    np.savez_compressed(out_path, features=np.stack(features, axis=0))
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
