import os
import time
import argparse
import urllib.request
import numpy as np
import cv2
import multiprocessing as mp
import signal
import subprocess
import queue as pyqueue  # Empty / Full


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
# Async TTS (espeak) process
# -----------------------------

def espeak_say(text: str, rate: int, voice: str, pitch: int, amp: int):
    cmd = ["espeak", "-v", str(voice), "-s", str(rate), "-p", str(pitch), "-a", str(amp)]
    subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def tts_worker_loop(tts_queue: mp.Queue, stop_event: mp.Event, args):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        subprocess.run(["espeak", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        print("[WARN] 'espeak' not found. Install with: sudo apt-get install espeak")
        return

    while not stop_event.is_set():
        try:
            item = tts_queue.get(timeout=0.1)
        except pyqueue.Empty:
            continue

        if item is None:
            break

        text = str(item).strip()
        if not text:
            continue

        try:
            print("[TTS]", text)
            espeak_say(text, args.tts_rate, args.espeak_voice, args.espeak_pitch, args.espeak_amp)
        except Exception as e:
            print("[WARN] espeak failed:", e)


def tts_enqueue(tts_queue: mp.Queue, text: str):
    if tts_queue is None:
        return
    try:
        tts_queue.put_nowait(text)
    except pyqueue.Full:
        pass


def safe_input(prompt: str) -> str:
    """
    Read from stdin safely; returns empty string on Ctrl+C / EOF.
    """
    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError):
        return ""


def sanitize_name(name: str) -> str:
    # keep it simple & filesystem safe
    name = name.strip()
    name = name.replace("/", "_").replace("\\", "_")
    name = name.replace("..", ".")
    return name


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="known")
    ap.add_argument("--min_face", type=int, default=80, help="Min face width in pixels")
    ap.add_argument("--score_th", type=float, default=0.9)
    ap.add_argument("--nms_th", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=5000)

    # TTS options
    ap.add_argument("--no_tts", action="store_true", help="Disable TTS")
    ap.add_argument("--tts_rate", type=int, default=175)
    ap.add_argument("--espeak_voice", type=str, default="nl", help="espeak voice, e.g. nl, nl-be, en-us")
    ap.add_argument("--espeak_pitch", type=int, default=50, help="0-99")
    ap.add_argument("--espeak_amp", type=int, default=100, help="0-200")
    ap.add_argument("--tts_queue_size", type=int, default=10)

    # How often to speak progress (avoid spamming)
    ap.add_argument("--speak_progress_every", type=int, default=5, help="Spreek voortgang om de N samples")

    # capture timing
    ap.add_argument("--capture_interval", type=float, default=0.5, help="Seconds between samples")

    args = ap.parse_args()

    yunet_path = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join("models", "face_recognition_sface_2021dec.onnx")
    download_if_missing(YUNET_URL, yunet_path)
    download_if_missing(SFACE_URL, sface_path)

    os.makedirs(args.outdir, exist_ok=True)

    # Start TTS worker
    stop_event = mp.Event()
    tts_queue = None
    tts_proc = None
    if not args.no_tts:
        tts_queue = mp.Queue(maxsize=args.tts_queue_size)
        tts_proc = mp.Process(target=tts_worker_loop, args=(tts_queue, stop_event, args), daemon=True)
        tts_proc.start()

    # Camera
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Cannot read from camera.")
        stop_event.set()
        if tts_queue is not None:
            try: tts_queue.put_nowait(None)
            except Exception: pass
        if tts_proc is not None:
            tts_proc.join(timeout=0.5)
        return

    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h), args.score_th, args.nms_th, args.topk)
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    win = "Enroll (SFace+YuNet) - druk 'q' om te stoppen"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # state machine
    STATE_WAIT_FACE = "WAIT_FACE"
    STATE_CAPTURING = "CAPTURING"
    STATE_CONFIRM_SAVE = "CONFIRM_SAVE"
    state = STATE_WAIT_FACE

    current_name = ""
    features = []
    last_capture = 0.0
    last_progress_said = 0

    # guidance throttles
    last_no_face_said = 0.0
    last_closer_said = 0.0

    if not args.no_tts:
        tts_enqueue(tts_queue, "Het systeem is gestart.")
        tts_enqueue(tts_queue, "Ik wacht op een gezicht. Kijk naar de camera.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[ERROR] Cannot read frame.")
                break

            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            face = largest_face(faces)

            now = time.time()

            # draw state
            cv2.putText(frame, f"Status: {state}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if state == STATE_WAIT_FACE:
                # Wait until a sufficiently large face is visible
                if face is None:
                    cv2.putText(frame, "Wachten op gezicht...", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    if (not args.no_tts) and (now - last_no_face_said > 6.0):
                        tts_enqueue(tts_queue, "Ik zie geen gezicht. Kijk naar de camera.")
                        last_no_face_said = now
                else:
                    x, y, fw, fh = face[:4].astype(int)
                    if fw < args.min_face:
                        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 165, 255), 2)
                        cv2.putText(frame, "Kom dichterbij...", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                        if (not args.no_tts) and (now - last_closer_said > 4.0):
                            tts_enqueue(tts_queue, "Kom iets dichter bij de camera.")
                            last_closer_said = now
                    else:
                        # We have a face; ask name (blocking input, but TTS keeps speaking)
                        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
                        if not args.no_tts:
                            tts_enqueue(tts_queue, "Ik heb een gezicht gedetecteerd.")
                            tts_enqueue(tts_queue, "Wat is de naam van deze persoon? Typ de naam in de terminal.")

                        # IMPORTANT: freeze this moment, then ask for name
                        cv2.imshow(win, frame)
                        cv2.waitKey(1)

                        name = safe_input("\nNaam van de persoon (ENTER om te annuleren): ").strip()
                        name = sanitize_name(name)
                        if not name:
                            if not args.no_tts:
                                tts_enqueue(tts_queue, "Oké, geannuleerd. Ik wacht opnieuw op een gezicht.")
                            # small cooldown to avoid immediate retrigger
                            time.sleep(0.5)
                            continue

                        current_name = name
                        features = []
                        last_capture = 0.0
                        last_progress_said = 0

                        if not args.no_tts:
                            tts_enqueue(tts_queue, f"Oké {current_name}. Ik ga nu voorbeelden opnemen.")
                            tts_enqueue(tts_queue, "Beweeg langzaam links en rechts, en een beetje omhoog en omlaag.")

                        state = STATE_CAPTURING

            elif state == STATE_CAPTURING:
                msg = f"{current_name}: {len(features)}/{args.samples}"
                cv2.putText(frame, msg, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 200, 40), 2)

                if face is None:
                    cv2.putText(frame, "Gezicht kwijt...", (10, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    if (not args.no_tts) and (now - last_no_face_said > 4.0):
                        tts_enqueue(tts_queue, "Ik ben het gezicht kwijt. Kijk opnieuw naar de camera.")
                        last_no_face_said = now
                else:
                    x, y, fw, fh = face[:4].astype(int)

                    if fw < args.min_face:
                        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 165, 255), 2)
                        cv2.putText(frame, "Kom dichterbij...", (10, 95),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                        if (not args.no_tts) and (now - last_closer_said > 3.0):
                            tts_enqueue(tts_queue, "Kom iets dichter bij de camera.")
                            last_closer_said = now
                    else:
                        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

                        if now - last_capture >= args.capture_interval and len(features) < args.samples:
                            aligned = recognizer.alignCrop(frame, face)
                            feat = recognizer.feature(aligned).astype(np.float32)
                            features.append(feat)
                            last_capture = now

                            if (not args.no_tts) and (len(features) % args.speak_progress_every == 0) and (len(features) != last_progress_said):
                                last_progress_said = len(features)
                                tts_enqueue(tts_queue, f"{len(features)} van {args.samples}.")

                if len(features) >= args.samples:
                    if not args.no_tts:
                        tts_enqueue(tts_queue, "Klaar met opnemen.")
                        tts_enqueue(tts_queue, "Mag ik deze persoon opslaan? Antwoord met j of n in de terminal.")
                    state = STATE_CONFIRM_SAVE

            elif state == STATE_CONFIRM_SAVE:
                # Show final frame and ask confirmation (blocking input)
                cv2.putText(frame, "Opname klaar. Zie terminal: opslaan? (j/n)", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.imshow(win, frame)
                cv2.waitKey(1)

                ans = safe_input(f"\n{current_name} opslaan? (j/n): ").strip().lower()
                if ans.startswith("j"):
                    if len(features) < 8:
                        print("[WARN] Te weinig samples. Neem liever 15–30 samples.")
                        if not args.no_tts:
                            tts_enqueue(tts_queue, "Te weinig voorbeelden. Probeer opnieuw met meer voorbeelden.")
                    else:
                        out_path = os.path.join(args.outdir, f"{current_name}.npz")
                        np.savez_compressed(out_path, features=np.stack(features, axis=0))
                        print(f"[OK] Saved: {out_path} (n={len(features)})")
                        if not args.no_tts:
                            tts_enqueue(tts_queue, f"Opgeslagen. {current_name} is toegevoegd.")
                else:
                    print("[INFO] Not saved.")
                    if not args.no_tts:
                        tts_enqueue(tts_queue, "Oké. Ik sla niets op.")

                # Reset and go back to waiting
                current_name = ""
                features = []
                last_capture = 0.0
                last_progress_said = 0
                last_no_face_said = 0.0
                last_closer_said = 0.0

                if not args.no_tts:
                    tts_enqueue(tts_queue, "Ik wacht opnieuw op een gezicht.")
                state = STATE_WAIT_FACE

            # show
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                if not args.no_tts:
                    tts_enqueue(tts_queue, "Gestopt.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Shutdown TTS worker cleanly
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
