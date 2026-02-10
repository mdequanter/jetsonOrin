import os
import time
import argparse
import urllib.request
import numpy as np
import cv2
import multiprocessing as mp
import signal

try:
    import pyttsx3
    TTS_OK = True
except Exception:
    TTS_OK = False

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def download_if_missing(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    print(f"[INFO] Downloading {os.path.basename(path)} ...")
    urllib.request.urlretrieve(url, path)
    print(f"[OK] Saved to {path}")


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


def speak(engine, text: str):
    """Blocking TTS call. Keep usage throttled with cooldown."""
    print("[TTS]", text)
    if engine is None:
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("[WARN] TTS failed:", e)


def face_direction(x: int, w_face: int, frame_w: int) -> str:
    """
    Return left/center/right based on face center position in the frame.
    """
    cx = x + (w_face // 2)
    if cx < frame_w / 3:
        return "left"
    elif cx > 2 * frame_w / 3:
        return "right"
    return "center"


def open_camera_linux(cam_index: int, width: int, height: int, fps: int):
    dev = f"/dev/video{cam_index}"

    # MJPEG GStreamer (typical for USB cams on Jetson)
    gst_pipeline = (
        f"v4l2src device={dev} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"jpegdec ! videoconvert ! appsink drop=true sync=false max-buffers=1"
    )

    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[INFO] Camera opened via GStreamer.")
        return cap

    print("[WARN] GStreamer open failed. Falling back to V4L2...")

    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        print("[INFO] Camera opened via V4L2 (OpenCV).")
        return cap

    cap = cv2.VideoCapture(cam_index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        print("[INFO] Camera opened via default backend (OpenCV).")
        return cap

    return cap


def worker_loop(args, stop_event: mp.Event):
    """
    Vision loop runs in a separate process so Ctrl+C always works from the parent.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Models
    yunet_path = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join("models", "face_recognition_sface_2021dec.onnx")

    known = load_known(args.known)
    if not known:
        print(f"[ERROR] No known identities in '{args.known}'. Run enroll_sface.py first.")
        return
    print("[INFO] Known:", ", ".join(sorted(known.keys())))

    # TTS engine (English)
    engine = None
    if TTS_OK and not args.no_tts:
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", args.tts_rate)
            # Optional: try to pick an English voice if present
            if args.force_en_voice:
                try:
                    for v in engine.getProperty("voices"):
                        # voice.languages can vary by platform; do a best-effort check
                        langs = str(getattr(v, "languages", "")).lower()
                        name = str(getattr(v, "name", "")).lower()
                        vid = str(getattr(v, "id", "")).lower()
                        if "en" in langs or "english" in name or "english" in vid:
                            engine.setProperty("voice", v.id)
                            break
                except Exception:
                    pass
        except Exception as e:
            print("[WARN] TTS init failed:", e)
            engine = None

    # Camera
    WIDTH, HEIGHT, FPS = args.width, args.height, args.fps
    cap = open_camera_linux(args.cam, WIDTH, HEIGHT, FPS)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    ok, frame = cap.read()
    if not ok or frame is None:
        print("[ERROR] Cannot read first frame.")
        cap.release()
        return

    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h), args.score_th, args.nms_th, args.topk)
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    last_spoken = {}  # key -> timestamp, where key = (name, direction) or ("Unknown", direction)

    frame_id = 0
    last_label = "..."

    win = "Recognize (SFace+YuNet)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, WIDTH, HEIGHT)
    cv2.moveWindow(win, 50, 50)

    print("[INFO] Running. Stop with Ctrl+C in the terminal.")

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Frame grab failed.")
                break
            frame_id += 1

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

            if frame_id % args.infer_every != 0:
                cv2.putText(frame, last_label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
                cv2.imshow(win, cv2.resize(frame, (WIDTH, HEIGHT)))
                cv2.waitKey(1)
                continue

            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
            face = largest_face(faces)

            label = "No face"
            color = (0, 0, 255)

            if face is not None:
                x, y, fw, fh = face[:4].astype(int)

                direction = face_direction(x, fw, w)  # left/center/right

                if fw >= args.min_face:
                    aligned = recognizer.alignCrop(frame, face)
                    feat = recognizer.feature(aligned).astype(np.float32)

                    best_name, best_score, second_score = best_match(recognizer, feat, known)
                    confident = (best_score >= args.threshold) and ((best_score - second_score) >= args.margin)

                    if confident:
                        label = f"{best_name} ({best_score:.2f}) - {direction}"
                        color = (0, 255, 0)

                        if engine is not None:
                            now = time.time()
                            key = (best_name, direction)
                            last = last_spoken.get(key, 0.0)
                            if now - last >= args.cooldown:
                                # English TTS message
                                speak(engine, f"{best_name} is on the {direction}")
                                last_spoken[key] = now
                    else:
                        label = f"Unknown ({best_score:.2f}) - {direction}"
                        color = (0, 165, 255)

                        if engine is not None and args.speak_unknown:
                            now = time.time()
                            key = ("Unknown", direction)
                            last = last_spoken.get(key, 0.0)
                            if now - last >= args.cooldown:
                                speak(engine, f"Unknown person on the {direction}")
                                last_spoken[key] = now

                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
                else:
                    label = f"Come closer - {direction}"
                    color = (0, 165, 255)
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

            last_label = label
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow(win, cv2.resize(frame, (WIDTH, HEIGHT)))
            cv2.waitKey(1)

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--known", type=str, default="known")
    ap.add_argument("--min_face", type=int, default=120)
    ap.add_argument("--threshold", type=float, default=0.50, help="Cosine similarity threshold (higher=stricter)")
    ap.add_argument("--margin", type=float, default=0.06, help="Best-second margin")
    ap.add_argument("--cooldown", type=float, default=6.0, help="Seconds between repeated TTS for same (name,direction)")
    ap.add_argument("--score_th", type=float, default=0.9)
    ap.add_argument("--nms_th", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=5000)
    ap.add_argument("--infer_every", type=int, default=2)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)

    # TTS options
    ap.add_argument("--no_tts", action="store_true", help="Disable TTS")
    ap.add_argument("--tts_rate", type=int, default=175)
    ap.add_argument("--force_en_voice", action="store_true", help="Try to select an English voice if available")
    ap.add_argument("--speak_unknown", action="store_true", help="Also speak when person is unknown")

    args = ap.parse_args()

    yunet_path = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join("models", "face_recognition_sface_2021dec.onnx")
    download_if_missing(YUNET_URL, yunet_path)
    download_if_missing(SFACE_URL, sface_path)

    stop_event = mp.Event()
    p = mp.Process(target=worker_loop, args=(args, stop_event), daemon=True)
    p.start()

    print("[INFO] Ctrl+C to stop.")
    try:
        while p.is_alive():
            p.join(timeout=0.2)
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received, stopping...")
        stop_event.set()
        p.join(timeout=1.0)
        if p.is_alive():
            print("[WARN] Worker stuck in OpenCV call. Terminating process.")
            p.terminate()
            p.join()

    print("[INFO] Exited.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
