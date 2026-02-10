import os
import time
import argparse
import urllib.request
import numpy as np
import cv2

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
    known = {}  # name -> (N, 1, D) features
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
    # returns best_name, best_score, second_score (cosine similarity; higher is better)
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
    print("[SAY]", text)
    if engine is None:
        return
    engine.say(text)
    engine.runAndWait()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--known", type=str, default="known")
    ap.add_argument("--min_face", type=int, default=120)
    ap.add_argument("--threshold", type=float, default=0.50, help="Cosine similarity threshold (higher = stricter)")
    ap.add_argument("--margin", type=float, default=0.06, help="Best-second margin")
    ap.add_argument("--cooldown", type=float, default=6.0)
    ap.add_argument("--score_th", type=float, default=0.9)
    ap.add_argument("--nms_th", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=5000)
    ap.add_argument("--infer_every", type=int, default=2, help="Run detection/recognition every N frames")
    args = ap.parse_args()

    yunet_path = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join("models", "face_recognition_sface_2021dec.onnx")
    download_if_missing(YUNET_URL, yunet_path)
    download_if_missing(SFACE_URL, sface_path)

    known = load_known(args.known)
    if not known:
        print(f"[ERROR] No known identities in '{args.known}'. Run enroll_sface.py first.")
        return
    print("[INFO] Known:", ", ".join(sorted(known.keys())))

    engine = None
    if TTS_OK:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    ok, frame = cap.read()
    if not ok:
        print("[ERROR] Cannot read from camera.")
        return

    h, w = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        yunet_path, "", (w, h),
        args.score_th, args.nms_th, args.topk
    )
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    last_spoken = {}  # name -> timestamp
    frame_id = 0
    last_label = "..."

    print("[INFO] Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        if frame_id % args.infer_every != 0:
            cv2.putText(frame, last_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            #cv2.imshow("Recognize (SFace+YuNet)", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            continue

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        face = largest_face(faces)

        label = "Geen gezicht"
        color = (0, 0, 255)

        if face is not None:
            x, y, fw, fh = face[:4].astype(int)
            if fw >= args.min_face:
                aligned = recognizer.alignCrop(frame, face)
                feat = recognizer.feature(aligned).astype(np.float32)

                best_name, best_score, second_score = best_match(recognizer, feat, known)
                confident = (best_score >= args.threshold) and ((best_score - second_score) >= args.margin)

                if confident:
                    label = f"{best_name} ({best_score:.2f})"
                    color = (0, 255, 0)
                    now = time.time()
                    last = last_spoken.get(best_name, 0.0)
                    if now - last >= args.cooldown:
                        speak(engine, f"Ik denk dat het {best_name} is")
                        last_spoken[best_name] = now
                else:
                    label = f"Onbekend ({best_score:.2f})"
                    color = (0, 165, 255)

                cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
            else:
                label = "Kom dichterbij"
                color = (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

        last_label = label

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Recognize (SFace+YuNet)", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
