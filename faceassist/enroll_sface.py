import os
import time
import argparse
import urllib.request
import numpy as np
import cv2

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
    # faces rows: x, y, w, h, score, 10 landmark coords (5 points)
    if faces is None or len(faces) == 0:
        return None
    areas = faces[:, 2] * faces[:, 3]
    return faces[int(np.argmax(areas))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Naam van de persoon (bv. Tom)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--outdir", type=str, default="known")
    ap.add_argument("--min_face", type=int, default=120, help="Min face width in pixels")
    ap.add_argument("--score_th", type=float, default=0.9)
    ap.add_argument("--nms_th", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=5000)
    args = ap.parse_args()

    yunet_path = os.path.join("models", "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join("models", "face_recognition_sface_2021dec.onnx")
    download_if_missing(YUNET_URL, yunet_path)
    download_if_missing(SFACE_URL, sface_path)

    os.makedirs(args.outdir, exist_ok=True)

    #cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW) # for Windows to avoid warning about DirectShow; on Linux/Mac this flag is ignored
    cap = cv2.VideoCapture(args.cam)
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

    features = []
    last_capture = 0.0

    print(f"[INFO] Enrolling '{args.name}' with SFace+YuNet.")
    print("[INFO] Variëer hoek/licht (links/rechts, iets omhoog/omlaag).")
    print("[INFO] Druk 'q' om te stoppen.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Cannot read frame.")
            break

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        face = largest_face(faces)
        msg = f"Samples: {len(features)}/{args.samples}"
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 200, 40), 2)

        if face is not None:
            x, y, fw, fh = face[:4].astype(int)
            if fw >= args.min_face:
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

                now = time.time()
                if now - last_capture > 0.5 and len(features) < args.samples:
                    aligned = recognizer.alignCrop(frame, face)
                    feat = recognizer.feature(aligned).astype(np.float32)
                    features.append(feat)
                    last_capture = now
            else:
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 165, 255), 2)
                cv2.putText(frame, "Kom dichterbij", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "Geen gezicht", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Enroll (SFace+YuNet)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if len(features) >= args.samples:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(features) < 8:
        print("[WARN] Te weinig samples. Neem liever 15–30 samples.")
        return

    out_path = os.path.join(args.outdir, f"{args.name}.npz")
    np.savez_compressed(out_path, features=np.stack(features, axis=0))
    print(f"[OK] Saved: {out_path} (n={len(features)})")

if __name__ == "__main__":
    main()
