import os
import glob
import csv
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


def iter_images(root: str, recursive: bool = True):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    if recursive:
        for ext in exts:
            yield from glob.glob(os.path.join(root, "**", ext), recursive=True)
    else:
        for ext in exts:
            yield from glob.glob(os.path.join(root, ext))


def largest_face(faces: np.ndarray):
    if faces is None or len(faces) == 0:
        return None
    areas = faces[:, 2] * faces[:, 3]
    return faces[int(np.argmax(areas))]


def load_known(known_dir: str):
    known = {}  # name -> (N, D) features
    if not os.path.isdir(known_dir):
        return known
    for fn in os.listdir(known_dir):
        if fn.lower().endswith(".npz"):
            name = os.path.splitext(fn)[0]
            data = np.load(os.path.join(known_dir, fn))
            feats = data["features"].astype(np.float32)
            known[name] = feats
    return known


def best_match(recognizer, feat: np.ndarray, known: dict):
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


def annotate(img, face_row, label: str):
    x, y, w, h = face_row[:4].astype(int)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label, (max(0, x), max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Map met afbeeldingen om te herkennen")
    ap.add_argument("--known", type=str, default="known", help="Map met .npz known identities")
    ap.add_argument("--models_dir", type=str, default="models", help="Map met ONNX modellen")
    ap.add_argument("--recursive", action="store_true", help="Doorloop submappen")
    ap.add_argument("--min_face", type=int, default=80, help="Min face width in pixels")
    ap.add_argument("--threshold", type=float, default=0.50, help="Cosine similarity threshold")
    ap.add_argument("--margin", type=float, default=0.06, help="Best-second margin")
    ap.add_argument("--csv_out", type=str, default="results.csv", help="CSV output bestand")
    ap.add_argument("--annotate_dir", type=str, default="", help="Als gezet: bewaar geannoteerde images hier")
    ap.add_argument("--score_th", type=float, default=0.9)
    ap.add_argument("--nms_th", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=5000)
    args = ap.parse_args()

    yunet_path = os.path.join(args.models_dir, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(args.models_dir, "face_recognition_sface_2021dec.onnx")
    download_if_missing(YUNET_URL, yunet_path)
    download_if_missing(SFACE_URL, sface_path)

    known = load_known(args.known)
    if not known:
        raise SystemExit(f"[ERROR] No known identities in '{args.known}'. Run build script first.")

    if args.annotate_dir:
        os.makedirs(args.annotate_dir, exist_ok=True)

    detector = cv2.FaceDetectorYN.create(
        yunet_path, "", (320, 320),
        args.score_th, args.nms_th, args.topk
    )
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    img_paths = list(iter_images(args.input_dir, recursive=args.recursive))
    #print (f"[INFO] Found {len(img_paths)} images in '{args.input_dir}' (recursive={args.recursive}).")
    if not img_paths:
        raise SystemExit(f"[ERROR] No images found in '{args.input_dir}' (recursive={args.recursive}).")

    print(f"[INFO] Images: {len(img_paths)}")
    print(f"[INFO] Known: {', '.join(sorted(known.keys()))}")

    rows = []
    n_face = 0
    n_match = 0
    n_unknown = 0
    n_skip = 0

    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            n_skip += 1
            rows.append([p, "READ_FAIL", "", "", "", "", ""])
            continue

        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)
        face = largest_face(faces)

        if face is None:
            rows.append([p, "NO_FACE", "", "", "", "", ""])
            continue

        x, y, fw, fh = face[:4].astype(int)
        if fw < args.min_face:
            rows.append([p, "FACE_TOO_SMALL", "", "", x, y, fw, fh])
            continue

        n_face += 1

        try:
            aligned = recognizer.alignCrop(img, face)
            feat = recognizer.feature(aligned).astype(np.float32)
        except Exception:
            n_skip += 1
            rows.append([p, "FEATURE_FAIL", "", "", x, y, fw, fh])
            continue

        best_name, best_score, second_score = best_match(recognizer, feat, known)
        confident = (best_score >= args.threshold) and ((best_score - second_score) >= args.margin)

        if confident:
            status = "MATCH"
            pred = best_name
            n_match += 1
        else:
            status = "UNKNOWN"
            pred = ""
            n_unknown += 1

        rows.append([p, status, pred, f"{best_score:.4f}", x, y, fw, fh])

        if args.annotate_dir:
            label = pred if status == "MATCH" else f"UNKNOWN ({best_score:.2f})"
            out_img = img.copy()
            out_img = annotate(out_img, face, label)

            # keep folder structure if recursive
            rel = os.path.relpath(p, args.input_dir)
            out_path = os.path.join(args.annotate_dir, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, out_img)

    # Write CSV
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "status", "predicted_name", "best_score", "x", "y", "w", "h"])
        for r in rows:
            writer.writerow(r)

    print("\n[SUMMARY]")
    print(f"  Faces processed: {n_face}")
    print(f"  Matches: {n_match}")
    print(f"  Unknown: {n_unknown}")
    print(f"  Skipped/Fail: {n_skip}")
    print(f"  CSV: {os.path.abspath(args.csv_out)}")
    if args.annotate_dir:
        print(f"  Annotated images: {os.path.abspath(args.annotate_dir)}")


if __name__ == "__main__":
    main()
