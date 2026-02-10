import os
import glob
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
    # faces rows: x, y, w, h, score, 10 landmark coords
    if faces is None or len(faces) == 0:
        return None
    areas = faces[:, 2] * faces[:, 3]
    return faces[int(np.argmax(areas))]


def iter_images(folder: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    for ext in exts:
        for p in glob.glob(os.path.join(folder, ext)):
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class_dir", type=str, default="class",
                    help="Root folder met subfolders per persoon (default: class)")
    ap.add_argument("--outdir", type=str, default="known",
                    help="Output folder voor .npz (default: known)")
    ap.add_argument("--models_dir", type=str, default="models",
                    help="Waar ONNX modellen staan/gedownload worden (default: models)")
    ap.add_argument("--min_face", type=int, default=80,
                    help="Minimum face width in pixels (default 80)")
    ap.add_argument("--score_th", type=float, default=0.9,
                    help="YuNet score threshold (default 0.9)")
    ap.add_argument("--nms_th", type=float, default=0.3,
                    help="YuNet NMS threshold (default 0.3)")
    ap.add_argument("--topk", type=int, default=5000,
                    help="YuNet topK (default 5000)")
    ap.add_argument("--max_per_person", type=int, default=0,
                    help="Max embeddings per persoon (0 = geen limiet)")
    ap.add_argument("--save_debug", action="store_true",
                    help="Slaat debug crops op in known/_debug/<persoon>/")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print ("xxxxxx")

    yunet_path = os.path.join(args.models_dir, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(args.models_dir, "face_recognition_sface_2021dec.onnx")
    download_if_missing(YUNET_URL, yunet_path)
    download_if_missing(SFACE_URL, sface_path)

    # Initialize once; setInputSize will be updated per image
    # For create(), input size must be set but can be overridden later with setInputSize
    detector = cv2.FaceDetectorYN.create(
        yunet_path, "", (320, 320),
        args.score_th, args.nms_th, args.topk
    )
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    class_root = args.class_dir
    if not os.path.isdir(class_root):
        raise SystemExit(f"[ERROR] class_dir not found: {class_root}")

    people = [d for d in os.listdir(class_root) if os.path.isdir(os.path.join(class_root, d))]
    if not people:
        raise SystemExit(f"[ERROR] No person subfolders found in: {class_root}")

    total_people = 0
    total_imgs = 0
    total_used = 0

    print(f"[INFO] Found {len(people)} person folders in '{class_root}'.")

    for person in sorted(people):
        person_dir = os.path.join(class_root, person)
        img_paths = list(iter_images(person_dir))
        if not img_paths:
            print(f"[WARN] {person}: no images found, skipping.")
            continue

        feats = []
        used = 0
        skipped = 0

        debug_dir = os.path.join(args.outdir, "_debug", person)
        if args.save_debug:
            os.makedirs(debug_dir, exist_ok=True)

        for img_path in img_paths:
            total_imgs += 1
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            face = largest_face(faces)
            if face is None:
                skipped += 1
                continue

            x, y, fw, fh = face[:4].astype(int)
            if fw < args.min_face:
                skipped += 1
                continue

            try:
                aligned = recognizer.alignCrop(img, face)
                feat = recognizer.feature(aligned).astype(np.float32)
            except Exception:
                skipped += 1
                continue

            feats.append(feat)
            used += 1
            total_used += 1

            if args.save_debug:
                # Save aligned face crop for inspection
                base = os.path.splitext(os.path.basename(img_path))[0]
                out_crop = os.path.join(debug_dir, f"{base}_crop.jpg")
                cv2.imwrite(out_crop, aligned)

            if args.max_per_person > 0 and used >= args.max_per_person:
                break

        if used < 3:
            print(f"[WARN] {person}: only {used} usable faces (skipped {skipped}). Not saving.")
            continue

        out_path = os.path.join(args.outdir, f"{person}.npz")
        np.savez_compressed(out_path, features=np.stack(feats, axis=0))
        total_people += 1

        print(f"[OK] {person}: saved {used} features to {out_path} (skipped {skipped}, total imgs {len(img_paths)})")

    print("\n[SUMMARY]")
    print(f"  People saved: {total_people}")
    print(f"  Total images seen: {total_imgs}")
    print(f"  Total features saved: {total_used}")
    print(f"  Output folder: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
