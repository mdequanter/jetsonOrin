import argparse
import csv
import os
import re
import urllib.request
from pathlib import Path

import cv2


YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_if_missing(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    print(f"[INFO] Downloading {path.name} ...")
    urllib.request.urlretrieve(url, path)
    print(f"[OK] Saved to {path}")


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._") or "image"


def iter_images(input_dir: Path, recursive: bool, output_dir: Path):
    walker = input_dir.rglob("*") if recursive else input_dir.glob("*")
    output_dir = output_dir.resolve()

    for path in walker:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        try:
            resolved = path.resolve()
            if output_dir in resolved.parents or resolved == output_dir:
                continue
        except OSError:
            pass

        yield path


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def crop_face(img, face, margin: float):
    h, w = img.shape[:2]
    x, y, fw, fh = face[:4]

    pad_x = fw * margin
    pad_y = fh * margin
    x1 = max(0, int(round(x - pad_x)))
    y1 = max(0, int(round(y - pad_y)))
    x2 = min(w, int(round(x + fw + pad_x)))
    y2 = min(h, int(round(y + fh + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)

    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def main():
    base_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(
        description="Extract every detected face from pictures in a folder into one output folder."
    )
    ap.add_argument("input_dir", type=Path, help="Folder with pictures to scan")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=base_dir / "extracted_faces",
        help="Folder where cropped face images are saved (default: faceassist/extracted_faces)",
    )
    ap.add_argument(
        "--models_dir",
        type=Path,
        default=base_dir / "models",
        help="Folder containing the YuNet ONNX model (default: faceassist/models)",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Also scan subfolders of input_dir",
    )
    ap.add_argument(
        "--min_face",
        type=int,
        default=40,
        help="Minimum detected face width and height in pixels (default: 40)",
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=0.25,
        help="Extra crop margin around each face as a fraction of face size (default: 0.25)",
    )
    ap.add_argument(
        "--score_th",
        type=float,
        default=0.9,
        help="YuNet confidence threshold (default: 0.9)",
    )
    ap.add_argument(
        "--nms_th",
        type=float,
        default=0.3,
        help="YuNet non-maximum suppression threshold (default: 0.3)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=5000,
        help="YuNet topK detections (default: 5000)",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV manifest path (default: <outdir>/manifest.csv)",
    )
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.outdir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"[ERROR] input_dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    yunet_path = args.models_dir / "face_detection_yunet_2023mar.onnx"
    download_if_missing(YUNET_URL, yunet_path)

    detector = cv2.FaceDetectorYN.create(
        str(yunet_path),
        "",
        (320, 320),
        args.score_th,
        args.nms_th,
        args.topk,
    )

    manifest_path = (args.manifest or output_dir / "manifest.csv").resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    images_seen = 0
    images_with_faces = 0
    faces_saved = 0
    skipped_images = 0
    skipped_faces = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "output_file",
                "source_file",
                "face_index",
                "score",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )

        for img_path in iter_images(input_dir, args.recursive, output_dir):
            images_seen += 1
            img = cv2.imread(str(img_path))
            if img is None:
                skipped_images += 1
                print(f"[WARN] Could not read image: {img_path}")
                continue

            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            if faces is None or len(faces) == 0:
                continue

            faces = sorted(faces, key=lambda f: float(f[4]), reverse=True)
            saved_for_image = 0
            rel = img_path.relative_to(input_dir)
            source_stem = safe_name("__".join(rel.with_suffix("").parts))

            for face_index, face in enumerate(faces, start=1):
                x, y, fw, fh = face[:4]
                if fw < args.min_face or fh < args.min_face:
                    skipped_faces += 1
                    continue

                crop, (x1, y1, x2, y2) = crop_face(img, face, args.margin)
                if crop is None:
                    skipped_faces += 1
                    continue

                score = float(face[4])
                filename = f"{source_stem}_face{face_index:02d}_s{score:.3f}.jpg"
                out_path = unique_path(output_dir / filename)

                if not cv2.imwrite(str(out_path), crop):
                    skipped_faces += 1
                    print(f"[WARN] Could not write crop: {out_path}")
                    continue

                faces_saved += 1
                saved_for_image += 1
                writer.writerow(
                    [
                        out_path.name,
                        str(img_path),
                        face_index,
                        f"{score:.6f}",
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                )

            if saved_for_image:
                images_with_faces += 1
                print(f"[OK] {img_path}: saved {saved_for_image} face(s)")

    print("\n[SUMMARY]")
    print(f"  Images scanned: {images_seen}")
    print(f"  Images with saved faces: {images_with_faces}")
    print(f"  Face crops saved: {faces_saved}")
    print(f"  Images skipped/unreadable: {skipped_images}")
    print(f"  Faces skipped/small: {skipped_faces}")
    print(f"  Output folder: {output_dir}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
