from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: ultralytics. Install it with: pip install ultralytics"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = "/home/jetson/Models/KaaiGang.pt"
DEFAULT_VIDEO_PATH = SCRIPT_DIR / "Videos" / "gangKaai.mp4"
WINDOW_NAME = "Gang Kaai segmentation"

# BGR colors for OpenCV.
COLORS = (
    (0, 255, 255),
    (255, 0, 255),
    (0, 180, 255),
    (80, 220, 80),
    (255, 140, 0),
    (220, 120, 220),
    (120, 220, 255),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show Ultralytics segmentation polygons and inference time."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"YOLO segmentation model path. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--video",
        default=str(DEFAULT_VIDEO_PATH),
        help=f"Input video path. Default: {DEFAULT_VIDEO_PATH}",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument(
        "--device",
        default=None,
        help='Ultralytics device, for example "cpu", "0", or "0,1".',
    )
    parser.add_argument(
        "--alpha", type=float, default=0.35, help="Mask fill opacity from 0.0 to 1.0."
    )
    parser.add_argument("--line-width", type=int, default=2, help="Polygon line width.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. Use 0 to process the whole video.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop the input video.")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open an OpenCV window. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--save-output",
        default=None,
        help="Optional path to save an annotated video.",
    )
    return parser.parse_args()


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def draw_text_block(frame: np.ndarray, lines: list[str], origin=(12, 28)) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    line_gap = 8
    padding = 8

    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    box_width = max(width for width, _ in sizes) + padding * 2
    box_height = sum(height for _, height in sizes) + line_gap * (len(lines) - 1) + padding * 2

    x, y = origin
    cv2.rectangle(
        frame,
        (x - padding, y - sizes[0][1] - padding),
        (x - padding + box_width, y - sizes[0][1] - padding + box_height),
        (0, 0, 0),
        -1,
    )

    cursor_y = y
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, cursor_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        if index + 1 < len(lines):
            cursor_y += sizes[index][1] + line_gap


def draw_label(frame: np.ndarray, text: str, anchor: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    padding = 5

    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, anchor[0])
    y = max(text_size[1] + padding * 2, anchor[1])

    cv2.rectangle(
        frame,
        (x, y - text_size[1] - padding * 2),
        (x + text_size[0] + padding * 2, y + baseline),
        color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + padding, y - padding),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def result_arrays(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return [], []

    classes = []
    confidences = []
    if boxes.cls is not None:
        classes = boxes.cls.detach().cpu().numpy().astype(int).tolist()
    if boxes.conf is not None:
        confidences = boxes.conf.detach().cpu().numpy().tolist()
    return classes, confidences


def draw_segmentation_polygons(
    frame: np.ndarray,
    result,
    alpha: float,
    line_width: int,
) -> int:
    masks = getattr(result, "masks", None)
    if masks is None or masks.xy is None:
        return 0

    height, width = frame.shape[:2]
    overlay = frame.copy()
    detections = []
    classes, confidences = result_arrays(result)
    names = getattr(result, "names", {})

    for index, polygon in enumerate(masks.xy):
        if polygon is None or len(polygon) < 3:
            continue

        points = np.round(polygon).astype(np.int32).reshape((-1, 1, 2))
        points[:, 0, 0] = np.clip(points[:, 0, 0], 0, width - 1)
        points[:, 0, 1] = np.clip(points[:, 0, 1], 0, height - 1)

        class_id = classes[index] if index < len(classes) else index
        confidence = confidences[index] if index < len(confidences) else None
        color = COLORS[class_id % len(COLORS)]

        cv2.fillPoly(overlay, [points], color)
        detections.append((points, class_id, confidence, color))

    alpha = min(1.0, max(0.0, alpha))
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)

    for points, class_id, confidence, color in detections:
        cv2.polylines(frame, [points], True, color, line_width, cv2.LINE_AA)
        min_x = int(points[:, 0, 0].min())
        min_y = int(points[:, 0, 1].min())
        label = class_name(names, class_id)
        if confidence is not None:
            label = f"{label} {confidence:.2f}"
        draw_label(frame, label, (min_x, min_y), color)

    return len(detections)


def create_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if suffix != ".avi" else "XVID"))
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {output_path}")
    return writer


def main() -> int:
    args = parse_args()

    model_path = Path(args.model).expanduser()
    video_path = Path(args.video).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = YOLO(str(model_path), verbose=False)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None

    if args.save_output:
        writer = create_video_writer(
            Path(args.save_output).expanduser(),
            fps,
            (frame_width, frame_height),
        )

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    predict_kwargs = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            start = time.perf_counter()
            result = model.predict(frame, **predict_kwargs)[0]
            total_ms = (time.perf_counter() - start) * 1000.0
            inference_ms = result.speed.get("inference", total_ms) if result.speed else total_ms

            mask_count = draw_segmentation_polygons(
                frame,
                result,
                alpha=args.alpha,
                line_width=args.line_width,
            )
            draw_text_block(
                frame,
                [
                    f"Inference: {inference_ms:.1f} ms",
                    f"Total: {total_ms:.1f} ms | Masks: {mask_count}",
                ],
            )

            if writer is not None:
                writer.write(frame)

            if args.no_display:
                print(
                    f"frame={frame_index} inference_ms={inference_ms:.1f} "
                    f"total_ms={total_ms:.1f} masks={mask_count}",
                    flush=True,
                )
            else:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            frame_index += 1
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
