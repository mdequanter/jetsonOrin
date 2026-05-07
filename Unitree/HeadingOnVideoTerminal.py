from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: ultralytics. Install it with: pip install ultralytics"
    ) from exc

from HeadingOnVideo import (
    DEFAULT_BOTTOM_CONNECT_RATIO,
    DEFAULT_BOTTOM_SEED_X_RATIO,
    DEFAULT_MODEL_PATH,
    DEFAULT_ROW_WEIGHT_POWER,
    DEFAULT_SCAN_HEIGHTS,
    DEFAULT_VIDEO_PATH,
    calculate_heading,
    count_mask_contours,
    create_row_angle_plan,
    extract_rowwise_midpoints,
    parse_scan_heights,
    segmentation_mask_from_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run terminal-only heading inference on a video."
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
        "--scan-heights",
        default=",".join(str(value) for value in DEFAULT_SCAN_HEIGHTS),
        help="Comma-separated row positions as frame-height ratios, for example 0.2,0.4,0.6.",
    )
    parser.add_argument(
        "--row-weight-power",
        type=float,
        default=DEFAULT_ROW_WEIGHT_POWER,
        help="How strongly lower scan rows steer the heading. Use 0 for equal weights.",
    )
    parser.add_argument(
        "--bottom-connect-ratio",
        type=float,
        default=DEFAULT_BOTTOM_CONNECT_RATIO,
        help="Bottom frame-height ratio used to keep only ground-connected segmentation.",
    )
    parser.add_argument(
        "--bottom-seed-x-ratio",
        type=float,
        default=DEFAULT_BOTTOM_SEED_X_RATIO,
        help="Horizontal frame ratio used as the bottom seed. 0.5 means bottom center.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print every N frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. Use 0 to process the whole video.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop the input video.")
    return parser.parse_args()


def format_heading(heading_angle: float | None) -> str:
    if heading_angle is None:
        return "n/a"
    return f"{heading_angle:.1f} deg"


def main() -> int:
    args = parse_args()
    scan_heights = parse_scan_heights(args.scan_heights)
    print_every = max(args.print_every, 1)

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

    predict_kwargs = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    frame_index = 0
    processed_frames = 0
    started_at = time.perf_counter()

    print(f"model={model_path}")
    print(f"video={video_path}")
    print(
        "columns: frame inference_ms total_ms heading row_angles midpoints ground_contours fps",
        flush=True,
    )

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

            mask = segmentation_mask_from_result(
                result,
                frame.shape[:2],
                args.bottom_connect_ratio,
                args.bottom_seed_x_ratio,
            )
            midpoints = extract_rowwise_midpoints(mask, scan_heights) if mask is not None else []
            heading_angle, _weighted_target, arrow_start = calculate_heading(
                midpoints,
                frame.shape[:2],
                args.row_weight_power,
            )
            row_angle_plan, _row_angles = create_row_angle_plan(midpoints, arrow_start)
            contour_count = count_mask_contours(mask)
            processed_frames += 1

            if frame_index % print_every == 0:
                elapsed = max(time.perf_counter() - started_at, 0.001)
                fps = processed_frames / elapsed
                print(
                    f"frame={frame_index} "
                    f"inference_ms={inference_ms:.1f} "
                    f"total_ms={total_ms:.1f} "
                    f"heading=\"{format_heading(heading_angle)}\" "
                    f"row_angles=\"{row_angle_plan}\" "
                    f"midpoints={len(midpoints)} "
                    f"ground_contours={contour_count} "
                    f"fps={fps:.1f}",
                    flush=True,
                )

            frame_index += 1
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("Stopped by user.", flush=True)
    finally:
        cap.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
