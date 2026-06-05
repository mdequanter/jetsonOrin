from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
from aiortc import MediaStreamTrack
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: ultralytics. Install it with: pip install ultralytics"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CANDIDATES = (
    SCRIPT_DIR / "models" / "KaaiGang.pt",
    SCRIPT_DIR.parent / "faceassist" / "models" / "KaaiGang.pt",
    Path("/home/jetson/Models/KaaiGang.pt"),
)
DEFAULT_MODEL_PATH = next(
    (path for path in DEFAULT_MODEL_CANDIDATES if path.exists()),
    DEFAULT_MODEL_CANDIDATES[0],
)
DEFAULT_VIDEO_PATH = SCRIPT_DIR / "Videos" / "gangKaai.mp4"
WINDOW_NAME = "Gang Kaai heading"
DEFAULT_SCAN_HEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DEFAULT_ROW_WEIGHT_POWER = 2.0
DEFAULT_LATERAL_STEP_RATIO = 0.05
DEFAULT_DISPLAY_SCALE = 1.0
DEFAULT_BOTTOM_CONNECT_RATIO = 0.08
DEFAULT_BOTTOM_SEED_X_RATIO = 0.5

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
        "--lateral-step-ratio",
        type=float,
        default=DEFAULT_LATERAL_STEP_RATIO,
        help="Frame-width ratio used to color row arrows as left, right, or straight.",
    )
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
    parser.add_argument(
        "--display-scale",
        type=float,
        default=DEFAULT_DISPLAY_SCALE,
        help="Scale factor for the preview window. This does not change inference or saved video size.",
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
    return parser.parse_args()


def parse_scan_heights(value: str) -> list[float]:
    heights = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        height = float(part)
        if not 0.0 < height < 1.0:
            raise ValueError(f"Scan height must be between 0 and 1: {height}")
        heights.append(height)

    if not heights:
        raise ValueError("At least one scan height is required.")
    return heights


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


def keep_bottom_connected_components(
    mask: np.ndarray,
    bottom_connect_ratio: float = DEFAULT_BOTTOM_CONNECT_RATIO,
    bottom_seed_x_ratio: float = DEFAULT_BOTTOM_SEED_X_RATIO,
) -> np.ndarray | None:
    binary_mask = (mask > 0).astype(np.uint8)
    if cv2.countNonZero(binary_mask) == 0:
        return None

    height, width = binary_mask.shape[:2]
    bottom_rows = max(1, int(height * max(bottom_connect_ratio, 0.001)))
    bottom_start = max(0, height - bottom_rows)
    seed_x = int(np.clip(bottom_seed_x_ratio, 0.0, 1.0) * (width - 1))

    label_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )
    if label_count <= 1:
        return None

    bottom_labels = np.unique(labels[bottom_start:, :][binary_mask[bottom_start:, :] > 0])
    bottom_labels = bottom_labels[bottom_labels != 0]
    if len(bottom_labels) == 0:
        return None

    best_label = None
    best_score = None
    for label in bottom_labels:
        bottom_y, bottom_x = np.where(labels[bottom_start:, :] == label)
        if len(bottom_x) == 0:
            continue

        x_distance = int(np.min(np.abs(bottom_x - seed_x)))
        area = int(stats[label, cv2.CC_STAT_AREA])
        score = (x_distance, -area)
        if best_score is None or score < best_score:
            best_label = label
            best_score = score

    if best_label is None:
        return None

    connected_mask = (labels == best_label).astype(np.uint8) * 255
    return connected_mask


def segmentation_mask_from_result(
    result,
    frame_size: tuple[int, int],
    bottom_connect_ratio: float = DEFAULT_BOTTOM_CONNECT_RATIO,
    bottom_seed_x_ratio: float = DEFAULT_BOTTOM_SEED_X_RATIO,
) -> np.ndarray | None:
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None or len(masks.data) == 0:
        return None

    height, width = frame_size
    mask_data = masks.data.detach().cpu().numpy()
    merged_mask = np.any(mask_data > 0.5, axis=0).astype(np.uint8) * 255
    resized_mask = cv2.resize(merged_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return keep_bottom_connected_components(
        resized_mask,
        bottom_connect_ratio,
        bottom_seed_x_ratio,
    )


def contours_from_mask(mask: np.ndarray | None) -> list[np.ndarray]:
    if mask is None:
        return []
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv2.contourArea(contour) > 10.0]


def count_mask_contours(mask: np.ndarray | None) -> int:
    return len(contours_from_mask(mask))


def draw_connected_mask_contours(
    frame: np.ndarray,
    mask: np.ndarray | None,
    alpha: float,
    line_width: int,
) -> int:
    contours = contours_from_mask(mask)
    if not contours:
        return 0

    overlay = frame.copy()
    color = (0, 255, 255)
    cv2.drawContours(overlay, contours, -1, color, cv2.FILLED)
    alpha = min(1.0, max(0.0, alpha))
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)
    cv2.drawContours(frame, contours, -1, color, line_width, cv2.LINE_AA)
    return len(contours)


def extract_rowwise_midpoints(mask: np.ndarray, scan_heights: list[float]) -> list[tuple[int, int]]:
    height = mask.shape[0]
    midpoints = []

    for row_ratio in scan_heights:
        y = int(height * row_ratio)
        if y >= height:
            continue

        filled_x = np.where(mask[y, :] > 0)[0]
        if len(filled_x) == 0:
            continue

        midpoints.append((int(np.mean(filled_x)), y))

    return midpoints


def calculate_heading(
    midpoints: list[tuple[int, int]],
    frame_size: tuple[int, int],
    row_weight_power: float,
) -> tuple[float | None, tuple[int, int] | None, tuple[int, int]]:
    height, width = frame_size
    start = (width // 2, height - 1)
    if not midpoints:
        return None, None, start

    y_positions = np.array([point[1] for point in midpoints], dtype=np.float32)
    x_positions = np.array([point[0] for point in midpoints], dtype=np.float32)
    if row_weight_power <= 0:
        weights = np.ones_like(y_positions)
    else:
        weights = np.power(np.maximum(y_positions / max(height - 1, 1), 0.001), row_weight_power)

    target = (
        int(np.average(x_positions, weights=weights)),
        min(point[1] for point in midpoints),
    )
    dx = target[0] - start[0]
    dy = start[1] - target[1]
    angle = float(np.degrees(np.arctan2(dy, dx)))
    return angle, target, start


def order_midpoints_near_to_far(midpoints: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(midpoints, key=lambda point: point[1], reverse=True)


def first_near_midpoint(midpoints: list[tuple[int, int]]) -> tuple[int, int] | None:
    ordered_points = order_midpoints_near_to_far(midpoints)
    if not ordered_points:
        return None
    return ordered_points[0]


def lateral_step_pixels(frame_width: int, lateral_step_ratio: float) -> float:
    return max(1.0, frame_width * max(lateral_step_ratio, 0.001))


def segment_angle_degrees(start: tuple[int, int], end: tuple[int, int]) -> float:
    dx = end[0] - start[0]
    dy = start[1] - end[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def create_row_angle_plan(
    midpoints: list[tuple[int, int]],
    start_point: tuple[int, int] | None = None,
) -> tuple[str, list[float]]:
    ordered_points = order_midpoints_near_to_far(midpoints)
    if start_point is not None:
        ordered_points = [start_point, *ordered_points]

    if len(ordered_points) < 2:
        return "Rows: n/a", []

    angles = [
        segment_angle_degrees(start, end)
        for start, end in zip(ordered_points, ordered_points[1:])
    ]
    angle_text = ", ".join(
        f"{index}={angle:.1f}"
        for index, angle in enumerate(angles, start=1)
    )
    return f"Rows: {angle_text} deg", angles


def draw_row_transition_arrows(
    frame: np.ndarray,
    midpoints: list[tuple[int, int]],
    lateral_step_ratio: float,
) -> None:
    ordered_points = order_midpoints_near_to_far(midpoints)
    if len(ordered_points) < 2:
        return

    step_px = lateral_step_pixels(frame.shape[1], lateral_step_ratio)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for index, (start, end) in enumerate(zip(ordered_points, ordered_points[1:]), start=1):
        dx = end[0] - start[0]
        if dx > step_px * 0.5:
            color = (0, 165, 255)
        elif dx < -step_px * 0.5:
            color = (255, 180, 0)
        else:
            color = (0, 255, 255)

        cv2.arrowedLine(frame, start, end, color, 3, cv2.LINE_AA, tipLength=0.28)

        label_x = int((start[0] + end[0]) / 2)
        label_y = int((start[1] + end[1]) / 2)
        label = str(index)
        text_size, baseline = cv2.getTextSize(label, font, 0.5, 2)
        cv2.rectangle(
            frame,
            (label_x - 6, label_y - text_size[1] - 6),
            (label_x + text_size[0] + 6, label_y + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            font,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


def draw_rowwise_midpoints(frame: np.ndarray, midpoints: list[tuple[int, int]]) -> None:
    if not midpoints:
        return

    height, width = frame.shape[:2]
    ordered_points = order_midpoints_near_to_far(midpoints)

    for x, y in ordered_points:
        cv2.line(frame, (0, y), (width - 1, y), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_start_to_first_midpoint_arrow(
    frame: np.ndarray,
    start: tuple[int, int],
    target: tuple[int, int] | None,
) -> None:
    if target is None:
        return

    cv2.arrowedLine(frame, start, target, (0, 255, 0), 4, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(frame, start, 7, (0, 255, 0), -1, cv2.LINE_AA)


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


def resize_for_display(frame: np.ndarray, display_scale: float) -> np.ndarray:
    if abs(display_scale - 1.0) < 0.001:
        return frame
    return cv2.resize(
        frame,
        None,
        fx=display_scale,
        fy=display_scale,
        interpolation=cv2.INTER_LINEAR,
    )


def main() -> int:
    args = parse_args()
    scan_heights = parse_scan_heights(args.scan_heights)
    display_scale = max(args.display_scale, 0.1)

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path), verbose=False)
    frame_queue = Queue()

    logging.basicConfig(level=logging.FATAL)
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip="unitree.local")
    writer = None

    height, width = 720, 1280
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imshow(WINDOW_NAME, img)
    cv2.waitKey(1)

    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    predict_kwargs = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            frame_queue.put(img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)

        async def setup():
            try:
                await conn.connect()
                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(recv_camera_stream)
            except Exception as e:
                logging.error(f"Error in WebRTC connection: {e}")

        loop.run_until_complete(setup())
        loop.run_forever()

    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
    asyncio_thread.start()

    frame_index = 0

    try:
        while True:
            if frame_queue.empty():
                time.sleep(0.01)
                continue

            frame = frame_queue.get()

            if writer is None and args.save_output:
                writer = create_video_writer(
                    Path(args.save_output).expanduser(),
                    30.0,
                    (frame.shape[1], frame.shape[0]),
                )

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
            contour_count = draw_connected_mask_contours(
                frame,
                mask,
                alpha=args.alpha,
                line_width=args.line_width,
            )
            midpoints = extract_rowwise_midpoints(mask, scan_heights) if mask is not None else []
            heading_angle, _weighted_target, arrow_start = calculate_heading(
                midpoints,
                frame.shape[:2],
                args.row_weight_power,
            )
            first_midpoint = first_near_midpoint(midpoints)
            row_angle_plan, row_angles = create_row_angle_plan(midpoints, arrow_start)

            draw_row_transition_arrows(frame, midpoints, args.lateral_step_ratio)
            draw_rowwise_midpoints(frame, midpoints)
            draw_start_to_first_midpoint_arrow(frame, arrow_start, first_midpoint)

            heading_text = "Heading: n/a"
            if heading_angle is not None:
                heading_text = f"Heading: {heading_angle:.1f} deg"

            draw_text_block(
                frame,
                [
                    f"Inference: {inference_ms:.1f} ms",
                    f"Total: {total_ms:.1f} ms | Ground contours: {contour_count}",
                    f"{heading_text} | Midpoints: {len(midpoints)} | Weight: {args.row_weight_power:g}",
                    row_angle_plan,
                ],
            )

            if writer is not None:
                writer.write(frame)

            if args.no_display:
                print(
                    f"frame={frame_index} inference_ms={inference_ms:.1f} "
                    f"total_ms={total_ms:.1f} ground_contours={contour_count} "
                    f"heading={heading_angle if heading_angle is not None else 'n/a'} "
                    f"midpoints={len(midpoints)} "
                    f"row_angles={[round(angle, 1) for angle in row_angles]} "
                    f"row_weight_power={args.row_weight_power:g}",
                    flush=True,
                )
            else:
                cv2.imshow(WINDOW_NAME, resize_for_display(frame, display_scale))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            frame_index += 1
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    finally:
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
