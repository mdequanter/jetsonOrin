from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from queue import Empty, Queue

import cv2
from aiortc import MediaStreamTrack
from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod,
)
from ultralytics import YOLO

from HeadingOnVideo import (
    DEFAULT_BOTTOM_CONNECT_RATIO,
    DEFAULT_BOTTOM_SEED_X_RATIO,
    DEFAULT_DISPLAY_SCALE,
    DEFAULT_LATERAL_STEP_RATIO,
    DEFAULT_MODEL_PATH,
    DEFAULT_ROW_WEIGHT_POWER,
    DEFAULT_SCAN_HEIGHTS,
    calculate_heading,
    create_row_angle_plan,
    draw_connected_mask_contours,
    draw_row_transition_arrows,
    draw_rowwise_midpoints,
    draw_start_to_first_midpoint_arrow,
    draw_text_block,
    extract_rowwise_midpoints,
    first_near_midpoint,
    parse_scan_heights,
    resize_for_display,
    segmentation_mask_from_result,
)


WINDOW_NAME = "Gang Kaai heading stream"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show HeadingOnVideo segmentation heading on the Unitree camera stream."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help=f"YOLO segmentation model path. Default: {DEFAULT_MODEL_PATH}",
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
    parser.add_argument("--line-width", type=int, default=2, help="Contour line width.")
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
        "--display-scale",
        type=float,
        default=DEFAULT_DISPLAY_SCALE,
        help="Scale factor for the preview window. This does not change inference size.",
    )
    parser.add_argument(
        "--ip",
        default="unitree.local",
        help="Unitree LocalSTA IP or hostname. Default: unitree.local",
    )
    return parser.parse_args()


def annotate_frame(frame, model: YOLO, args: argparse.Namespace, scan_heights: list[float]):
    predict_kwargs = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_kwargs["device"] = args.device

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
    row_angle_plan, _row_angles = create_row_angle_plan(midpoints, arrow_start)

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

    return frame


def main() -> int:
    args = parse_args()
    scan_heights = parse_scan_heights(args.scan_heights)
    display_scale = max(args.display_scale, 0.1)
    model = YOLO(str(args.model), verbose=False)
    frame_queue: Queue = Queue(maxsize=1)

    logging.basicConfig(level=logging.FATAL)
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=args.ip)

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
            frame_queue.put(img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)

        async def setup():
            try:
                await conn.connect()
                conn.video.switchVideoChannel(True)
                conn.video.add_track_callback(recv_camera_stream)
            except Exception as exc:
                logging.error(f"Error in WebRTC connection: {exc}")

        loop.run_until_complete(setup())
        loop.run_forever()

    loop = asyncio.new_event_loop()
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
    asyncio_thread.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            try:
                frame = frame_queue.get(timeout=0.1)
            except Empty:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue

            annotated = annotate_frame(frame, model, args, scan_heights)
            cv2.imshow(WINDOW_NAME, resize_for_display(annotated, display_scale))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
