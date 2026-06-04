#!/usr/bin/env python3
"""
Control Unitree sport actions with ArUco markers shown to the robot camera.

Default marker mapping:
  1 -> h / Hello
  2 -> x / Stretch
  3 -> y / Sit
  4 -> t / Rize sit
  5 -> f / Scrape
  6 -> g / Front Jump

Examples:
  python Unitree/arucoContro.py
  python Unitree/arucoContro.py --ip 192.168.0.73 --preview
  python Unitree/arucoContro.py --zero-based
"""

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue

import cv2
import numpy as np


logging.basicConfig(level=logging.FATAL)


@dataclass(frozen=True)
class Action:
    key: str
    name: str
    api_id: int | str
    parameter: dict | None = None


DEFAULT_ACTIONS = {
    1: Action("h", "Hello", "Hello"),
    2: Action("x", "Stretch", "Stretch", {"data": False}),
    3: Action("y", "Sit", 1009, {"data": False}),
    4: Action("t", "Rize sit", 1010, {"data": False}),
    5: Action("f", "Scrape", 1029, {"data": False}),
    6: Action("g", "Front Jump", 1031, {"data": False}),
}

ZERO_BASED_ACTIONS = {
    marker_id - 1: action for marker_id, action in DEFAULT_ACTIONS.items()
}


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect ArUco markers from the Unitree stream and run sport actions."
    )
    parser.add_argument(
        "--ip",
        default="unitree.local",
        help="Unitree robot IP/host for LocalSTA mode.",
    )
    parser.add_argument(
        "--dictionary",
        default="DICT_4X4_50",
        help="OpenCV ArUco dictionary name, for example DICT_4X4_50 or DICT_5X5_100.",
    )
    parser.add_argument(
        "--cooldown",
        type=positive_float,
        default=5.0,
        help="Seconds before the same marker can trigger again.",
    )
    parser.add_argument(
        "--confirm-frames",
        type=positive_int,
        default=5,
        help="Require the same marker for this many frames before triggering.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=positive_float,
        default=30.0,
        help="Seconds to wait for the first video frame.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show an OpenCV preview window with detected markers.",
    )
    parser.add_argument(
        "--zero-based",
        action="store_true",
        help="Use marker IDs 0-5 instead of 1-6 for the action mapping.",
    )
    return parser


def queue_error(error_queue: Queue, exc: Exception) -> None:
    try:
        error_queue.put_nowait(exc)
    except Full:
        pass


def aruco_dictionary(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "This OpenCV install has no cv2.aruco module. Install opencv-contrib-python."
        )

    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        valid_names = sorted(name for name in dir(cv2.aruco) if name.startswith("DICT_"))
        raise ValueError(
            f"Unknown ArUco dictionary {dictionary_name!r}. Valid examples: "
            + ", ".join(valid_names[:8])
        )

    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def aruco_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def detect_markers(img: np.ndarray, dictionary, parameters):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters,
        )

    if ids is None:
        return [], corners, ids

    marker_ids = [int(marker_id[0]) for marker_id in ids]
    return marker_ids, corners, ids


def largest_marker_id(marker_ids: list[int], corners) -> int | None:
    if not marker_ids:
        return None

    best_id = None
    best_area = -1.0
    for marker_id, marker_corners in zip(marker_ids, corners):
        points = marker_corners.reshape((4, 2)).astype(np.float32)
        area = cv2.contourArea(points)
        if area > best_area:
            best_area = area
            best_id = marker_id

    return best_id


def print_mapping(actions: dict[int, Action]) -> None:
    print("[INFO] ArUco action mapping:", flush=True)
    for marker_id, action in sorted(actions.items()):
        print(f"  marker {marker_id}: {action.key} / {action.name}", flush=True)


def start_unitree(frame_queue: Queue, error_queue: Queue, ip: str):
    try:
        from aiortc import MediaStreamTrack
        from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
        from unitree_webrtc_connect.webrtc_driver import (
            UnitreeWebRTCConnection,
            WebRTCConnectionMethod,
        )
    except ImportError as exc:
        print(
            "[ERROR] Missing Unitree WebRTC dependencies. Run this on the Jetson "
            "environment where your Unitree scripts work.",
            file=sys.stderr,
            flush=True,
        )
        print(f"[ERROR] Import error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ip)

    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")

            if not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except Empty:
                    pass
            frame_queue.put(img)

    async def ensure_normal_mode():
        response = await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1001},
        )
        data = json.loads(response["data"]["data"])
        if data["name"] != "normal":
            await conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["MOTION_SWITCHER"],
                {"api_id": 1002, "parameter": {"name": "normal"}},
            )
            await asyncio.sleep(5)

    async def send_action(action: Action):
        api_id = SPORT_CMD[action.api_id] if isinstance(action.api_id, str) else action.api_id
        payload = {"api_id": api_id}
        if action.parameter is not None:
            payload["parameter"] = action.parameter
        await conn.datachannel.pub_sub.publish_request_new(RTC_TOPIC["SPORT_MOD"], payload)

    async def setup():
        print(f"[INFO] Connecting to Unitree at {ip}", flush=True)
        await conn.connect()
        await ensure_normal_mode()
        conn.video.switchVideoChannel(True)
        conn.video.add_track_callback(recv_camera_stream)

    def run_loop(loop):
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(setup())
            loop.run_forever()
        except Exception as exc:
            queue_error(error_queue, exc)

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=run_loop, args=(loop,), daemon=True)
    thread.start()
    return loop, thread, send_action


def stop_unitree(loop, thread) -> None:
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5.0)


def wait_for_first_frame(frame_queue: Queue, error_queue: Queue, timeout: float):
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            return frame_queue.get(timeout=0.2), None
        except Empty:
            pass

        try:
            return None, error_queue.get_nowait()
        except Empty:
            pass

    return None, TimeoutError("No video frame received before startup timeout.")


def main() -> int:
    args = build_parser().parse_args()
    actions = ZERO_BASED_ACTIONS if args.zero_based else DEFAULT_ACTIONS
    dictionary = aruco_dictionary(args.dictionary)
    parameters = aruco_parameters()

    print_mapping(actions)
    print("[INFO] Press Ctrl+C to stop.", flush=True)

    frame_queue = Queue(maxsize=1)
    error_queue = Queue(maxsize=1)
    loop, thread, send_action = start_unitree(frame_queue, error_queue, args.ip)

    first_img, connect_error = wait_for_first_frame(
        frame_queue,
        error_queue,
        args.startup_timeout,
    )
    if first_img is None:
        print(f"[ERROR] Could not open Unitree video stream: {connect_error}", flush=True)
        stop_unitree(loop, thread)
        return 1

    last_triggered_at: dict[int, float] = {}
    stable_marker_id = None
    stable_count = 0
    pending_img = first_img
    frame_id = 0

    try:
        while True:
            try:
                stream_error = error_queue.get_nowait()
            except Empty:
                stream_error = None
            if stream_error is not None:
                print(f"[ERROR] Unitree stream failed: {stream_error}", flush=True)
                return 1

            if pending_img is not None:
                img = pending_img
                pending_img = None
            else:
                try:
                    img = frame_queue.get(timeout=0.05)
                except Empty:
                    continue

            marker_ids, corners, ids = detect_markers(img, dictionary, parameters)
            marker_id = largest_marker_id(marker_ids, corners)

            if marker_id != stable_marker_id:
                stable_marker_id = marker_id
                stable_count = 1 if marker_id is not None else 0
            elif marker_id is not None:
                stable_count += 1

            now = time.time()
            if marker_id is not None and frame_id % 15 == 0:
                print(f"[DETECT] visible={marker_ids} selected={marker_id}", flush=True)

            if marker_id in actions and stable_count == args.confirm_frames:
                last_at = last_triggered_at.get(marker_id, 0.0)
                if now - last_at >= args.cooldown:
                    action = actions[marker_id]
                    asyncio.run_coroutine_threadsafe(send_action(action), loop)
                    last_triggered_at[marker_id] = now
                    print(
                        f"[ACTION] marker={marker_id} key={action.key} name={action.name}",
                        flush=True,
                    )

            if args.preview:
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(img, corners, ids)
                cv2.imshow("Unitree ArUco Control", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1

    except KeyboardInterrupt:
        print("[INFO] Stopping.", flush=True)
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        stop_unitree(loop, thread)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
