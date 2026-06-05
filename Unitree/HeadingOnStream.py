import cv2
import numpy as np

import argparse
import asyncio
import logging
import os
import threading
import time
from queue import Queue
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack
from ultralytics import YOLO

MODEL_PATH = r"models/unrealsim.pt"
DETECTION_CONFIDENCE = 0.3
SCAN_HEIGHTS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

model = YOLO(MODEL_PATH, verbose=False)

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

ROBOT_IP = "unitree.local"
ARUCO_DICTIONARY = "DICT_4X4_50"
RECOVERY_DELAY_SECONDS = 3.0
FOLLOW_MARKER_ID = 14
FOLLOW_COMMAND_INTERVAL_SECONDS = 0.2
FOLLOW_LOST_STOP_SECONDS = 1.0
FOLLOW_FORWARD_GAIN = 0.45
FOLLOW_TURN_GAIN = 0.8
FOLLOW_MAX_FORWARD_SPEED = 0.35
FOLLOW_MAX_TURN_SPEED = 0.8
FOLLOW_SIZE_DEADBAND = 0.12
FOLLOW_CENTER_DEADBAND = 0.12
LIGHT_TOGGLE_MARKER_ID = 26
LIGHT_BRIGHTNESS = 1

def parse_args():
    parser = argparse.ArgumentParser(
        description="Control the Unitree with ArUco markers from the WebRTC video stream."
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show the OpenCV video window. Without this, the script runs terminal-only.",
    )
    return parser.parse_args()

ARUCO_ACTIONS = {
    22: ("h / Hello", {"api_id": SPORT_CMD["Hello"]}, True),
    23: ("x / Stretch", {"api_id": SPORT_CMD["Stretch"], "parameter": {"data": False}}, True),
    24: ("y / Sit", {"api_id": 1009, "parameter": {"data": False}}, False),
    25: ("t / Rize sit", {"api_id": 1010, "parameter": {"data": False}}, True),
    26: ("toggle light", None, False),
    27: ("g / Front Jump", {"api_id": 1031, "parameter": {"data": False}}, True),
}

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))

def create_aruco_detector(dictionary_name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "This OpenCV install has no cv2.aruco module. Install opencv-contrib-python."
        )

    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise RuntimeError(f"Unknown ArUco dictionary: {dictionary_name}")

    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    else:
        dictionary = cv2.aruco.Dictionary_get(dictionary_id)

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return lambda frame: detector.detectMarkers(frame)

    return lambda frame: cv2.aruco.detectMarkers(
        frame,
        dictionary,
        parameters=parameters,
    )

def detect_and_draw_aruco(img, detect_markers):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detect_markers(gray)

    if ids is None:
        return [], {}

    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    marker_ids = [int(marker_id[0]) for marker_id in ids]
    marker_info = {}

    for marker_id, marker_corners in zip(marker_ids, corners):
        points = marker_corners.reshape((4, 2)).astype(int)
        center_x = int(points[:, 0].mean())
        center_y = int(points[:, 1].mean())
        side_lengths = [
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[1] - points[2]),
            np.linalg.norm(points[2] - points[3]),
            np.linalg.norm(points[3] - points[0]),
        ]
        marker_info[marker_id] = {
            "center_x": center_x,
            "center_y": center_y,
            "side_px": float(np.mean(side_lengths)),
        }
        cv2.putText(
            img,
            f"ID {marker_id}",
            (center_x - 30, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if marker_id in ARUCO_ACTIONS:
            action_name, _payload, _needs_recovery = ARUCO_ACTIONS[marker_id]
            cv2.putText(
                img,
                action_name,
                (center_x - 55, center_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if marker_id == FOLLOW_MARKER_ID:
            cv2.putText(
                img,
                "follow",
                (center_x - 35, center_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return marker_ids, marker_info

async def send_move(conn, x=0, y=0, z=0):
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": x, "y": y, "z": z},
        },
    )

async def send_light(conn, brightness):
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["VUI"],
        {
            "api_id": 1005,
            "parameter": {"brightness": brightness},
        },
    )

async def send_action(conn, payload, needs_recovery):
    action_error = None

    try:
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            payload,
        )
    except Exception as exc:
        action_error = exc

    if needs_recovery:
        await asyncio.sleep(RECOVERY_DELAY_SECONDS)

        recovery_payload = {
            "api_id": SPORT_CMD["RecoveryStand"],
            "parameter": {"data": False},
        }
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["SPORT_MOD"],
            recovery_payload,
        )

    if action_error is not None:
        raise action_error

def report_action_result(future, marker_id, action_name, needs_recovery, action_state):
    try:
        future.result()
        recovery_text = " + RecoveryStand" if needs_recovery else ""
        print(f"Action completed: marker {marker_id} ({action_name}){recovery_text}", flush=True)
    except Exception as exc:
        print(
            f"Action failed for marker {marker_id} ({action_name}): {exc}",
            flush=True,
        )
    finally:
        action_state["in_progress"] = False

def report_move_result(future):
    try:
        future.result()
    except Exception as exc:
        print(f"Follow move failed: {exc}", flush=True)

def report_light_result(future, brightness, light_state, action_state):
    try:
        future.result()
        light_state["on"] = brightness > 0
        status = "on" if light_state["on"] else "off"
        print(f"Light toggled {status}", flush=True)
    except Exception as exc:
        print(f"Light toggle failed: {exc}", flush=True)
    finally:
        action_state["in_progress"] = False

def stop_follow_marker(conn, loop, follow_state):
    if follow_state["active"]:
        asyncio.run_coroutine_threadsafe(send_move(conn), loop).add_done_callback(report_move_result)
        follow_state["active"] = False
        follow_state["target_side_px"] = None
        print("Follow marker paused: stop", flush=True)

def update_follow_marker(conn, loop, img, marker_info, follow_state):
    marker = marker_info.get(FOLLOW_MARKER_ID)
    now = time.time()

    if marker is None:
        if follow_state["active"] and now - follow_state["last_seen_at"] >= FOLLOW_LOST_STOP_SECONDS:
            asyncio.run_coroutine_threadsafe(send_move(conn), loop).add_done_callback(report_move_result)
            follow_state["active"] = False
            follow_state["target_side_px"] = None
            print("Follow marker lost: stop", flush=True)
        return

    follow_state["active"] = True
    follow_state["last_seen_at"] = now

    if follow_state["target_side_px"] is None:
        follow_state["target_side_px"] = marker["side_px"]
        print(
            f"Follow marker {FOLLOW_MARKER_ID}: target size set to "
            f"{follow_state['target_side_px']:.1f}px",
            flush=True,
        )

    if now - follow_state["last_command_at"] < FOLLOW_COMMAND_INTERVAL_SECONDS:
        return

    target_side = follow_state["target_side_px"]
    current_side = marker["side_px"]
    size_error = (target_side - current_side) / max(target_side, 1.0)
    center_error = (marker["center_x"] - (img.shape[1] / 2)) / (img.shape[1] / 2)

    x_speed = 0 if abs(size_error) < FOLLOW_SIZE_DEADBAND else size_error * FOLLOW_FORWARD_GAIN
    z_speed = 0 if abs(center_error) < FOLLOW_CENTER_DEADBAND else -center_error * FOLLOW_TURN_GAIN

    x_speed = clamp(x_speed, -FOLLOW_MAX_FORWARD_SPEED, FOLLOW_MAX_FORWARD_SPEED)
    z_speed = clamp(z_speed, -FOLLOW_MAX_TURN_SPEED, FOLLOW_MAX_TURN_SPEED)

    future = asyncio.run_coroutine_threadsafe(send_move(conn, x=x_speed, z=z_speed), loop)
    future.add_done_callback(report_move_result)
    follow_state["last_command_at"] = now

def main():
    args = parse_args()
    frame_queue = Queue()
    detect_markers = create_aruco_detector(ARUCO_DICTIONARY)
    last_printed_ids = None
    executed_marker_ids = set()
    action_state = {"in_progress": False}
    light_state = {"on": False}
    previous_marker_ids = set()
    follow_state = {
        "active": False,
        "target_side_px": None,
        "last_seen_at": 0,
        "last_command_at": 0,
    }

    # Choose a connection method (uncomment the correct one)
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=ROBOT_IP)
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, serialNumber="B42D2000XXXXXXXX")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.Remote, serialNumber="B42D2000XXXXXXXX", username="email@gmail.com", password="pass")
    # conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    # Async function to receive video frames and put them in the queue
    async def recv_camera_stream(track: MediaStreamTrack):
        while True:
            frame = await track.recv()
            # Convert the frame to a NumPy array
            img = frame.to_ndarray(format="bgr24")
            frame_queue.put(img)

    def run_asyncio_loop(loop):
        asyncio.set_event_loop(loop)
        async def setup():
            try:
                # Connect to the device
                await conn.connect()

                # Switch video channel on and start receiving video frames
                conn.video.switchVideoChannel(True)

                # Add callback to handle received video frames
                conn.video.add_track_callback(recv_camera_stream)
            except Exception as e:
                logging.error(f"Error in WebRTC connection: {e}")

        # Run the setup coroutine and then start the event loop
        loop.run_until_complete(setup())
        loop.run_forever()

    # Create a new event loop for the asyncio code
    loop = asyncio.new_event_loop()

    # Start the asyncio event loop in a separate thread
    asyncio_thread = threading.Thread(target=run_asyncio_loop, args=(loop,))
    asyncio_thread.start()

    try:
        while True:
            if not frame_queue.empty():
                img = frame_queue.get()

                results = model(img, conf=DETECTION_CONFIDENCE, verbose=False)


                marker_ids, marker_info = detect_and_draw_aruco(img, detect_markers)
                if marker_ids != last_printed_ids:
                    print(f"ArUco markers: {marker_ids}", flush=True)
                    last_printed_ids = marker_ids

                current_marker_ids = set(marker_ids)
                if (
                    not action_state["in_progress"]
                    and LIGHT_TOGGLE_MARKER_ID in current_marker_ids
                    and LIGHT_TOGGLE_MARKER_ID not in previous_marker_ids
                ):
                    action_state["in_progress"] = True
                    brightness = 0 if light_state["on"] else LIGHT_BRIGHTNESS
                    future = asyncio.run_coroutine_threadsafe(
                        send_light(conn, brightness),
                        loop,
                    )
                    future.add_done_callback(
                        lambda done, seen_brightness=brightness: report_light_result(
                            done,
                            seen_brightness,
                            light_state,
                            action_state,
                        )
                    )
                    stop_follow_marker(conn, loop, follow_state)
                    print("Action triggered: marker 26 -> toggle light", flush=True)

                if not action_state["in_progress"]:
                    marker_id = next(
                        (
                            current_id for current_id in sorted(marker_ids)
                            if current_id in ARUCO_ACTIONS
                            and current_id != LIGHT_TOGGLE_MARKER_ID
                            and current_id not in executed_marker_ids
                        ),
                        None,
                    )
                    if marker_id is not None:
                        executed_marker_ids.add(marker_id)
                        action_state["in_progress"] = True
                        action_name, payload, needs_recovery = ARUCO_ACTIONS[marker_id]
                        future = asyncio.run_coroutine_threadsafe(
                            send_action(conn, payload.copy(), needs_recovery),
                            loop,
                        )
                        future.add_done_callback(
                            lambda done, seen_id=marker_id, seen_action=action_name, seen_recovery=needs_recovery: report_action_result(
                                done,
                                seen_id,
                                seen_action,
                                seen_recovery,
                                action_state,
                            )
                        )
                        print(
                            f"Action triggered once: marker {marker_id} -> {action_name}",
                            flush=True,
                        )
                        stop_follow_marker(conn, loop, follow_state)

                if not action_state["in_progress"]:
                    update_follow_marker(conn, loop, img, marker_info, follow_state)
                previous_marker_ids = current_marker_ids
                if args.preview:
                    cv2.imshow('Video', img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                # Sleep briefly to prevent high CPU usage
                time.sleep(0.01)
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        # Stop the asyncio event loop
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()

if __name__ == "__main__":
    main()
