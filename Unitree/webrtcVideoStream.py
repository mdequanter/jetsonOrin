import cv2
import numpy as np

# Create an OpenCV window and display a blank image
height, width = 720, 1280  # Adjust the size as needed
img = np.zeros((height, width, 3), dtype=np.uint8)
cv2.imshow('Video', img)
cv2.waitKey(1)  # Ensure the window is created

import asyncio
import logging
import os
import threading
import time
from queue import Queue
from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection, WebRTCConnectionMethod
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD
from aiortc import MediaStreamTrack

# Enable logging for debugging
logging.basicConfig(level=logging.FATAL)

ROBOT_IP = "192.168.0.73"
ARUCO_DICTIONARY = "DICT_4X4_50"
ACTION_COOLDOWN_SECONDS = 5.0

ARUCO_ACTIONS = {
    22: ("h / Hello", {"api_id": SPORT_CMD["Hello"]}),
    23: ("x / Stretch", {"api_id": SPORT_CMD["Stretch"], "parameter": {"data": False}}),
    24: ("y / Sit", {"api_id": 1009, "parameter": {"data": False}}),
    25: ("t / Rize sit", {"api_id": 1010, "parameter": {"data": False}}),
    26: ("f / Scrape", {"api_id": 1029, "parameter": {"data": False}}),
    27: ("g / Front Jump", {"api_id": 1031, "parameter": {"data": False}}),
}

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
        return []

    cv2.aruco.drawDetectedMarkers(img, corners, ids)
    marker_ids = [int(marker_id[0]) for marker_id in ids]

    for marker_id, marker_corners in zip(marker_ids, corners):
        points = marker_corners.reshape((4, 2)).astype(int)
        center_x = int(points[:, 0].mean())
        center_y = int(points[:, 1].mean())
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
            action_name, _payload = ARUCO_ACTIONS[marker_id]
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

    return marker_ids

async def send_action(conn, payload):
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        payload,
    )

def report_action_result(future, marker_id, action_name):
    try:
        future.result()
    except Exception as exc:
        print(
            f"Action failed for marker {marker_id} ({action_name}): {exc}",
            flush=True,
        )

def main():
    frame_queue = Queue()
    detect_markers = create_aruco_detector(ARUCO_DICTIONARY)
    last_printed_ids = None
    last_triggered_at = {}

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
                marker_ids = detect_and_draw_aruco(img, detect_markers)
                if marker_ids != last_printed_ids:
                    print(f"ArUco markers: {marker_ids}", flush=True)
                    last_printed_ids = marker_ids

                now = time.time()
                for marker_id in marker_ids:
                    if marker_id not in ARUCO_ACTIONS:
                        continue

                    last_at = last_triggered_at.get(marker_id, 0)
                    if now - last_at < ACTION_COOLDOWN_SECONDS:
                        continue

                    action_name, payload = ARUCO_ACTIONS[marker_id]
                    future = asyncio.run_coroutine_threadsafe(
                        send_action(conn, payload.copy()),
                        loop,
                    )
                    future.add_done_callback(
                        lambda done, seen_id=marker_id, seen_action=action_name: report_action_result(
                            done,
                            seen_id,
                            seen_action,
                        )
                    )
                    last_triggered_at[marker_id] = now
                    print(
                        f"Action triggered: marker {marker_id} -> {action_name}",
                        flush=True,
                    )
                # Display the frame
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Sleep briefly to prevent high CPU usage
                time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()
        # Stop the asyncio event loop
        loop.call_soon_threadsafe(loop.stop)
        asyncio_thread.join()

if __name__ == "__main__":
    main()
