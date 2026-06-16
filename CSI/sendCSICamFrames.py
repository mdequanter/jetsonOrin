#!/usr/bin/env python3
import argparse
import asyncio
import json
import time
import uuid

import cv2
import websockets


DEFAULT_SIGNALING_SERVER = "ws://localhost:9000/ws/pathnavigation"
DEFAULT_CAPTURE_WIDTH = 1280
DEFAULT_CAPTURE_HEIGHT = 720
DEFAULT_SEND_WIDTH = 640
DEFAULT_SEND_HEIGHT = 480


def build_gstreamer_pipeline(sensor_id=0, width=1280, height=720, framerate=30):
    return [
        "gst-launch-1.0",
        "nvarguscamerasrc",
        f"sensor-id={sensor_id}",
        "!",
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={framerate}/1",
        "!",
        "nvvidconv",
        "!",
        "nveglglessink",
    ]


def open_camera(sensor_id, width, height, framerate):
    pipeline = build_gstreamer_pipeline(
        sensor_id=sensor_id,
        width=width,
        height=height,
        framerate=framerate,
    )
    camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not camera.isOpened():
        raise RuntimeError("Could not open CSI camera with GStreamer pipeline.")
    return camera


async def receive_headings(websocket, sent_at_by_frame_id, latency_state):
    async for message in websocket:
        if not isinstance(message, str):
            continue

        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            print("Received:", message)
            continue

        if payload.get("type") == "room_joined":
            print(
                f"Joined room '{payload.get('room')}' "
                f"with {payload.get('peers')} peer(s)."
            )
            continue

        heading = payload.get("marker_heading", payload.get("heading"))
        if heading is None:
            print("Received JSON:", payload)
            continue

        frame_id = payload.get("frame_id")
        session_id = payload.get("sessionId")
        latency_ms = None
        if frame_id is not None:
            sent_at = sent_at_by_frame_id.pop(str(frame_id), None)
            if sent_at is not None:
                latency_ms = round((time.monotonic() - sent_at) * 1000.0, 1)
                latency_state["last_latency"] = latency_ms

        latency_text = f", latency={latency_ms}ms" if latency_ms is not None else ""
        print(
            f"Heading: {heading} deg, frame_id={frame_id}, "
            f"sessionId={session_id}{latency_text}"
        )


async def send_frames(args):
    camera = open_camera(
        sensor_id=args.sensor_id,
        width=args.capture_width,
        height=args.capture_height,
        framerate=args.framerate,
    )

    session_id = args.session_id or f"csi-{uuid.uuid4().hex[:12]}"
    frame_id = 0
    frame_interval = 1.0 / args.send_fps
    last_sent_at = 0.0
    latency_state = {"last_latency": None}
    sent_at_by_frame_id = {}

    try:
        async with websockets.connect(
            args.server,
            compression=None,
            max_size=None,
        ) as websocket:
            print(f"Connected to signaling server ({args.server})")
            print(f"Session ID: {session_id}")

            receiver = asyncio.create_task(
                receive_headings(websocket, sent_at_by_frame_id, latency_state)
            )
            try:
                while True:
                    ok, frame = camera.read()
                    if not ok:
                        await asyncio.sleep(0.05)
                        continue

                    now = time.monotonic()
                    wait_time = frame_interval - (now - last_sent_at)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                    last_sent_at = time.monotonic()

                    if (
                        frame.shape[1] != args.send_width
                        or frame.shape[0] != args.send_height
                    ):
                        frame = cv2.resize(frame, (args.send_width, args.send_height))

                    encode_params = [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        int(args.jpeg_quality),
                    ]
                    encoded, jpeg = cv2.imencode(".jpg", frame, encode_params)
                    if not encoded:
                        continue

                    current_frame_id = str(frame_id)
                    frame_id += 1

                    metadata = {
                        "type": "frame_meta",
                        "frame_id": current_frame_id,
                        "sessionId": session_id,
                        "latitude": None,
                        "longitude": None,
                        "lastlatency": latency_state["last_latency"],
                        "model": args.model,
                        "detection_confidence": args.detection_confidence,
                        "returnMasks": args.return_masks,
                        "sendMQTT": args.send_mqtt,
                    }

                    sent_at_by_frame_id[current_frame_id] = time.monotonic()
                    await websocket.send(json.dumps(metadata))
                    await websocket.send(jpeg.tobytes())

                    if len(sent_at_by_frame_id) > 200:
                        cutoff = time.monotonic() - 5.0
                        for key, value in list(sent_at_by_frame_id.items()):
                            if value < cutoff:
                                sent_at_by_frame_id.pop(key, None)

                    await asyncio.sleep(0)
            finally:
                receiver.cancel()
                try:
                    await receiver
                except asyncio.CancelledError:
                    pass
    finally:
        camera.release()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send Jetson CSI camera frames to the signaling server."
    )
    parser.add_argument("--server", default=DEFAULT_SIGNALING_SERVER)
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--capture-width", type=int, default=DEFAULT_CAPTURE_WIDTH)
    parser.add_argument("--capture-height", type=int, default=DEFAULT_CAPTURE_HEIGHT)
    parser.add_argument("--send-width", type=int, default=DEFAULT_SEND_WIDTH)
    parser.add_argument("--send-height", type=int, default=DEFAULT_SEND_HEIGHT)
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--send-fps", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--session-id", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--detection-confidence", type=float, default=0.8)
    parser.add_argument("--return-masks", action="store_true")
    parser.add_argument("--send-mqtt", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(send_frames(args))


if __name__ == "__main__":
    main()
