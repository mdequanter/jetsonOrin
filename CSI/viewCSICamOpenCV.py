#!/usr/bin/env python3
import argparse
import sys

import cv2


def build_pipeline(sensor_id=0, width=1280, height=720, framerate=30):
    return (
        "nvarguscamerasrc sensor-id={sensor_id} ! "
        "video/x-raw(memory:NVMM),width={width},height={height},framerate={framerate}/1 ! "
        "nvvidconv ! "
        "video/x-raw,format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    ).format(sensor_id=sensor_id, width=width, height=height, framerate=framerate)


def main():
    parser = argparse.ArgumentParser(
        description="Show the Jetson CSI camera preview through OpenCV."
    )
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--framerate", type=int, default=30)
    parser.add_argument("--window-name", default="CSI Camera")
    args = parser.parse_args()

    pipeline = build_pipeline(
        sensor_id=args.sensor_id,
        width=args.width,
        height=args.height,
        framerate=args.framerate,
    )

    camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not camera.isOpened():
        print("Could not open CSI camera with OpenCV/GStreamer.", file=sys.stderr)
        print(f"Pipeline: {pipeline}", file=sys.stderr)
        return 1

    cv2.namedWindow(args.window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read frame from CSI camera.", file=sys.stderr)
                return 1

            cv2.imshow(args.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
