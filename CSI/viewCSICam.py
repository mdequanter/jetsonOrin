import cv2


def build_pipeline(sensor_id=0, width=1280, height=720, framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw,format=BGRx ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true sync=false"
    )


def show_camera():
    window_title = "CSI Camera"

    pipeline = build_pipeline(
        sensor_id=0,
        width=1280,
        height=720,
        framerate=30,
        flip_method=0
    )

    print("Using pipeline:")
    print(pipeline)

    video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not video_capture.isOpened():
        print("Error: Unable to open camera")
        return

    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = video_capture.read()

            if not ret or frame is None:
                print("Error: Could not read frame")
                break

            cv2.imshow(window_title, frame)

            key = cv2.waitKey(10) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    show_camera()