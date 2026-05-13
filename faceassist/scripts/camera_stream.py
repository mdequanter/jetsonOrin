import cv2


def open_preview_camera():
    cap = cv2.VideoCapture(0)
    return cap

def generate_camera_frames():
    cap = open_preview_camera()

    if not cap.isOpened():
        print("[FOUT] Preview camera kon niet geopend worden.", flush=True)
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (640, 480))

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            jpg = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )

    finally:
        cap.release()