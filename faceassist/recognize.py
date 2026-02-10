import os
import time
import argparse
import numpy as np
import cv2

from insightface.app import FaceAnalysis



try:
    import pyttsx3
    TTS_OK = True
except Exception:
    TTS_OK = False


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def load_known(known_dir: str):
    known = {}  # name -> (N, D) embeddings
    if not os.path.isdir(known_dir):
        return known

    for fn in os.listdir(known_dir):
        if fn.lower().endswith(".npz"):
            name = os.path.splitext(fn)[0]
            path = os.path.join(known_dir, fn)
            data = np.load(path)
            emb = data["embeddings"].astype(np.float32)
            known[name] = emb
    return known


def best_match(emb: np.ndarray, known: dict):
    # returns: (best_name, best_score, second_score)
    scores = []
    for name, embs in known.items():
        # take best similarity across that person's stored embeddings
        s = max(cosine_similarity(emb, e) for e in embs)
        scores.append((name, s))

    if not scores:
        return None, 0.0, 0.0

    scores.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else -1.0
    return best_name, best_score, second_score


def speak(engine, text: str):
    print(f"[SAY] {text}")
    if engine is None:
        return
    engine.say(text)
    engine.runAndWait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--known", type=str, default="known", help="Map met .npz embeddings")
    parser.add_argument("--min_face", type=int, default=120, help="Minimum face bbox breedte in pixels")
    parser.add_argument("--threshold", type=float, default=0.55, help="Similarity threshold (0..1). Start: 0.55")
    parser.add_argument("--margin", type=float, default=0.05, help="Top-2 margin (best - second) minimale marge")
    parser.add_argument("--cooldown", type=float, default=6.0, help="Seconds tussen dezelfde melding")
    args = parser.parse_args()

    known = load_known(args.known)
    if not known:
        print(f"[ERROR] Geen bekende personen gevonden in '{args.known}'. Run eerst enroll.py.")
        return

    print("[INFO] Bekenden:", ", ".join(sorted(known.keys())))

    engine = None
    if TTS_OK:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)

    app = FaceAnalysis(
        name="buffalo_s",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(320, 320))

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 25)

    last_spoken = {}  # name -> timestamp

    print("[INFO] Druk op 'q' om te stoppen.")
    while True:
        ok, frame = cap.read()
        #frame = cv2.resize(frame, (640, 480))
        if not ok:
            print("[ERROR] Kan geen frame lezen van camera.")
            break

        faces = app.get(frame)
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        label = "Geen gezicht"
        color = (0, 0, 255)

        if len(faces) > 0:
            face = faces[0]  # 1 persoon tegelijk: pak grootste gezicht
            x1, y1, x2, y2 = face.bbox.astype(int)
            w = x2 - x1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)

            if w >= args.min_face:
                emb = face.embedding.astype(np.float32)
                best_name, best_score, second_score = best_match(emb, known)

                if best_name is not None:
                    # Decide known vs unknown
                    confident = (best_score >= args.threshold) and ((best_score - second_score) >= args.margin)

                    if confident:
                        label = f"{best_name} ({best_score:.2f})"
                        color = (0, 255, 0)

                        now = time.time()
                        last = last_spoken.get(best_name, 0.0)
                        if now - last >= args.cooldown:
                            speak(engine, f"Ik denk dat het {best_name} is")
                            last_spoken[best_name] = now
                    else:
                        label = f"Onbekend ({best_score:.2f})"
                        color = (0, 165, 255)
                else:
                    label = "Onbekend"
                    color = (0, 165, 255)
            else:
                label = "Kom dichterbij"
                color = (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow("Recognize", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
