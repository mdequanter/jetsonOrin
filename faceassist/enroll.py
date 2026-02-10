import os
import time
import argparse
import numpy as np
import cv2

from insightface.app import FaceAnalysis


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Naam van de persoon (bv. Tom)")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--samples", type=int, default=15, help="Aantal embeddings om op te slaan")
    parser.add_argument("--outdir", type=str, default="known", help="Output map")
    parser.add_argument("--min_face", type=int, default=120, help="Minimum face bbox breedte in pixels")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(320, 320))

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    embeddings = []
    last_capture = 0.0

    print(f"[INFO] Enrolling '{args.name}'.")
    print("[INFO] Kijk in de camera en verander licht/hoek (links/rechts/omhoog/omlaag).")
    print("[INFO] Druk op 'q' om te stoppen.")

    while True:
        ok, frame = cap.read()
        #frame = cv2.resize(frame, (640, 480))

        if not ok:
            print("[ERROR] Kan geen frame lezen van camera.")
            break

        # detect
        faces = app.get(frame)
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        msg = f"Samples: {len(embeddings)}/{args.samples}"
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 200, 40), 2)

        if len(faces) > 0:
            face = faces[0]
            x1, y1, x2, y2 = face.bbox.astype(int)
            w = x2 - x1
            if w >= args.min_face:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                now = time.time()
                # capture max ~2 per second
                if now - last_capture > 0.5 and len(embeddings) < args.samples:
                    emb = face.embedding.astype(np.float32)
                    embeddings.append(emb)
                    last_capture = now
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, "Kom dichterbij", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "Geen gezicht", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Enroll", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if len(embeddings) >= args.samples:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < 5:
        print("[WARN] Te weinig samples. Probeer opnieuw (minstens 10-15 aangeraden).")
        return

    out_path = os.path.join(args.outdir, f"{args.name}.npz")
    np.savez_compressed(out_path, embeddings=np.stack(embeddings, axis=0))
    print(f"[OK] Opgeslagen: {out_path}  (n={len(embeddings)})")


if __name__ == "__main__":
    main()
