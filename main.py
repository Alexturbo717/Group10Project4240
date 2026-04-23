## Group 10 CSCE4240 Project
## Abdul Hashmi, Alexander Surma, Sterling Hardy
import cv2
import os
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# CONSTANTS - change if need to tweak
KNOWN_FACES_DIR = "known_faces"   # one subfolder per person
SIMILARITY_THRESHOLD = 0.50       # cosine similarity: higher = stricter (0.0-1.0)
PROCESS_EVERY_N = 2               # run recognition every N frames to stay smooth

# Load InsightFace buffalo_sc model it's a lightweight but accurate model
def load_model():
    app = FaceAnalysis(
        name="buffalo_sc",
        providers=["CPUExecutionProvider"]  # use CPU; change to CUDAExecutionProvider if you have a GPU
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

# This load a person face from the Known_faces
def load_known_faces(known_faces_dir, app):
    known_embeddings = []
    known_names = []

    if not os.path.exists(known_faces_dir):
        print(f"Warning: '{known_faces_dir}' not found. Creating it.")
        print("Add one subfolder per person, e.g.  known_faces/John Doe/1.jpg")
        os.makedirs(known_faces_dir)
        return known_embeddings, known_names

    for person_name in sorted(os.listdir(known_faces_dir)):
        person_dir = os.path.join(known_faces_dir, person_name)
        # skip loose files in root
        if not os.path.isdir(person_dir):
            continue  
        print(f"\n  Person: '{person_name}'")
        loaded = 0

        for filename in sorted(os.listdir(person_dir)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"    Warning: could not read '{img_path}'. Skipping.")
                continue
            # InsightFace works directly on BGR images (OpenCV default)
            faces = app.get(img)
            if not faces:
                print(f"    Warning: no face found in '{filename}'. Skipping.")
                continue
            # Use the largest face in the photo
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

            # Normalize the embedding to unit length for cosine similarity
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)
            known_embeddings.append(embedding)
            known_names.append(person_name)
            loaded += 1
    return known_embeddings, known_names

# MAIN
def main():
    app = load_model()

    print(f"\nLoading known faces from '{KNOWN_FACES_DIR}' ...")
    known_embeddings, known_names = load_known_faces(KNOWN_FACES_DIR, app)
    ready = len(known_embeddings) > 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("\nRunning. Press 'q' to quit.")
    frame_count  = 0
    last_results = []  # [(box, label, matched)]
    show_unknown_prompt = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1

        if frame_count % PROCESS_EVERY_N == 0:
            faces = app.get(frame)
            last_results = []
            show_unknown_prompt = False

            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                if ready and face.embedding is not None:
                    # Normalize live embedding then compute cosine similarity
                    # against all stored embeddings in one vectorized operation
                    live_emb = face.embedding / np.linalg.norm(face.embedding)
                    similarities = np.dot(known_embeddings, live_emb)

                    best_idx  = int(np.argmax(similarities))
                    best_score = float(similarities[best_idx])

                    if best_score >= SIMILARITY_THRESHOLD:
                        name    = known_names[best_idx]
                        matched = True
                    else:
                        name    = "Unknown"
                        matched = False
                        show_unknown_prompt = True

                    label = f"{name}  ({best_score:.2f})"
                else:
                    name, label, matched = "No DB", "No DB", False

                last_results.append(((x1, y1, x2, y2), label, matched))

        # Draw boxes and labels from last_results
        for (x1, y1, x2, y2), label, matched in last_results:
            color = (0, 200, 0) if matched else (0, 0, 220)  # green / red

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - th - baseline - 8),
                (x1 + tw + 6, y1),
                color, cv2.FILLED
            )
            cv2.putText(
                frame, label,
                (x1 + 3, y1 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )
        if show_unknown_prompt:
            overlay = frame.copy()

            # dark box in center
            cv2.rectangle(overlay, (120, 180), (520, 280), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

            cv2.rectangle(frame, (120, 180), (520, 280), (255, 255, 255), 2)
            cv2.putText(frame, "Unknown person detected", (155, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, "Would you like to add this user?", (135, 245),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, "Press Y for Yes", (225, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow('Face Recognition — press Q to quit', frame)
            key = cv2.waitKey(1) & 0xFF
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if show_unknown_prompt and key == ord('y'):
                print("Yes pressed for unknown user.")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()