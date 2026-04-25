## Group 10 CSCE4240 Project
## Abdul Hashmi, Alexander Surma, Sterling Hardy
import cv2
import os
import numpy as np
from AddFace import save_unknown_face
import insightface
import time
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
        print("Add one subfolder per person, e.g. known_faces/John Doe/1.jpg")
        os.makedirs(known_faces_dir)
        return known_embeddings, known_names

    for person_name in sorted(os.listdir(known_faces_dir)):
        person_dir = os.path.join(known_faces_dir, person_name)

        # skip loose files in root
        if not os.path.isdir(person_dir):
            continue

        print(f"\n  Person: '{person_name}'")

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
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )

            # Normalize the embedding to unit length for cosine similarity
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)

            known_embeddings.append(embedding)
            known_names.append(person_name)

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
    frame_count = 0
    last_results = []  # [(box, label, matched)]

    unknown_start_time = None
    show_unknown_prompt = False
    face_added_message = False
    face_added_message_time = 0
    last_unknown_box = None

    entering_name = False
    typed_name = ""
    clean_frame_for_save = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        clean_frame = frame.copy()

        frame_count += 1

        if frame_count % PROCESS_EVERY_N == 0:
            faces = app.get(frame)
            last_results = []
            unknown_detected = False

            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                if ready and face.embedding is not None:
                    # Normalize live embedding then compute cosine similarity
                    live_emb = face.embedding / np.linalg.norm(face.embedding)
                    similarities = np.dot(known_embeddings, live_emb)

                    best_idx = int(np.argmax(similarities))
                    best_score = float(similarities[best_idx])

                    if best_score >= SIMILARITY_THRESHOLD:
                        name = known_names[best_idx]
                        matched = True
                    else:
                        name = "Unknown"
                        matched = False
                        unknown_detected = True
                        last_unknown_box = (x1, y1, x2, y2)

                    label = f"{name} ({best_score:.2f})"

                else:
                    # If no known faces DB yet, still treat detected faces as unknown
                    if face.embedding is not None:
                        name = "Unknown"
                        label = "Unknown"
                        matched = False
                        unknown_detected = True
                        last_unknown_box = (x1, y1, x2, y2)
                    else:
                        name = "No Face"
                        label = "No Face"
                        matched = False

                last_results.append(((x1, y1, x2, y2), label, matched))

            if unknown_detected and not entering_name:
                if unknown_start_time is None:
                    unknown_start_time = time.time()
                elif time.time() - unknown_start_time >= 10:
                    show_unknown_prompt = True
            else:
                unknown_start_time = None
                show_unknown_prompt = False

        # Draw boxes and labels from last_results
        for (x1, y1, x2, y2), label, matched in last_results:
            color = (0, 200, 0) if matched else (0, 0, 220)  # green / red

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            )
            top_y = max(0, y1 - th - baseline - 8)

            cv2.rectangle(
                frame,
                (x1, top_y),
                (x1 + tw + 6, y1),
                color,
                cv2.FILLED
            )

            cv2.putText(
                frame,
                label,
                (x1 + 3, y1 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        # Bottom prompt
        if show_unknown_prompt and not entering_name:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 80), (w, h), (40, 40, 40), -1)
            cv2.putText(
                frame,
                "Unknown face detected. Press Y to add user.",
                (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

        # Face added message
        if face_added_message:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 80), (w, h), (0, 120, 0), -1)
            cv2.putText(
                frame,
                "Face added.",
                (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            if time.time() - face_added_message_time >= 2:
                face_added_message = False

        if entering_name:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 100), (w, h), (40, 40, 40), -1)

            cv2.putText(
                frame,
                "Enter name: " + typed_name,
                (20, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Press ENTER to save or ESC to cancel",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1
            )

        cv2.imshow('Face Recognition — press Q to quit', frame)
        key = cv2.waitKey(1) & 0xFF

        if entering_name:
            if key == 13:  # ENTER
                if typed_name.strip() and last_unknown_box is not None and clean_frame_for_save is not None:
                    saved = save_unknown_face(
                        clean_frame_for_save,
                        last_unknown_box,
                        typed_name,
                        KNOWN_FACES_DIR
                    )

                    if saved:
                        known_embeddings, known_names = load_known_faces(KNOWN_FACES_DIR, app)
                        ready = len(known_embeddings) > 0

                        face_added_message = True
                        face_added_message_time = time.time()
                        print("Face added and known faces reloaded.")

                    entering_name = False
                    typed_name = ""
                    clean_frame_for_save = None

                else:
                    print("Name was empty or no face available.")

            elif key == 27:  # ESC
                entering_name = False
                typed_name = ""
                clean_frame_for_save = None
                print("Canceled adding face.")

            elif key == 8:  # BACKSPACE
                typed_name = typed_name[:-1]

            elif 32 <= key <= 126:
                typed_name += chr(key)

        if key == ord('q'):
            break

        if show_unknown_prompt and key in [ord('y'), ord('Y')]:
            show_unknown_prompt = False
            unknown_start_time = None
            entering_name = True
            typed_name = ""
            clean_frame_for_save = clean_frame.copy()

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()