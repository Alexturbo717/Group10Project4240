## Group 10 CSCE4240 Project 
## Abdul Hashmi, Alexander Surma, Sterling Hardy
import cv2
import os
import numpy as np

def load_known_faces(known_faces_dir):
    # Load known faces from a directory.
    # Each image file should be named after the person (e.g., 'John_Doe.jpg').
    # Returns a list of (name, face_encoding) tuples.

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    known_faces = []   # list of (label_id, name)
    face_images = []   # grayscale face crops
    labels = []        # integer label per face

    if not os.path.exists(known_faces_dir):
        print(f"Warning: Known faces directory '{known_faces_dir}' not found.")
        print("Creating the directory — add images named 'Firstname_Lastname.jpg' to it.")
        os.makedirs(known_faces_dir)
        return recognizer, known_faces

    label_id = 0
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0].replace('_', ' ')
            img_path = os.path.join(known_faces_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image '{img_path}'. Skipping.")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(detected) == 0:
                print(f"Warning: No face detected in '{filename}'. Skipping.")
                continue

            # Use the largest detected face
            (x, y, w, h) = max(detected, key=lambda r: r[2] * r[3])
            face_crop = gray[y:y+h, x:x+w]
            face_images.append(face_crop)
            labels.append(label_id)
            known_faces.append((label_id, name))
            print(f"  Loaded: {name} (label {label_id})")
            label_id += 1

    if face_images:
        recognizer.train(face_images, np.array(labels))
        print(f"Recognizer trained on {len(face_images)} face(s).")
    else:
        print("No valid face images found in the known faces directory.")

    return recognizer, known_faces


def main():
    KNOWN_FACES_DIR = "known_faces"   # folder with reference images
    CONFIDENCE_THRESHOLD = 80         # lower = stricter match

    print(f"Loading known faces from '{KNOWN_FACES_DIR}' ...")
    recognizer, known_faces = load_known_faces(KNOWN_FACES_DIR)

    # Build a quick id to name lookup
    id_to_name = {label_id: name for label_id, name in known_faces}
    recognizer_ready = len(known_faces) > 0

    # Load Haar Cascade for detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            face_crop = gray[y:y+h, x:x+w]

            if recognizer_ready:
                label_id, confidence = recognizer.predict(face_crop)
                if confidence < CONFIDENCE_THRESHOLD:
                    name = id_to_name.get(label_id, "Unknown")
                    box_color = (0, 255, 0)        # green for recognized
                else:
                    name = "Unknown"
                    box_color = (0, 0, 255)        # red for unknown

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Draw name label with a filled background for readability
            label_text = f"{name}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                frame,
                (x, y - text_h - baseline - 6),
                (x + text_w + 4, y),
                box_color,
                cv2.FILLED
            )
            cv2.putText(
                frame, label_text,
                (x + 2, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()