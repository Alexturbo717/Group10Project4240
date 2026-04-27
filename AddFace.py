import cv2
import os
import re

def clean_name(name):
    # Removes weird characters so folder names are safe
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    return name

def save_unknown_face(frame, face_box, name, known_faces_dir="known_faces", pose_name="front"):
    name = clean_name(name)

    if not name:
        print("Invalid name. Face was not saved.")
        return False

    person_dir = os.path.join(known_faces_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    x1, y1, x2, y2 = face_box

    h, w = frame.shape[:2]
    padding = 100

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    face_img = frame[y1:y2, x1:x2]

    if face_img.size == 0:
        print("Could not crop face.")
        return False

    save_path = os.path.join(person_dir, f"{pose_name}.jpg")
    cv2.imwrite(save_path, face_img)

    print(f"Saved {pose_name} face image to: {save_path}")
    return True