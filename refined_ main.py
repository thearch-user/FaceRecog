import cv2
import os
import numpy as np
import time
import datetime
import logging
from pathlib import Path

# Setup directories
Path("faces").mkdir(parents=True, exist_ok=True)
Path("trainer").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename="logs/face_recognition.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize face detector and recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load training data if exists
trainer_file = "trainer/trainer.yml"
label_file = "trainer/labels.txt"
labels = {}

if os.path.exists(trainer_file):
    recognizer.read(trainer_file)

if os.path.exists(label_file):
    with open(label_file, "r") as f:
        for line in f:
            name, idx = line.strip().split(",")
            labels[int(idx)] = name

def beep(freq=1000, dur=100):
    """Beep for feedback (Windows only)."""
    try:
        import winsound
        winsound.Beep(freq, dur)
    except ImportError:
        pass  # Ignore if not on Windows

def train_model():
    faces = []
    ids = []
    label_map = {}
    label_id = 0

    for filename in os.listdir("faces"):
        if filename.endswith(".jpg"):
            path = os.path.join("faces", filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            detected_faces = face_cascade.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in detected_faces:
                face_img = img[y:y+h, x:x+w]
                name = filename.split("_")[0]
                if name not in label_map:
                    label_map[name] = label_id
                    label_id += 1
                faces.append(face_img)
                ids.append(label_map[name])

    if faces:
        recognizer.train(faces, np.array(ids))
        recognizer.save(trainer_file)
        with open(label_file, "w") as f:
            for name, idx in label_map.items():
                f.write(f"{name},{idx}\n")
        logging.info("Model training complete.")
        print("[INFO] Training complete.")
    else:
        print("[WARN] No faces found for training.")

def main():
    cap = cv2.VideoCapture(0)
    print("[INFO] Starting webcam. Press 'd' to save face, 'q' to quit.")

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label = "Unknown"
            color = (0, 0, 255)
            confidence_text = ""

            if len(labels) > 0:
                id_, conf = recognizer.predict(roi_gray)
                if conf < 55:  # adjustable confidence threshold
                    label = labels.get(id_, "Unknown")
                    color = (0, 255, 0)
                    confidence_text = f" ({conf:.2f})"
                    beep(1500, 100)
                    logging.info(f"Face recognized: {label} with confidence {conf:.2f}")
                else:
                    logging.warning("Face not confidently recognized.")
            else:
                logging.warning("No labels available for prediction.")

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label}{confidence_text}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        elapsed_time = time.time() - start_time
        fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
        start_time = time.time()

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("[INFO] Exiting program.")
            break
        elif key == ord('d'):
            if len(faces) > 0:
                name = input("Enter name to save face as: ").strip()
                if not name:
                    print("[WARN] Name cannot be empty.")
                    continue

                count = 0
                for (x, y, w, h) in faces:
                    face_img = gray[y:y+h, x:x+w]
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"faces/{name}_{timestamp}_{count}.jpg"
                    cv2.imwrite(filename, face_img)
                    print(f"[INFO] Saved face: {filename}")
                    count += 1

                logging.info(f"Saved {count} face(s) for {name}.")
                train_model()
            else:
                print("[WARN] No face detected to save.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
