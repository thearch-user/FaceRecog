import cv2
import os
import numpy as np

# Create directories
os.makedirs("faces", exist_ok=True)
os.makedirs("trainer", exist_ok=True)

# Initialize face detector and recognizer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load training data if exists
if os.path.exists("trainer/trainer.yml"):
    recognizer.read("trainer/trainer.yml")

# Load labels
labels = {}
if os.path.exists("trainer/labels.txt"):
    with open("trainer/labels.txt", "r") as f:
        for line in f:
            name, idx = line.strip().split(",")
            labels[int(idx)] = name

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
            face = face_cascade.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in face:
                face_img = img[y:y+h, x:x+w]
                name = os.path.splitext(filename)[0]
                if name not in label_map:
                    label_map[name] = label_id
                    label_id += 1
                faces.append(face_img)
                ids.append(label_map[name])

    if faces:
        recognizer.train(faces, np.array(ids))
        recognizer.save("trainer/trainer.yml")
        with open("trainer/labels.txt", "w") as f:
            for name, idx in label_map.items():
                f.write(f"{name},{idx}\n")
        print("[INFO] Training complete.")

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            label = "Unknown"
            color = (0, 0, 255)

            if len(labels) > 0:
                id_, conf = recognizer.predict(roi_gray)
                if conf < 60:  # confidence threshold
                    label = labels.get(id_, "Unknown")
                    color = (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Face Detector", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('d'):
            # Save face when 'd' is pressed
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_img = gray[y:y+h, x:x+w]
                name = input("Enter name to save face as: ")
                cv2.imwrite(f"faces/{name}.jpg", face_img)
                print(f"[INFO] Saved face as faces/{name}.jpg")
                train_model()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

