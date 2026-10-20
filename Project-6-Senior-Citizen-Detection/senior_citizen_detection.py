
import os
import csv
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk


# -------------------- CONFIGURATION --------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CSV_FILE = os.path.join(BASE_DIR, "visit_log.csv")

FACE_CONFIG = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(
    MODEL_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)
AGE_CONFIG = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_net.caffemodel")
GENDER_CONFIG = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")

AGE_LABELS = [
    "0-2", "4-6", "8-12", "15-20",
    "25-32", "38-43", "48-53", "60-100"
]
GENDER_LABELS = ["Male", "Female"]

FACE_CONFIDENCE = 0.60

# Tracking settings
TRACK_TIMEOUT_SECONDS = 12.0
MATCH_DISTANCE_PIXELS = 180
MIN_IOU_MATCH = 0.05

# Model estimates are approximate.
# "60-100" is an age band, not an exact age.


# -------------------- CSV LOGGING --------------------

def prepare_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Timestamp",
                "Source",
                "Face_Number",
                "Estimated_Age_Band",
                "Predicted_Gender",
                "Possible_Senior_Citizen"
            ])


def log_visit(source, face_number, age_band, gender, is_senior):
    prepare_csv()

    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source,
            face_number,
            age_band,
            gender,
            "Yes" if is_senior else "No"
        ])


# -------------------- FACE TRACKING --------------------

def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


class FaceTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def match_faces(self, boxes, now):
        # Remove tracks that have been absent for too long.
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if now - track["last_seen"] > TRACK_TIMEOUT_SECONDS
        ]

        for track_id in expired:
            del self.tracks[track_id]

        assignments = {}
        used_tracks = set()

        # Match strongest bounding-box overlaps first.
        candidates = []

        for box_index, box in enumerate(boxes):
            center = box_center(box)

            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue

                old_box = track["box"]
                old_center = track["center"]

                distance = float(np.hypot(
                    center[0] - old_center[0],
                    center[1] - old_center[1]
                ))

                overlap = calculate_iou(box, old_box)

                max_distance = max(
                    MATCH_DISTANCE_PIXELS,
                    int(max(
                        box[2] - box[0],
                        box[3] - box[1],
                        old_box[2] - old_box[0],
                        old_box[3] - old_box[1]
                    ) * 1.5)
                )

                if overlap >= MIN_IOU_MATCH or distance <= max_distance:
                    # Prefer overlap; distance breaks close matches.
                    score = overlap - (distance / max_distance) * 0.25
                    candidates.append(
                        (score, box_index, track_id, distance)
                    )

        candidates.sort(reverse=True, key=lambda item: item[0])

        assigned_boxes = set()

        for score, box_index, track_id, distance in candidates:
            if box_index in assigned_boxes or track_id in used_tracks:
                continue

            box = boxes[box_index]
            track = self.tracks[track_id]

            track["box"] = box
            track["center"] = box_center(box)
            track["last_seen"] = now

            assignments[box_index] = track_id
            assigned_boxes.add(box_index)
            used_tracks.add(track_id)

        # Create tracks only for faces that were not matched.
        for box_index, box in enumerate(boxes):
            if box_index in assigned_boxes:
                continue

            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = {
                "box": box,
                "center": box_center(box),
                "last_seen": now,
                "logged": False
            }

            assignments[box_index] = track_id

        return assignments


# -------------------- MAIN APPLICATION --------------------

class SeniorCitizenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Senior Citizen Detection System")
        self.root.geometry("1100x760")
        self.root.minsize(850, 600)

        self.cap = None
        self.running = False
        self.after_id = None
        self.source_name = "Webcam"
        self.tracker = FaceTracker()
        self.last_frame = None
        self.photo = None

        self.face_net = None
        self.age_net = None
        self.gender_net = None

        self.load_models()
        self.build_gui()
        prepare_csv()

        self.root.protocol("WM_DELETE_WINDOW", self.close_app)

    def load_models(self):
        required_files = [
            FACE_CONFIG, FACE_MODEL,
            AGE_CONFIG, AGE_MODEL,
            GENDER_CONFIG, GENDER_MODEL
        ]

        missing = [path for path in required_files if not os.path.exists(path)]

        if missing:
            messagebox.showerror(
                "Missing model files",
                "Some model files are missing:\n\n"
                + "\n".join(missing)
            )
            self.root.after(100, self.root.destroy)
            return

        try:
            self.face_net = cv2.dnn.readNetFromCaffe(
                FACE_CONFIG, FACE_MODEL
            )
            self.age_net = cv2.dnn.readNetFromCaffe(
                AGE_CONFIG, AGE_MODEL
            )
            self.gender_net = cv2.dnn.readNetFromCaffe(
                GENDER_CONFIG, GENDER_MODEL
            )
        except Exception as error:
            messagebox.showerror(
                "Model loading error",
                f"Could not load the models:\n{error}"
            )
            self.root.after(100, self.root.destroy)

    def build_gui(self):
        title = tk.Label(
            self.root,
            text="Senior Citizen Detection System",
            font=("Arial", 20, "bold")
        )
        title.pack(pady=12)

        info = tk.Label(
            self.root,
            text=(
                "Detect faces, estimate age band and gender, "
                "and log visits to CSV"
            ),
            font=("Arial", 10)
        )
        info.pack(pady=(0, 10))

        buttons = ttk.Frame(self.root)
        buttons.pack(pady=5)

        ttk.Button(
            buttons,
            text="Start Webcam",
            command=self.start_webcam
        ).grid(row=0, column=0, padx=5)

        ttk.Button(
            buttons,
            text="Open Video",
            command=self.open_video
        ).grid(row=0, column=1, padx=5)

        ttk.Button(
            buttons,
            text="Open Image",
            command=self.open_image
        ).grid(row=0, column=2, padx=5)

        ttk.Button(
            buttons,
            text="Stop",
            command=self.stop_detection
        ).grid(row=0, column=3, padx=5)

        ttk.Button(
            buttons,
            text="Open CSV Log",
            command=self.open_csv
        ).grid(row=0, column=4, padx=5)

        self.video_label = tk.Label(
            self.root,
            text="Start the webcam or open an image/video",
            bg="#202020",
            fg="white",
            font=("Arial", 13),
            width=95,
            height=25
        )
        self.video_label.pack(
            fill="both", expand=True, padx=15, pady=12
        )

        self.status_var = tk.StringVar(
            value="Ready. Choose Webcam, Video or Image."
        )

        tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Arial", 10),
            anchor="w"
        ).pack(fill="x", padx=15, pady=(0, 10))

        note = tk.Label(
            self.root,
            text=(
                "Note: Age and gender are model predictions, "
                "not verified facts. Use responsibly."
            ),
            font=("Arial", 9),
            fg="gray"
        )
        note.pack(pady=(0, 8))

    def predict_face(self, face):
        blob = cv2.dnn.blobFromImage(
            face,
            scalefactor=1.0,
            size=(227, 227),
            mean=(78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )

        self.gender_net.setInput(blob)
        gender_scores = self.gender_net.forward()[0]
        gender = GENDER_LABELS[int(np.argmax(gender_scores))]

        self.age_net.setInput(blob)
        age_scores = self.age_net.forward()[0]
        age_band = AGE_LABELS[int(np.argmax(age_scores))]

        is_senior = age_band == "60-100"

        return age_band, gender, is_senior

    def detect_faces(self, frame):
        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        found = []

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])

            if confidence < FACE_CONFIDENCE:
                continue

            coords = detections[0, 0, i, 3:7] * np.array([
                width, height, width, height
            ])

            x1, y1, x2, y2 = coords.astype(int)

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            # Avoid extremely small face crops.
            if (x2 - x1) < 30 or (y2 - y1) < 30:
                continue

            found.append((x1, y1, x2, y2))

        return found

    def process_frame(self, frame):
        now = time.monotonic()
        boxes = self.detect_faces(frame)
        assignments = self.tracker.match_faces(boxes, now)

        for face_number, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box
            track_id = assignments[face_number - 1]

            face = frame[y1:y2, x1:x2]

            try:
                age_band, gender, is_senior = self.predict_face(face)
            except Exception:
                age_band, gender, is_senior = "Unknown", "Unknown", False

            track = self.tracker.tracks[track_id]

            if is_senior:
                color = (0, 180, 0)
                label = f"Possible Senior | {age_band} | {gender}"
            else:
                color = (0, 165, 255)
                label = f"Age {age_band} | {gender}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                label,
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2,
                cv2.LINE_AA
            )

            # One log row per approximate track.
            if not track["logged"]:
                log_visit(
                    self.source_name,
                    face_number,
                    age_band,
                    gender,
                    is_senior
                )
                track["logged"] = True

        cv2.putText(
            frame,
            f"Faces detected: {len(boxes)}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        return frame, len(boxes)

    def show_frame(self, frame):
        self.last_frame = frame.copy()

        max_width = 1000
        max_height = 520

        height, width = frame.shape[:2]
        scale = min(max_width / width, max_height / height, 1.0)

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        resized = cv2.resize(frame, (new_width, new_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image=image)

        self.video_label.configure(
            image=self.photo,
            text="",
            width=new_width,
            height=new_height
        )

    def start_webcam(self):
        self.stop_detection()

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            messagebox.showerror(
                "Webcam error",
                "Could not open the webcam. Check camera permissions."
            )
            return

        self.source_name = "Webcam"
        self.tracker = FaceTracker()
        self.running = True
        self.status_var.set("Webcam running...")
        self.update_frame()

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        self.stop_detection()
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            messagebox.showerror(
                "Video error",
                "Could not open the selected video."
            )
            return

        self.source_name = os.path.basename(path)
        self.tracker = FaceTracker()
        self.running = True
        self.status_var.set("Video running...")
        self.update_frame()

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ok, frame = self.cap.read()

        if not ok:
            self.stop_detection()
            self.status_var.set("Video finished.")
            return

        processed, face_count = self.process_frame(frame)
        self.show_frame(processed)

        self.status_var.set(
            f"Source: {self.source_name} | "
            f"Faces detected: {face_count} | "
            f"CSV: {os.path.basename(CSV_FILE)}"
        )

        self.after_id = self.root.after(30, self.update_frame)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        self.stop_detection()

        frame = cv2.imread(path)

        if frame is None:
            messagebox.showerror(
                "Image error",
                "Could not open the selected image."
            )
            return

        self.source_name = os.path.basename(path)
        self.tracker = FaceTracker()

        processed, face_count = self.process_frame(frame)
        self.show_frame(processed)

        self.status_var.set(
            f"Image: {os.path.basename(path)} | "
            f"Faces detected: {face_count}"
        )

    def stop_detection(self):
        self.running = False

        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def open_csv(self):
        prepare_csv()

        try:
            if os.name == "posix":
                import subprocess
                subprocess.Popen(["open", CSV_FILE])
            else:
                os.startfile(CSV_FILE)
        except Exception as error:
            messagebox.showinfo(
                "CSV file location",
                f"Your visit log is saved here:\n{CSV_FILE}\n\n{error}"
            )

    def close_app(self):
        self.stop_detection()
        self.root.destroy()


# -------------------- RUN APPLICATION --------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = SeniorCitizenApp(root)
    root.mainloop()
