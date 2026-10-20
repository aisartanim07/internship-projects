
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
from datetime import datetime


class LongHairDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Long Hair Detection")
        self.root.geometry("1100x760")
        self.root.minsize(850, 620)
        self.root.configure(bg="#f1f5f9")

        self.image = None
        self.output_image = None
        self.photo = None
        self.report = ""

        tk.Label(
            root,
            text="Long Hair Detection",
            font=("Arial", 24, "bold"),
            bg="#f1f5f9",
            fg="#243b55"
        ).pack(pady=12)

        tk.Label(
            root,
            text="Upload a photo to analyse visible hair-region features",
            font=("Arial", 12),
            bg="#f1f5f9"
        ).pack(pady=3)

        main = tk.Frame(root, bg="#f1f5f9")
        main.pack(fill="both", expand=True, padx=12, pady=10)

        left = tk.LabelFrame(
            main, text="Image Preview",
            font=("Arial", 12, "bold"), padx=8, pady=8
        )
        left.pack(side="left", fill="both", expand=True, padx=6)

        self.image_label = tk.Label(
            left, text="Upload an image to begin",
            bg="white", font=("Arial", 12)
        )
        self.image_label.pack(fill="both", expand=True)

        tk.Button(
            left, text="Upload Image",
            command=self.upload_image,
            font=("Arial", 11, "bold"),
            bg="#1976d2", fg="white",
            padx=12, pady=7
        ).pack(pady=8)

        right = tk.LabelFrame(
            main, text="Analysis Results",
            font=("Arial", 12, "bold"), padx=10, pady=10
        )
        right.pack(side="right", fill="both", expand=True, padx=6)

        self.result_label = tk.Label(
            right,
            text="Upload a photo, then click Analyze Image.",
            font=("Arial", 11),
            justify="left",
            anchor="nw",
            wraplength=430
        )
        self.result_label.pack(fill="both", expand=True)

        buttons = tk.Frame(root, bg="#f1f5f9")
        buttons.pack(pady=8)

        tk.Button(
            buttons, text="Analyze Image",
            command=self.analyze_image,
            font=("Arial", 12, "bold"),
            bg="#218838", fg="white",
            padx=18, pady=9
        ).pack(side="left", padx=6)

        tk.Button(
            buttons, text="Save Report",
            command=self.save_report,
            font=("Arial", 11, "bold"),
            bg="#6f42c1", fg="white",
            padx=14, pady=9
        ).pack(side="left", padx=6)

        tk.Button(
            buttons, text="Reset",
            command=self.reset,
            font=("Arial", 11, "bold"),
            bg="#64748b", fg="white",
            padx=14, pady=9
        ).pack(side="left", padx=6)

        tk.Label(
            root,
            text=(
                "Educational demo: colour coverage is only a rough "
                "estimate. Hair length does not determine gender."
            ),
            font=("Arial", 9),
            bg="#f1f5f9", fg="gray"
        ).pack(pady=5)

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Choose a photo",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")
            ]
        )

        if not path:
            return

        image = cv2.imread(path)

        if image is None:
            messagebox.showerror(
                "Image Error",
                "The selected image could not be opened."
            )
            return

        self.image = image
        self.output_image = None
        self.report = ""
        self.show_image(image)

        self.result_label.config(
            text="Image uploaded successfully.\n\n"
                 "Click Analyze Image to continue."
        )

    def show_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preview = Image.fromarray(rgb)
        preview.thumbnail((480, 450))

        self.photo = ImageTk.PhotoImage(preview)
        self.image_label.config(image=self.photo, text="")

    def load_face_detector(self):
        # First check the XML file inside the project folder.
        local_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "haarcascade_frontalface_default.xml"
        )

        possible_paths = [local_path]

        # Also check OpenCV's built-in cascade directory.
        try:
            possible_paths.append(
                os.path.join(
                    cv2.data.haarcascades,
                    "haarcascade_frontalface_default.xml"
                )
            )
        except AttributeError:
            pass

        for path in possible_paths:
            if os.path.isfile(path):
                detector = cv2.CascadeClassifier(path)
                if not detector.empty():
                    return detector

        return None

    @staticmethod
    def estimate_coverage(region):
        """
        Estimates dark or colourful pixel coverage.
        This is a visual heuristic, not a trained hair classifier.
        """
        if region is None or region.size == 0:
            return None

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        brightness = hsv[:, :, 2]

        mask = (
            (brightness < 100)
            | ((saturation > 55) & (brightness < 175))
        )

        return float(np.mean(mask))

    def analyze_image(self):
        if self.image is None:
            messagebox.showwarning(
                "No Image",
                "Please upload an image first."
            )
            return

        detector = self.load_face_detector()

        if detector is None:
            messagebox.showerror(
                "Face Model Missing",
                "The face-detection XML file was not found.\n\n"
                "Open the project terminal and run:\n\n"
                'curl -fL "https://raw.githubusercontent.com/'
                'opencv/opencv/4.x/data/haarcascades/'
                'haarcascade_frontalface_default.xml" '
                "-o haarcascade_frontalface_default.xml\n\n"
                "Then run the application again."
            )
            return

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            self.result_label.config(
                text="No face detected.\n\n"
                     "Try a clear, front-facing photo with good lighting."
            )
            return

        # Analyse the largest detected face.
        x, y, w, h = max(
            faces, key=lambda face: face[2] * face[3]
        )

        image_h, image_w = self.image.shape[:2]

        # Estimate regions beside the face where hair may appear.
        side_width = max(10, int(w * 0.55))
        side_top = max(0, y + int(h * 0.10))
        side_bottom = min(image_h, y + int(h * 1.65))

        left_x1 = max(0, x - side_width)
        left_x2 = max(0, x)
        right_x1 = min(image_w, x + w)
        right_x2 = min(image_w, x + w + side_width)

        left_region = self.image[
            side_top:side_bottom, left_x1:left_x2
        ]
        right_region = self.image[
            side_top:side_bottom, right_x1:right_x2
        ]

        left_coverage = self.estimate_coverage(left_region)
        right_coverage = self.estimate_coverage(right_region)

        available = [
            value for value in (left_coverage, right_coverage)
            if value is not None
        ]

        if not available:
            self.result_label.config(
                text="There is not enough image area beside the face "
                     "to perform the analysis. Try another photo."
            )
            return

        average = sum(available) / len(available)

        # These thresholds are illustrative, not trained predictions.
        if average >= 0.45 and len(available) == 2:
            estimate = "Possible long-hair pattern"
        elif average <= 0.18:
            estimate = "Low dark/colourful pixel coverage"
        else:
            estimate = "Unclear — manual review recommended"

        output = self.image.copy()

        # Green rectangle: detected face.
        cv2.rectangle(
            output, (x, y), (x + w, y + h), (0, 200, 0), 3
        )

        # Orange rectangles: side regions analysed.
        if left_x2 > left_x1 and side_bottom > side_top:
            cv2.rectangle(
                output,
                (left_x1, side_top),
                (left_x2, side_bottom),
                (0, 140, 255), 2
            )

        if right_x2 > right_x1 and side_bottom > side_top:
            cv2.rectangle(
                output,
                (right_x1, side_top),
                (right_x2, side_bottom),
                (0, 140, 255), 2
            )

        self.output_image = output
        self.show_image(output)

        left_text = (
            f"{left_coverage * 100:.1f}%"
            if left_coverage is not None else "Unavailable"
        )
        right_text = (
            f"{right_coverage * 100:.1f}%"
            if right_coverage is not None else "Unavailable"
        )

        self.report = (
            "LONG HAIR DETECTION - ANALYSIS REPORT\n"
            "=====================================\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "Face detected: Yes\n"
            f"Face width: {w} pixels\n"
            f"Face height: {h} pixels\n"
            f"Left-side dark/colourful coverage: {left_text}\n"
            f"Right-side dark/colourful coverage: {right_text}\n"
            f"Visual heuristic result: {estimate}\n\n"
            "LIMITATIONS\n"
            "This result is based on pixel colour coverage, not a "
            "trained hair-length classifier. Background, clothing, "
            "lighting and hair colour may affect the estimate. "
            "A labelled dataset and validated model are required "
            "for dependable long/short hair classification. "
            "Hair length cannot determine a person's gender.\n"
        )

        self.result_label.config(
            text=(
                "ANALYSIS COMPLETED\n\n"
                "Face: Detected\n"
                f"Face width: {w} pixels\n"
                f"Face height: {h} pixels\n\n"
                "SIDE-REGION ESTIMATES\n"
                f"Left-side coverage: {left_text}\n"
                f"Right-side coverage: {right_text}\n\n"
                "Visual estimate:\n"
                f"{estimate}\n\n"
                "Green box: detected face\n"
                "Orange boxes: analysed side regions\n\n"
                "LIMITATIONS\n"
                "This is a colour-based heuristic, not a trained "
                "hair detector. Clothing, shadows, background and "
                "hair colour can affect the result. It cannot "
                "reliably confirm hair length.\n\n"
                "Hair length cannot determine a person's gender."
            )
        )

    def save_report(self):
        if not self.report:
            messagebox.showwarning(
                "No Report",
                "Analyse an image before saving the report."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save analysis report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            initialfile="long_hair_analysis_report.txt"
        )

        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as file:
                file.write(self.report)

            messagebox.showinfo(
                "Saved",
                "Analysis report saved successfully."
            )
        except OSError as error:
            messagebox.showerror(
                "Save Error",
                f"Could not save the report:\n{error}"
            )

    def reset(self):
        self.image = None
        self.output_image = None
        self.report = ""
        self.photo = None

        self.image_label.config(
            image="", text="Upload an image to begin"
        )
        self.result_label.config(
            text="Upload a photo, then click Analyze Image."
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = LongHairDetectionApp(root)
    root.mainloop()
