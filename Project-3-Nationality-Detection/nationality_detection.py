
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False


class NationalityDetectionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Nationality and Face Analysis")
        self.root.geometry("1050x720")
        self.root.minsize(850, 650)

        self.image = None
        self.image_path = None
        self.preview_photo = None

        title = tk.Label(
            root,
            text="Nationality and Face Analysis",
            font=("Arial", 22, "bold")
        )
        title.pack(pady=12)

        subtitle = tk.Label(
            root,
            text="Upload an image to estimate age, facial expression "
                 "and clothing colour",
            font=("Arial", 11)
        )
        subtitle.pack(pady=3)

        selection_frame = tk.Frame(root)
        selection_frame.pack(pady=12)

        tk.Label(
            selection_frame,
            text="Select nationality / region:",
            font=("Arial", 11, "bold")
        ).grid(row=0, column=0, padx=6, pady=5)

        self.region = tk.StringVar(value="India")

        choices = [
            "India",
            "United States",
            "African Region",
            "Other"
        ]

        self.region_menu = ttk.Combobox(
            selection_frame,
            textvariable=self.region,
            values=choices,
            state="readonly",
            width=20
        )
        self.region_menu.grid(row=0, column=1, padx=6)
        self.region_menu.bind(
            "<<ComboboxSelected>>",
            self.update_nationality_field
        )

        tk.Label(
            selection_frame,
            text="Nationality (for Other):"
        ).grid(row=1, column=0, padx=6, pady=5)

        self.other_nationality = tk.StringVar()

        self.nationality_entry = tk.Entry(
            selection_frame,
            textvariable=self.other_nationality,
            width=23,
            state="disabled"
        )
        self.nationality_entry.grid(row=1, column=1, padx=6)

        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=15, pady=10)

        # Input image panel
        input_frame = tk.LabelFrame(
            main_frame,
            text="Input Image Preview",
            font=("Arial", 12, "bold"),
            padx=10,
            pady=10
        )
        input_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=8
        )

        self.image_label = tk.Label(
            input_frame,
            text="Upload an image to preview it here",
            bg="#eeeeee",
            width=40,
            height=18
        )
        self.image_label.pack(fill="both", expand=True)

        tk.Button(
            input_frame,
            text="Upload Image",
            command=self.upload_image,
            font=("Arial", 11),
            bg="#1976D2",
            fg="white",
            padx=12,
            pady=7
        ).pack(pady=12)

        # Results panel
        output_frame = tk.LabelFrame(
            main_frame,
            text="Analysis Results",
            font=("Arial", 12, "bold"),
            padx=12,
            pady=12
        )
        output_frame.pack(
            side="right",
            fill="both",
            expand=True,
            padx=8
        )

        self.result_label = tk.Label(
            output_frame,
            text="Your results will appear here.",
            font=("Arial", 11),
            justify="left",
            anchor="nw",
            wraplength=390
        )
        self.result_label.pack(
            fill="both",
            expand=True,
            anchor="nw"
        )

        tk.Button(
            root,
            text="Analyze Image",
            command=self.analyze_image,
            font=("Arial", 12, "bold"),
            bg="#2E7D32",
            fg="white",
            padx=20,
            pady=9
        ).pack(pady=12)

        tk.Label(
            root,
            text="Note: Nationality is selected by the user. "
                 "Age and expression are estimates, not verified facts.",
            font=("Arial", 9),
            fg="gray"
        ).pack(pady=5)

    def update_nationality_field(self, event=None):
        if self.region.get() == "Other":
            self.nationality_entry.config(state="normal")
        else:
            self.nationality_entry.config(state="disabled")
            self.other_nationality.set("")

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select a person's image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")
            ]
        )

        if not path:
            return

        image = cv2.imread(path)

        if image is None:
            messagebox.showerror(
                "Error",
                "The selected image could not be opened."
            )
            return

        self.image = image
        self.image_path = path

        # Show a preview without changing the original image.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        pil_image.thumbnail((440, 400))

        self.preview_photo = ImageTk.PhotoImage(pil_image)
        self.image_label.config(
            image=self.preview_photo,
            text=""
        )

        self.result_label.config(
            text="Image uploaded successfully.\n\n"
                 "Select the correct region and click Analyze Image."
        )

    def get_clothing_colour(self):
        """
        Estimate the dominant colour in the central lower image area.
        This is only a rough clothing-region estimate; the area may
        include skin, background or other objects.
        """
        height, width = self.image.shape[:2]

        y1 = int(height * 0.35)
        y2 = int(height * 0.85)
        x1 = int(width * 0.25)
        x2 = int(width * 0.75)

        region = self.image[y1:y2, x1:x2]

        if region.size == 0:
            return "Unable to estimate"

        # Reduce image size to make colour estimation faster.
        region = cv2.resize(region, (100, 100))
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        pixels = hsv.reshape(-1, 3)
        pixels = pixels[pixels[:, 2] > 25]

        if len(pixels) == 0:
            return "Too dark to estimate"

        # Use the most common hue/saturation/value range.
        hue = float(np.median(pixels[:, 0]))
        saturation = float(np.median(pixels[:, 1]))
        value = float(np.median(pixels[:, 2]))

        if value < 55:
            return "Black / very dark"
        elif saturation < 30 and value > 190:
            return "White / light"
        elif saturation < 30:
            return "Grey"
        elif hue < 10 or hue >= 170:
            return "Red"
        elif hue < 22:
            return "Orange / brown"
        elif hue < 35:
            return "Yellow"
        elif hue < 85:
            return "Green"
        elif hue < 130:
            return "Blue"
        elif hue < 150:
            return "Purple"
        else:
            return "Pink / red"

    def analyze_image(self):
        if self.image is None:
            messagebox.showwarning(
                "No Image",
                "Please upload an image first."
            )
            return

        if not DEEPFACE_AVAILABLE:
            messagebox.showerror(
                "Library Missing",
                "DeepFace is not installed.\n\n"
                "Open the VS Code terminal and run:\n"
                "python3 -m pip install deepface"
            )
            return

        region = self.region.get()

        if region == "Other":
            nationality = self.other_nationality.get().strip()

            if not nationality:
                messagebox.showwarning(
                    "Nationality Required",
                    "Enter the nationality for the Other option."
                )
                return
        elif region == "African Region":
            nationality = "African region (user selected)"
        else:
            nationality = region

        self.result_label.config(
            text="Analyzing image...\nPlease wait."
        )
        self.root.update_idletasks()

        try:
            # DeepFace estimates age and facial expression.
            analysis = DeepFace.analyze(
                img_path=self.image,
                actions=["age", "emotion"],
                detector_backend="opencv",
                enforce_detection=False,
                silent=True
            )

            if isinstance(analysis, list):
                face = analysis[0]
            else:
                face = analysis

            estimated_age = face.get("age", "Unavailable")

            emotions = face.get("emotion", {})
            if emotions:
                estimated_emotion = max(
                    emotions,
                    key=emotions.get
                )
                confidence = emotions[estimated_emotion]
                emotion_text = (
                    f"{estimated_emotion.title()} "
                    f"(model score: {confidence:.1f}%)"
                )
            else:
                emotion_text = "Unavailable"

            results = [
                "ANALYSIS RESULTS",
                "--------------------------",
                f"Selected nationality / region: {nationality}",
                "",
                f"Estimated facial expression: {emotion_text}"
            ]

            if region == "India":
                results.append(
                    f"Estimated age: {estimated_age} years"
                )
                results.append(
                    f"Estimated clothing-region colour: "
                    f"{self.get_clothing_colour()}"
                )

            elif region == "United States":
                results.append(
                    f"Estimated age: {estimated_age} years"
                )

            elif region == "African Region":
                results.append(
                    f"Estimated clothing-region colour: "
                    f"{self.get_clothing_colour()}"
                )

            else:
                # For other nationalities, show the selected nationality
                # and estimated facial expression only.
                pass

            results.extend([
                "",
                "IMPORTANT LIMITATIONS",
                "• Nationality was selected by the user; "
                "it was not predicted from the face.",
                "• Age and expression are model estimates and "
                "may be inaccurate.",
                "• Clothing colour is a rough estimate of an image "
                "region, not guaranteed clothing recognition.",
                "• If multiple people appear, the first detected "
                "face may be analysed."
            ])

            self.result_label.config(
                text="\n".join(results)
            )

        except Exception as error:
            self.result_label.config(
                text="Analysis could not be completed."
            )
            messagebox.showerror(
                "Analysis Error",
                "The face analysis failed.\n\n"
                "Try a clear, front-facing image with good lighting.\n\n"
                f"Details: {error}"
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = NationalityDetectionApp(root)
    root.mainloop()
