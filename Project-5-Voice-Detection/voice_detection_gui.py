
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

SEX_MODEL_PATH = os.path.join(MODEL_DIR, "sex_model.pkl")
AGE_MODEL_PATH = os.path.join(MODEL_DIR, "age_model.pkl")
EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.pkl")

# Must match the training script's feature extraction.
def extract_features(path):
    audio, sr = librosa.load(path, sr=16000, mono=True, duration=5.0)

    if len(audio) < sr // 2:
        raise ValueError("Audio must be at least half a second long.")

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        [
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(zcr),
            np.std(zcr),
            np.mean(rms),
            np.std(rms)
        ]
    ])

    return features.reshape(1, -1)


class VoiceDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Voice Age and Emotion Detection")
        self.root.geometry("760x600")
        self.root.minsize(650, 520)
        self.root.configure(bg="#f1f5f9")

        self.audio_path = None
        self.models = {}

        try:
            self.models["sex"] = joblib.load(SEX_MODEL_PATH)
            self.models["age"] = joblib.load(AGE_MODEL_PATH)
            self.models["emotion"] = joblib.load(EMOTION_MODEL_PATH)
        except Exception as error:
            messagebox.showerror(
                "Model loading error",
                f"Could not load trained models:\n{error}"
            )
            self.root.after(100, self.root.destroy)
            return

        title = tk.Label(
            root,
            text="AI Voice Age & Emotion Detection",
            font=("Arial", 21, "bold"),
            bg="#f1f5f9",
            fg="#172554"
        )
        title.pack(pady=(22, 8))

        subtitle = tk.Label(
            root,
            text="Upload a WAV voice recording for experimental analysis",
            font=("Arial", 11),
            bg="#f1f5f9",
            fg="#475569"
        )
        subtitle.pack(pady=(0, 18))

        self.file_label = tk.Label(
            root,
            text="No audio file selected",
            font=("Arial", 11),
            bg="white",
            fg="#334155",
            wraplength=650,
            padx=18,
            pady=15
        )
        self.file_label.pack(fill="x", padx=30, pady=5)

        button_frame = tk.Frame(root, bg="#f1f5f9")
        button_frame.pack(pady=15)

        tk.Button(
            button_frame,
            text="Upload WAV",
            command=self.upload_audio,
            font=("Arial", 11, "bold"),
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            width=15,
            padx=8,
            pady=8
        ).grid(row=0, column=0, padx=7)

        self.analyse_button = tk.Button(
            button_frame,
            text="Analyse Voice",
            command=self.start_analysis,
            font=("Arial", 11, "bold"),
            bg="#0f766e",
            fg="white",
            activebackground="#115e59",
            width=15,
            padx=8,
            pady=8
        )
        self.analyse_button.grid(row=0, column=1, padx=7)

        tk.Button(
            button_frame,
            text="Reset",
            command=self.reset,
            font=("Arial", 11, "bold"),
            bg="#64748b",
            fg="white",
            width=10,
            padx=8,
            pady=8
        ).grid(row=0, column=2, padx=7)

        self.status_label = tk.Label(
            root,
            text="Ready",
            font=("Arial", 10),
            bg="#f1f5f9",
            fg="#2563eb"
        )
        self.status_label.pack(pady=5)

        self.result_box = tk.Text(
            root,
            height=12,
            font=("Menlo", 11),
            wrap="word",
            bg="white",
            fg="#0f172a",
            padx=15,
            pady=12,
            relief="flat"
        )
        self.result_box.pack(fill="both", expand=True, padx=30, pady=(5, 20))
        self.result_box.insert(
            "1.0",
            "Your analysis results will appear here.\n\n"
            "Note: Age, sex and emotion estimates from voice are "
            "experimental and can be inaccurate."
        )
        self.result_box.config(state="disabled")

    def upload_audio(self):
        path = filedialog.askopenfilename(
            title="Choose a voice recording",
            filetypes=[
                ("WAV audio", "*.wav"),
                ("Audio files", "*.wav *.mp3 *.flac *.ogg")
            ]
        )
        if path:
            self.audio_path = path
            self.file_label.config(text=os.path.basename(path))
            self.set_results("Audio selected. Click Analyse Voice to continue.")
            self.status_label.config(text="Audio ready", fg="#0f766e")

    def start_analysis(self):
        if not self.audio_path:
            messagebox.showwarning("No audio", "Please upload an audio file first.")
            return

        self.analyse_button.config(state="disabled")
        self.status_label.config(text="Analysing audio...", fg="#b45309")
        threading.Thread(target=self.analyse_audio, daemon=True).start()

    def analyse_audio(self):
        try:
            features = extract_features(self.audio_path)

            sex_model = self.models["sex"]
            age_model = self.models["age"]
            emotion_model = self.models["emotion"]

            sex_prediction = str(sex_model.predict(features)[0])

            # Experimental model decision; not a reliable identity test.
            if sex_prediction.lower() != "male":
                result = (
                    "Upload male voice.\n\n"
                    "The model did not classify this recording as male.\n"
                    "This prediction may be incorrect."
                )
                self.root.after(0, self.finish_analysis, result)
                return

            estimated_age = float(age_model.predict(features)[0])
            estimated_age = max(18, min(100, estimated_age))

            result = (
                "VOICE ANALYSIS REPORT\n"
                "=====================\n\n"
                "Model classification: Male\n"
                f"Estimated age: {estimated_age:.1f} years\n\n"
                "Important: The classification and age are uncertain estimates."
            )

            if estimated_age > 60:
                emotion = str(emotion_model.predict(features)[0])
                result += (
                    "\n\nPossible senior citizen: YES (estimated age above 60)\n"
                    f"Estimated emotion: {emotion}\n\n"
                    "Emotion is an uncertain model prediction, not a diagnosis."
                )
            else:
                result += (
                    "\n\nPossible senior citizen: NO (estimated age is 60 or below)"
                )

            self.root.after(0, self.finish_analysis, result)

        except Exception as error:
            self.root.after(
                0,
                self.finish_analysis,
                f"Could not analyse the recording.\n\nDetails: {error}"
            )

    def finish_analysis(self, result):
        self.set_results(result)
        self.status_label.config(text="Analysis complete", fg="#0f766e")
        self.analyse_button.config(state="normal")

    def set_results(self, result):
        self.result_box.config(state="normal")
        self.result_box.delete("1.0", tk.END)
        self.result_box.insert("1.0", result)
        self.result_box.config(state="disabled")

    def reset(self):
        self.audio_path = None
        self.file_label.config(text="No audio file selected")
        self.set_results("Your analysis results will appear here.")
        self.status_label.config(text="Ready", fg="#2563eb")
        self.analyse_button.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceDetectionGUI(root)
    root.mainloop()
