
import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import soundfile as sf
import librosa
import time
import numpy as np
import os

audio_file = "recorded_voice.wav"
recording = False


def record_voice():
    global audio_file

    try:
        seconds = int(time_entry.get())

        if seconds < 1 or seconds > 30:
            messagebox.showwarning(
                "Invalid Duration",
                "Enter a duration between 1 and 30 seconds."
            )
            return

        messagebox.showinfo(
            "Recording",
            "Recording will start after you click OK."
        )

        status_label.config(text="Recording...")
        root.update()

        audio = sd.rec(
            int(seconds * 22050),
            samplerate=22050,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        sf.write(audio_file, audio, 22050)
        status_label.config(text="Recording saved.")
        messagebox.showinfo("Success", "Voice recording saved!")

    except Exception as error:
        status_label.config(text="Recording failed.")
        messagebox.showerror("Recording Error", str(error))


def upload_voice():
    global audio_file

    path = filedialog.askopenfilename(
        title="Choose an audio file",
        filetypes=[
            ("Audio files", "*.wav *.flac *.ogg"),
            ("All files", "*.*")
        ]
    )

    if path:
        audio_file = path
        status_label.config(
            text="Selected: " + os.path.basename(path)
        )


def analyse_voice():
    if not os.path.exists(audio_file):
        messagebox.showwarning(
            "No Audio",
            "Record your voice or upload an audio file first."
        )
        return

    try:
        y, sr = librosa.load(audio_file, sr=22050, mono=True)

        if len(y) == 0:
            messagebox.showwarning("Invalid Audio", "The audio file is empty.")
            return

        duration = len(y) / sr
        rms = float(np.sqrt(np.mean(y ** 2)))
        pitches = librosa.yin(
            y,
            fmin=65,
            fmax=400,
            sr=sr
        )
        valid_pitches = pitches[
            np.isfinite(pitches) & (pitches > 0)
        ]
        pitch = float(np.median(valid_pitches)) if len(valid_pitches) else 0

        result = (
            "VOICE ANALYSIS REPORT\n\n"
            f"Duration: {duration:.2f} seconds\n"
            f"Average volume (RMS): {rms:.4f}\n"
            f"Estimated pitch: {pitch:.1f} Hz\n\n"
            "Gender: Not reliably determined\n"
            "Age: Not reliably determined\n\n"
            "Note: Pitch and volume alone cannot reliably identify "
            "a person's gender, age, or emotions."
        )

        result_label.config(text=result)

        with open("voice_report.txt", "w", encoding="utf-8") as file:
            file.write(result)

        messagebox.showinfo(
            "Analysis Complete",
            "Analysis finished. Report saved as voice_report.txt."
        )

    except Exception as error:
        messagebox.showerror("Analysis Error", str(error))


def reset_app():
    global audio_file

    audio_file = "recorded_voice.wav"
    result_label.config(text="Your analysis report will appear here.")
    status_label.config(text="Ready")
    time_entry.delete(0, tk.END)
    time_entry.insert(0, "5")


root = tk.Tk()
root.title("Voice Detection and Analysis")
root.geometry("700x650")
root.configure(bg="#eef3f8")

tk.Label(
    root,
    text="VOICE DETECTION AND ANALYSIS",
    font=("Arial", 18, "bold"),
    bg="#eef3f8"
).pack(pady=20)

tk.Label(
    root,
    text="Record your voice or upload an audio file",
    font=("Arial", 12),
    bg="#eef3f8"
).pack(pady=5)

tk.Label(
    root,
    text="Recording duration (1–30 seconds):",
    bg="#eef3f8"
).pack(pady=5)

time_entry = tk.Entry(root, justify="center", width=10)
time_entry.insert(0, "5")
time_entry.pack(pady=5)


tk.Button(
    root,
    text="Record Voice",
    command=record_voice,
    width=22
).pack(pady=7)

tk.Button(
    root,
    text="Upload Audio",
    command=upload_voice,
    width=22
).pack(pady=7)

tk.Button(
    root,
    text="Analyse Voice",
    command=analyse_voice,
    width=22
).pack(pady=7)

tk.Button(
    root,
    text="Reset",
    command=reset_app,
    width=22
).pack(pady=7)

status_label = tk.Label(
    root,
    text="Ready",
    bg="#eef3f8",
    fg="blue"
)
status_label.pack(pady=10)

result_label = tk.Label(
    root,
    text="Your analysis report will appear here.",
    bg="white",
    justify="left",
    anchor="nw",
    wraplength=620,
    padx=15,
    pady=15,
    font=("Arial", 11)
)
result_label.pack(fill="both", expand=True, padx=20, pady=15)

root.mainloop()
