
import cv2
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# ---------- SETTINGS ----------

MODEL_PATH = "sign_model.pkl"
LABELS_PATH = "sign_labels.pkl"
IMAGE_SIZE = (64, 64)

# Testing mode: available all day.
# Set to False for the final 6 PM–10 PM restriction.
TESTING_MODE = True
START_HOUR = 18
END_HOUR = 22

# ---------- LOAD MODEL ----------

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(
        "Model files missing. Run python3 train_model.py first."
    )

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(LABELS_PATH, "rb") as file:
    labels = pickle.load(file)

# ---------- PREDICTION ----------

def predict_sign(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMAGE_SIZE)
    features = gray.flatten().astype("float32") / 255.0
    prediction = model.predict([features])[0]
    return prediction

# ---------- GUI ----------

root = tk.Tk()
root.title("Sign Language Detection System")
root.geometry("1000x720")
root.configure(bg="#eef2f7")

camera = None
camera_running = False
camera_job = None
photo_reference = None

tk.Label(
    root,
    text="SIGN LANGUAGE DETECTION",
    font=("Arial", 22, "bold"),
    bg="#eef2f7",
    fg="#183153"
).pack(pady=15)

status_label = tk.Label(
    root, font=("Arial", 11), bg="#eef2f7"
)
status_label.pack()

prediction_label = tk.Label(
    root,
    text="Predicted Sign: --",
    font=("Arial", 18, "bold"),
    bg="#eef2f7",
    fg="#087f5b"
)
prediction_label.pack(pady=10)

preview = tk.Label(
    root,
    text="Upload an image or start the webcam",
    font=("Arial", 13),
    bg="white",
    relief="groove"
)
preview.pack(padx=20, pady=10, fill="both", expand=True)

button_frame = tk.Frame(root, bg="#eef2f7")
button_frame.pack(pady=15)


# ---------- TIME CHECK ----------

def is_operating_time():
    if TESTING_MODE:
        return True

    from datetime import datetime
    hour = datetime.now().hour
    return START_HOUR <= hour < END_HOUR


def check_access():
    if is_operating_time():
        status_label.config(
            text="System active" +
                 (" | Testing mode: all day" if TESTING_MODE
                  else " | Operating hours: 6 PM–10 PM"),
            fg="#087f5b"
        )
        upload_button.config(state="normal")
        camera_button.config(state="normal")
        return True

    stop_camera()
    status_label.config(
        text="System unavailable. Operating hours: 6 PM–10 PM.",
        fg="red"
    )
    prediction_label.config(text="System is currently closed")
    upload_button.config(state="disabled")
    camera_button.config(state="disabled")
    return False


# ---------- DISPLAY IMAGE ----------

def show_image(image):
    global photo_reference

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    pil_image.thumbnail((850, 450))

    photo_reference = ImageTk.PhotoImage(pil_image)
    preview.config(image=photo_reference, text="")


# ---------- IMAGE UPLOAD ----------

def upload_image():
    if not check_access():
        return

    path = filedialog.askopenfilename(
        title="Select a sign image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ]
    )

    if not path:
        return

    image = cv2.imread(path)

    if image is None:
        messagebox.showerror("Error", "Could not read this image.")
        return

    try:
        result = predict_sign(image)
        show_image(image)
        prediction_label.config(text=f"Predicted Sign: {result}")
    except Exception as error:
        messagebox.showerror("Prediction Error", str(error))


# ---------- WEBCAM ----------

def start_camera():
    global camera, camera_running

    if not check_access() or camera_running:
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        camera = None
        messagebox.showerror(
            "Camera Error",
            "Could not open webcam. Check Mac camera permissions."
        )
        return

    camera_running = True
    camera_button.config(text="Stop Webcam")
    update_camera()


def update_camera():
    global camera_job

    if not camera_running:
        return

    if not check_access():
        return

    if camera is None:
        return

    success, frame = camera.read()

    if not success:
        stop_camera()
        messagebox.showerror("Camera Error", "Could not read webcam.")
        return

    frame = cv2.flip(frame, 1)

    try:
        result = predict_sign(frame)
        prediction_label.config(text=f"Predicted Sign: {result}")
    except Exception:
        prediction_label.config(text="Prediction unavailable")

    show_image(frame)
    camera_job = root.after(100, update_camera)


def stop_camera():
    global camera, camera_running, camera_job

    camera_running = False

    if camera_job is not None:
        try:
            root.after_cancel(camera_job)
        except tk.TclError:
            pass
        camera_job = None

    if camera is not None:
        camera.release()
        camera = None

    if "camera_button" in globals():
        camera_button.config(text="Start Webcam")


def toggle_camera():
    if camera_running:
        stop_camera()
    else:
        start_camera()


# ---------- CLOSE APP ----------

def close_app():
    stop_camera()
    root.destroy()


# ---------- BUTTONS ----------

upload_button = tk.Button(
    button_frame,
    text="Upload Image",
    font=("Arial", 12, "bold"),
    width=16,
    bg="#1971c2",
    fg="white",
    command=upload_image
)
upload_button.grid(row=0, column=0, padx=10)

camera_button = tk.Button(
    button_frame,
    text="Start Webcam",
    font=("Arial", 12, "bold"),
    width=16,
    bg="#087f5b",
    fg="white",
    command=toggle_camera
)
camera_button.grid(row=0, column=1, padx=10)

exit_button = tk.Button(
    button_frame,
    text="Exit",
    font=("Arial", 12, "bold"),
    width=12,
    bg="#c92a2a",
    fg="white",
    command=close_app
)
exit_button.grid(row=0, column=2, padx=10)


def periodic_time_check():
    check_access()
    root.after(30000, periodic_time_check)


root.protocol("WM_DELETE_WINDOW", close_app)

check_access()
periodic_time_check()
root.mainloop()
