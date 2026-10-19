import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import Counter

model = YOLO("yolov8n.pt")

video = None
image = None
running = False
car_votes = {}

root = tk.Tk()
root.title("Car Colour Detection")
root.geometry("1150x760")
root.configure(bg="#f1f5f9")

tk.Label(
    root,
    text="Car Colour Detection",
    font=("Arial", 24, "bold"),
    bg="#f1f5f9"
).pack(pady=15)

panels = tk.Frame(root, bg="#f1f5f9")
panels.pack()

tk.Label(
    panels,
    text="Input Preview",
    font=("Arial", 14, "bold"),
    bg="#f1f5f9"
).grid(row=0, column=0)

tk.Label(
    panels,
    text="Detection Output",
    font=("Arial", 14, "bold"),
    bg="#f1f5f9"
).grid(row=0, column=1)

input_view = tk.Canvas(
    panels,
    width=480,
    height=340,
    bg="white",
    highlightthickness=1
)
input_view.grid(row=1, column=0, padx=10, pady=10)

output_view = tk.Canvas(
    panels,
    width=480,
    height=340,
    bg="white",
    highlightthickness=1
)
output_view.grid(row=1, column=1, padx=10, pady=10)


def show_frame(frame, canvas):
    canvas.delete("all")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img.thumbnail((460, 320))

    photo = ImageTk.PhotoImage(img)

    canvas.create_image(
        240,
        170,
        image=photo,
        anchor="center"
    )

    canvas.image = photo


def get_colour(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))

    car = frame[y1:y2, x1:x2]

    if car.size == 0:
        return "Other"

    h, w = car.shape[:2]

    if h < 20 or w < 20:
        return "Other"

    body = car[
        int(h * 0.4):int(h * 0.85),
        int(w * 0.15):int(w * 0.85)
    ]

    if body.size == 0:
        return "Other"

    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array([95, 90, 45]),
        np.array([130, 255, 255])
    )

    ratio = cv2.countNonZero(mask) / mask.size

    if ratio > 0.35:
        return "Blue"

    return "Other"


def upload_image():
    global image

    stop_video()

    path = filedialog.askopenfilename(
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png")
        ]
    )

    if path:
        image = cv2.imread(path)

        if image is not None:
            show_frame(image, input_view)
            output_view.delete("all")

            status.config(
                text="Image uploaded. Click Detect Image."
            )


def detect_image():
    if image is None:
        status.config(
            text="Please upload an image first."
        )
        return

    frame = image.copy()

    blue = 0
    other = 0
    people = 0

    results = model.predict(
        frame,
        conf=0.5,
        classes=[0, 2],
        verbose=False
    )

    for result in results:

        for box in result.boxes:

            obj = int(box.cls[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            if obj == 2:

                colour = get_colour(
                    image,
                    x1,
                    y1,
                    x2,
                    y2
                )

                if colour == "Blue":
                    blue += 1
                    color = (0, 0, 255)
                    name = "Blue Car"

                else:
                    other += 1
                    color = (255, 0, 0)
                    name = "Other Car"

            elif obj == 0:

                people += 1
                color = (0, 255, 0)
                name = "Person"

            else:
                continue

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                name,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.putText(
        frame,
        f"Blue Cars: {blue}  Other Cars: {other}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"People: {people}",
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255, 255),
        2
    )

    show_frame(frame, output_view)

    status.config(
        text=f"Blue Cars: {blue} | "
             f"Other Cars: {other} | "
             f"People: {people}"
    )


def upload_video():
    global video, car_votes

    stop_video()

    path = filedialog.askopenfilename(
        filetypes=[
            ("Videos", "*.mp4 *.avi *.mov *.mkv")
        ]
    )

    if path:

        video = cv2.VideoCapture(path)

        if not video.isOpened():

            status.config(
                text="Unable to open video."
            )

            video = None
            return

        car_votes = {}

        status.config(
            text="Video uploaded. Click Start Video."
        )


def stop_video():
    global running, video

    running = False

    if video is not None:
        video.release()
        video = None


def start_video():
    global running

    if video is None:

        status.config(
            text="Please upload a video first."
        )

        return

    if not running:

        running = True
        process_video()


def process_video():
    global running, car_votes

    if not running or video is None:
        return

    ret, frame = video.read()

    if not ret:

        running = False

        blue = sum(
            1
            for votes in car_votes.values()
            if Counter(votes).most_common(1)[0][0] == "Blue"
        )

        other = len(car_votes) - blue

        status.config(
            text=f"Video finished | "
                 f"Blue: {blue} | "
                 f"Other: {other} | "
                 f"Total: {len(car_votes)}"
        )

        print("Video finished")
        print("Blue cars:", blue)
        print("Other cars:", other)
        print("Total unique cars:", len(car_votes))

        return

    input_frame = frame.copy()
    output = frame.copy()

    people = 0

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.45,
        classes=[0, 2],
        verbose=False
    )

    for result in results:

        for box in result.boxes:

            obj = int(box.cls[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            if obj == 2:

                colour = get_colour(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2
                )

                if box.id is not None:

                    track_id = int(box.id[0])

                    car_votes.setdefault(
                        track_id,
                        []
                    ).append(colour)

                    final_colour = Counter(
                        car_votes[track_id]
                    ).most_common(1)[0][0]

                else:

                    final_colour = colour

                if final_colour == "Blue":

                    color = (0, 0, 255)
                    name = "Blue Car"

                else:

                    color = (255, 0, 0)
                    name = "Other Car"

            elif obj == 0:

                people += 1

                color = (0, 255, 0)
                name = "Person"

            else:
                continue

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                output,
                name,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    blue = sum(
        1
        for votes in car_votes.values()
        if Counter(votes).most_common(1)[0][0] == "Blue"
    )

    other = len(car_votes) - blue

    cv2.putText(
        output,
        f"Blue Cars: {blue}  Other: {other}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Total Cars: {len(car_votes)}",
        (15, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"People: {people}",
        (15, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255, 255),
        2
    )

    show_frame(
        input_frame,
        input_view
    )

    show_frame(
        output,
        output_view
    )

    status.config(
        text=f"Blue: {blue} | "
             f"Other: {other} | "
             f"Total: {len(car_votes)} | "
             f"People visible: {people}"
    )

    root.after(
        30,
        process_video
    )


buttons = tk.Frame(
    root,
    bg="#f1f5f9"
)

buttons.pack(pady=15)

tk.Button(
    buttons,
    text="Upload Image",
    command=upload_image,
    bg="#2563eb",
    fg="white",
    padx=10,
    pady=8
).grid(row=0, column=0, padx=5)

tk.Button(
    buttons,
    text="Detect Image",
    command=detect_image,
    bg="#16a34a",
    fg="white",
    padx=10,
    pady=8
).grid(row=0, column=1, padx=5)

tk.Button(
    buttons,
    text="Upload Video",
    command=upload_video,
    bg="#2563eb",
    fg="white",
    padx=10,
    pady=8
).grid(row=0, column=2, padx=5)

tk.Button(
    buttons,
    text="Start Video",
    command=start_video,
    bg="#16a34a",
    fg="white",
    padx=10,
    pady=8
).grid(row=0, column=3, padx=5)

tk.Button(
    buttons,
    text="Stop Video",
    command=stop_video,
    bg="#dc2626",
    fg="white",
    padx=10,
    pady=8
).grid(row=0, column=4, padx=5)

status = tk.Label(
    root,
    text="Upload an image or video to begin",
    font=("Arial", 12),
    bg="#f1f5f9"
)

status.pack(pady=10)

root.mainloop()
