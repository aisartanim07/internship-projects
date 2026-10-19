
import cv2
import os
import time

signs = ["Hello", "Yes", "No", "Thank_You", "Help"]

for sign in signs:
    os.makedirs(f"dataset/{sign}", exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 1-5 to select a sign.")
print("Press A to automatically collect 30 images.")
print("Press Q to quit.")

current_sign = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error reading webcam.")
        break

    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        f"Selected: {current_sign}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Sign Language Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key in [ord("1"), ord("2"), ord("3"),
               ord("4"), ord("5")]:
        current_sign = signs[key - ord("1")]
        print("Selected:", current_sign)

    if key == ord("a") and current_sign:
        folder = f"dataset/{current_sign}"
        existing = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        print(f"Collecting images for {current_sign}...")

        for i in range(30):
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)
            filename = f"{folder}/{existing + i}.jpg"

            cv2.imwrite(filename, frame)

            cv2.putText(
                frame,
                f"Capturing {i + 1}/30",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Sign Language Data Collection", frame)
            cv2.waitKey(150)

        print(f"Saved images for {current_sign}")
        time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()
