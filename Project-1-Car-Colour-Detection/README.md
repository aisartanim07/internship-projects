
# Car Colour Detection

## Project Description
This project detects cars and people in traffic images and videos using YOLO and OpenCV. It classifies cars as blue or other colours and displays the detection results in a graphical user interface.

## Features
- Upload traffic images and videos.
- Detect cars and people.
- Identify blue cars.
- Display red boxes for blue cars.
- Display blue boxes for other cars.
- Display green boxes for people.
- Count unique tracked cars in videos.
- Show detection results in a GUI.

## Technologies Used
- Python
- OpenCV
- NumPy
- YOLOv8
- Tkinter
- Pillow

## Installation
Install the required libraries:

pip install ultralytics opencv-python numpy pillow

## How to Run
Run the following command:

python3 car_detection.py

## Project Files
- car_detection.py: Main Python program
- yolov8n.pt: YOLOv8 model

## Note
Car colour classification and counting are approximate and may vary depending on the video quality and tracking accuracy.