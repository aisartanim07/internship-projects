# Senior Citizen Detection System

## Project Overview
This project detects faces in webcam feeds, videos, and images. It estimates age bands and gender, highlights faces estimated in the 60–100 age band, and records detections in a CSV file.

## Features
- Webcam face detection
- Video and image input
- Estimated age-band and gender prediction
- Possible senior citizen highlighting
- CSV logging with timestamp, source, face number, age band, predicted gender, and possible senior status
- Tkinter graphical interface

## Technologies Used
- Python
- OpenCV
- NumPy
- Pillow
- Tkinter
- CSV

## Requirements
Install packages using:

    python3 -m pip install opencv-python numpy pillow

## How to Run
1. Ensure the `models` folder and its model files are present.
2. Open Terminal in this folder.
3. Run:

    python3 senior_citizen_detection.py

4. Select Start Webcam, Open Video, or Open Image.
5. The log is saved as `visit_log.csv`.

## Limitations
Age and gender predictions may be inaccurate. The model estimates broad age bands rather than exact ages. Face tracking is approximate and may create duplicate entries when people move, disappear, or cross paths. The system does not identify real-world identities.

## Responsible Use
Use this project for educational demonstrations. Obtain appropriate consent before processing people and handle video and CSV data responsibly.
