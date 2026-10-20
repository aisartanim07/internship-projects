# Project 4: Long Hair Detection

A Python-based computer vision application that detects faces and estimates visible hair-region coverage in uploaded images.

## Features
- Upload and preview an image.
- Detect faces using OpenCV Haar Cascade.
- Analyse colour coverage in regions beside the face.
- Display estimated coverage percentages and visual results.
- Save an analysis report.
- Reset the application for another image.

## Technologies Used
- Python
- OpenCV
- NumPy
- Tkinter
- Pillow

## Files
- `long_hair_detection.py` — Main application.
- `haarcascade_frontalface_default.xml` — Face detection model.

## How to Run
1. Install the required libraries:
   ```bash
   python3 -m pip install opencv-python pillow numpy
   ```
2. Keep the Haar Cascade XML file in the same folder as the Python script.
3. Run:
   ```bash
   python3 long_hair_detection.py
   ```

## Limitations
The application uses colour-based estimates rather than a trained hair-length classifier. Background, clothing, lighting, and hair colour can affect the results. Hair length cannot determine a person's gender.

## Project Status
The graphical interface, face detection, image analysis, and report-saving features have been tested.

