
# Project 3: Nationality and Face Analysis

A Python GUI application for image-based face analysis and clothing-colour estimation.

## Features
- Upload an image and preview it.
- Select a nationality or region.
- Estimate age and facial expression using DeepFace.
- Estimate clothing-region colour using OpenCV.
- Display results in a graphical interface.

## Technologies Used
- Python
- Tkinter
- OpenCV
- Pillow
- NumPy
- DeepFace

## Installation

```bash
python3 -m pip install deepface opencv-python pillow numpy
```

## Run the Project

```bash
python3 nationality_detection.py
```

## Limitations
Nationality is selected manually and is not predicted from appearance. Age and facial-expression results are estimates. Clothing colour is an approximate image-region estimate.
