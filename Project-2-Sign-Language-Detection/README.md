
# Sign Language Detection System

## Project Overview
This project is a Python-based Sign Language Detection System that
recognizes five basic hand gestures using machine learning.

## Supported Gestures
- Hello
- Yes
- No
- Thank You
- Help

## Features
- Upload an image for prediction.
- Use a webcam for real-time gesture prediction.
- Display the predicted sign in a graphical interface.
- Train a machine learning model using collected images.

## Technologies Used
- Python
- OpenCV
- NumPy
- Scikit-learn
- Tkinter
- Pillow

## Project Files
- collect_data.py - Collects gesture images.
- train_model.py - Trains the machine learning model.
- sign_language_gui.py - Runs the graphical interface.
- sign_model.pkl - Saved trained model.
- sign_labels.pkl - Saved gesture labels.
- dataset/ - Contains gesture images.

## Installation
Install the required libraries:

    python3 -m pip install opencv-python numpy scikit-learn pillow

## Run the Application
Start the application:

    python3 sign_language_gui.py

To collect training images:

    python3 collect_data.py

To train the model:

    python3 train_model.py

## Limitations
Prediction accuracy depends on the training images, lighting,
hand position, and background. The system is a basic demonstration
and may incorrectly classify some gestures.

## Operating Hours
The intended operating hours are 6:00 PM to 10:00 PM.
