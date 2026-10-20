# Project 5 – Voice Detection

## Overview
A Python GUI application that analyses an audio recording and estimates voice category, age, and emotion for predicted senior-citizen cases.

## Features
- Upload an audio recording.
- Record and analyse voice.
- Estimate voice category and age using machine-learning models.
- Predict emotion when the estimated age is above 60.
- Display an analysis report through a graphical interface.

## Technologies
- Python
- Tkinter
- Librosa
- NumPy
- Scikit-learn
- Joblib

## Model Training
The models were trained using audio samples from the CREMA-D dataset and actor demographic labels.

Download the dataset from the [CREMA-D repository](https://github.com/CheyneyComputerScience/CREMA-D). Run `train_voice_model.py` to train the models.

## How to Run
Install the required dependencies, ensure the trained model files are available in the `models` folder, and run:

```bash
python3 voice_detection_gui.py
```

## Limitations
Predictions are experimental estimates and may be inaccurate. Voice alone cannot reliably establish a person's actual age, sex, or emotion.
