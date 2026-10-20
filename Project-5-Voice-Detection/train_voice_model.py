
import os
import re
import warnings
import numpy as np
import pandas as pd
import librosa
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# Project and dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "crema-d-data"))
AUDIO_DIR = os.path.join(DATA_DIR, "AudioWAV")
DEMOGRAPHICS = os.path.join(DATA_DIR, "VideoDemographics.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COUNT = 40

def extract_features(path):
    """Extract compact acoustic features from a WAV recording."""
    try:
        audio, sr = librosa.load(path, sr=16000, mono=True, duration=5.0)
        if len(audio) < sr // 2:
            return None

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(audio)
        rms = librosa.feature.rms(y=audio)

        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            [np.mean(spectral_centroid), np.std(spectral_centroid),
             np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms)]
        ])
        return features.astype(np.float32)
    except Exception:
        return None

def main():
    if not os.path.isdir(AUDIO_DIR):
        print("ERROR: Audio folder not found:", AUDIO_DIR)
        print("Keep crema-d-data beside Voice_Detection in the main folder.")
        return

    if not os.path.isfile(DEMOGRAPHICS):
        print("ERROR: VideoDemographics.csv not found.")
        return

    demographics = pd.read_csv(DEMOGRAPHICS)
    demographics["ActorID"] = demographics["ActorID"].astype(int)
    demographics = demographics.set_index("ActorID")

    wav_files = sorted(
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith(".wav")
    )

    print("Audio files found:", len(wav_files))
    print("Extracting audio features. This may take a while...")

    X, sexes, ages, emotions, groups = [], [], [], [], []
    emotion_map = {
        "ANG": "Anger",
        "DIS": "Disgust",
        "FEA": "Fear",
        "HAP": "Happy",
        "NEU": "Neutral",
        "SAD": "Sad",
    }

    for index, filename in enumerate(wav_files, start=1):
        match = re.match(r"(\d+)_.*_([A-Z]{3})_[A-Z]{2}\.wav$", filename)
        if not match:
            continue

        actor_id = int(match.group(1))
        emotion_code = match.group(2)

        if actor_id not in demographics.index:
            continue
        if emotion_code not in emotion_map:
            continue

        row = demographics.loc[actor_id]
        path = os.path.join(AUDIO_DIR, filename)

        # Skip Git LFS pointer files and unavailable recordings
        try:
            with open(path, "rb") as file:
                if not file.read(4) == b"RIFF":
                    continue
        except OSError:
            continue

        features = extract_features(path)
        if features is None:
            continue

        X.append(features)
        sexes.append(str(row["Sex"]))
        ages.append(float(row["Age"]))
        emotions.append(emotion_map[emotion_code])
        groups.append(actor_id)

        if index % 100 == 0:
            print(f"Checked {index}/{len(wav_files)} files; usable clips: {len(X)}")

    if len(X) < 100:
        print("\nNot enough real downloaded recordings to train.")
        print("Download more audio files using Git LFS, then run again.")
        return

    X = np.asarray(X)
    sexes = np.asarray(sexes)
    ages = np.asarray(ages)
    emotions = np.asarray(emotions)
    groups = np.asarray(groups)

    print("\nUsable clips:", len(X))
    print("Unique actors:", len(np.unique(groups)))
    print("Sex labels:", dict(zip(*np.unique(sexes, return_counts=True))))
    print("Age range:", int(ages.min()), "to", int(ages.max()))
    print("Emotion labels:", dict(zip(*np.unique(emotions, return_counts=True))))

    # Split by actor to reduce leakage from the same speaker
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]

    # Sex classification
    sex_model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=42, n_jobs=-1)
    )
    sex_model.fit(X_train, sexes[train_idx])
    sex_predictions = sex_model.predict(X_test)
    print("\nSex classification test accuracy:",
          round(accuracy_score(sexes[test_idx], sex_predictions), 3))

    # Age regression
    age_model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    )
    age_model.fit(X_train, ages[train_idx])
    age_predictions = age_model.predict(X_test)
    print("Age prediction test MAE:",
          round(mean_absolute_error(ages[test_idx], age_predictions), 2),
          "years")

    # Emotion classification
    emotion_model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=42, n_jobs=-1)
    )
    emotion_model.fit(X_train, emotions[train_idx])
    emotion_predictions = emotion_model.predict(X_test)
    print("Emotion classification test accuracy:",
          round(accuracy_score(emotions[test_idx], emotion_predictions), 3))

    joblib.dump(sex_model, os.path.join(MODEL_DIR, "sex_model.pkl"))
    joblib.dump(age_model, os.path.join(MODEL_DIR, "age_model.pkl"))
    joblib.dump(emotion_model, os.path.join(MODEL_DIR, "emotion_model.pkl"))

    print("\nSUCCESS! Models saved in:", MODEL_DIR)
    print("Saved: sex_model.pkl, age_model.pkl, emotion_model.pkl")
    print("\nNote: These are experimental estimates, not reliable")
    print("proof of a person's sex, exact age, or emotional state.")

if __name__ == "__main__":
    main()
