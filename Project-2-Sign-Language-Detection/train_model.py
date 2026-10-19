
import os
import cv2
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = "dataset"
MODEL_PATH = "sign_model.pkl"
LABELS_PATH = "sign_labels.pkl"
IMAGE_SIZE = (64, 64)

X = []
y = []

if not os.path.isdir(DATASET_PATH):
    print("Dataset folder not found.")
    raise SystemExit

labels = sorted([
    name for name in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, name))
])

for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    count = 0

    for filename in os.listdir(folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder, filename)
        image = cv2.imread(path)

        if image is None:
            continue

        image = cv2.resize(image, IMAGE_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = image.flatten().astype(np.float32) / 255.0

        X.append(features)
        y.append(label)
        count += 1

    print(f"{label}: {count} images")

if len(X) < 10 or len(set(y)) < 2:
    print("Not enough images. Collect images for each sign first.")
    raise SystemExit

X = np.array(X)
y = np.array(y)

# Keep the test split stratified when every class has enough images.
unique, counts = np.unique(y, return_counts=True)

if min(counts) < 2:
    print("Collect at least 2 images for every sign.")
    raise SystemExit

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", probability=True, class_weight="balanced")
)

print("\nTraining model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nTest split accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

with open(LABELS_PATH, "wb") as file:
    pickle.dump(labels, file)

print("\nModel saved as:", MODEL_PATH)
print("Labels saved as:", LABELS_PATH)
print("Training completed.")
