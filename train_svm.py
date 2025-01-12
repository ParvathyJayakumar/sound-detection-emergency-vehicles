import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report


# Function to extract features
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Function to prepare dataset
def prepare_dataset(dataset_path):
    features, labels = [], []
    for label, folder in enumerate(["ambulance", "others"]):  # Assign 0 for ambulance, 1 for others
        folder_path = os.path.join(dataset_path, folder)
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                audio_path = os.path.join(folder_path, file_name)
                features.append(extract_features(audio_path))
                labels.append(label)
    return np.array(features), np.array(labels)

# Define dataset paths
train_path = r"C:\Proj\train"
test_path = r"C:\Proj\test"

# Prepare datasets
X_train, y_train = prepare_dataset(train_path)
X_test, y_test = prepare_dataset(test_path)

# Train the SVM model
svm_model = SVC(kernel="linear", probability=False)
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model (optional)
import pickle
with open("svm_model.pkl", "wb") as file:
    pickle.dump(svm_model, file)
