**Sound Detection for Emergency Vehicles - SVM Model**

This repository contains a machine learning model to detect the sounds of emergency vehicles (ambulances) from traffic noise using Support Vector Machine (SVM) classification. The dataset is structured into training and testing sets, with separate folders for ambulance sounds and traffic noise sounds.This project uses Support Vector Machines (SVM), a powerful supervised learning algorithm, to classify audio clips as either ambulance or traffic noise. 

The process consists of the following steps:

**1.Feature Extraction:**

Audio features, such as Mel-Frequency Cepstral Coefficients (MFCCs), are extracted from the audio files. These features capture the characteristics of the sound, which are crucial for the classification task.

**2.Training:**

The extracted features from the training audio files are used to train the SVM classifier. SVM works by finding a hyperplane that best separates the two classes (ambulance vs. traffic noise).

**3.Testing:**

After the model is trained, it is tested on unseen test data (ambulance and traffic noise clips) to evaluate its performance in classifying real-world sound samples.

**Model Performance**

After training the SVM model on the dataset, the following performance metrics were evaluated using the test set of ambulance and traffic noise audio files:

**Accuracy: 95%**

**Precision: 0.91**

**Recall: 0.93**

**F1-Score: 0.92**
