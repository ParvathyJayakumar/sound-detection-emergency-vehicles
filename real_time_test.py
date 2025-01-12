import pyaudio
import numpy as np
import librosa
import pickle

# Load the trained SVM model
with open("svm_model.pkl", "rb") as file:
    svm_model = pickle.load(file)

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = RATE * 1

def extract_features_from_audio(audio_data, sr):
    """Extract MFCC features from audio data."""
    mfcc = librosa.feature.mfcc(y=audio_data.astype(np.float32), sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

def process_real_time_audio(model):
    """Process real-time audio and classify using the SVM model."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Listening for audio streams...")
    try:
        while True:
            # Capture audio data
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            # Extract features
            features = extract_features_from_audio(audio_data, RATE)
            # Predict
            prediction = model.predict(features)
            if prediction[0] == 0:
                print("Prediction: Ambulance Siren Detected")
            else:
                print("Prediction: Traffic Noise")
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

# Start real-time testing
process_real_time_audio(svm_model)
