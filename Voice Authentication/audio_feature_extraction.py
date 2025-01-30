import librosa
import numpy as np

def extract_features(filename):
    y, sr = librosa.load(filename, sr=44100)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)  # Take mean to get a fixed-size feature vector