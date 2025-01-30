import pickle
from record_audio import record_audio
from audio_feature_extraction import extract_features
import os
import numpy as np


def enroll_user(user_name, num_samples=3):
    voiceprints = []

    for i in range(num_samples):
        filename = f"{user_name}_sample{i}.wav"
        record_audio(filename)
        features = extract_features(filename)
        voiceprints.append(features)

    # Save user voiceprint
    with open(f"{user_name}_voiceprint.pkl", "wb") as f:
        pickle.dump(voiceprints, f)

    print(f"Enrollment complete for {user_name}.")

    # Delete the wav files
    for i in range(num_samples):
        filename = f"{user_name}_sample{i}.wav"
        os.remove(filename)
        

def authenticate_user(user_name, threshold=30):
    test_filename = "test_sample.wav"
    record_audio(test_filename)

    test_features = extract_features(test_filename)

    try:
        with open(f"{user_name}_voiceprint.pkl", "rb") as f:
            enrolled_voiceprints = pickle.load(f)
    except FileNotFoundError:
        print("User not found!")
        return False

    # Compare test features with stored voiceprints
    distances = [np.linalg.norm(test_features - vp) for vp in enrolled_voiceprints] # Calculate Euclidean distance of test features with each enrolled voiceprint
    avg_distance = np.mean(distances) # Calculate average distance

    print(f"Distance: {avg_distance}")
    os.remove(test_filename)

    if avg_distance < threshold:
        print("Authentication successful!")
        return True
    else:
        print("Authentication failed.")
        return False
    
    
enroll_user("john_doe")
authenticate_user("john_doe")