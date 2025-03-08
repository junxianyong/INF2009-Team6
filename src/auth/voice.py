import pickle
from scipy.spatial.distance import cosine
import os
import numpy as np
import time
import librosa
import hashlib as hash
import speech_recognition as sr
import pyaudio
import wave
from utils.logger_mixin import LoggerMixin
import logging


class VoiceAuth(LoggerMixin):

    def __init__(
        self,
        voice_auth_config,
        logging_level=logging.INFO,
    ):
        self._voiceprints_file = voice_auth_config["voiceprints_file"]
        self._sr_rate = voice_auth_config["sr_rate"]
        self._num_mfcc = voice_auth_config["num_mfcc"]
        self._linear_threshold = voice_auth_config["linear_threshold"]
        self._cos_threshold = voice_auth_config["cos_threshold"]
        self._logger = self._setup_logger(__name__, logging_level)

    def _extract_features(self, filename) -> np.ndarray:
        """Extract MFCC features from an audio file."""
        y, sr = librosa.load(filename, sr=self._sr_rate)  # Load the audio file
        y = librosa.effects.preemphasis(y)  # Pre-emphasis filter
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self._num_mfcc
        )  # Extract MFCC features
        return np.mean(mfccs.T, axis=0)  # Average to get a fixed-size feature vector

    def _extract_words(self, filename):
        """Extract spoken words from an audio file using speech recognition."""
        recognizer = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)
        try:
            rec = recognizer.recognize_google(audio)
            self._logger.info("The audio file contains: " + rec)
            return rec
        except sr.UnknownValueError:
            self._logger.error("Could not understand the audio")
            return ""
        except sr.RequestError as e:
            self._logger.error("Could not request results: %s", e)
            return ""

    def _hash_string(self, string: str) -> str:
        """Return a SHA-256 hash of the provided string."""
        return hash.sha256(string.encode()).hexdigest()

    def _load_voiceprints(self):
        """Load all voiceprints from the central file."""
        try:
            with open(self._voiceprints_file, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {}  # Return empty dict if file doesn't exist or is empty

    def _save_voiceprints(self, voiceprints):
        """Save all voiceprints to the central file."""
        with open(self._voiceprints_file, "wb") as f:
            pickle.dump(voiceprints, f)

    def enroll_user(self, user_name, wav_file=[]):
        """Enroll a new user with multiple samples for consistency."""
        if len(wav_file) < 1:
            self._logger.error("No audio file provided!")
            return False

        enrollment_data = []
        for wav in wav_file:
            features = self._extract_features(wav)
            words = self._extract_words(wav)
            hashed_words = self._hash_string(words)
            enrollment_data.append({"features": features, "hashed_words": hashed_words})

        # Ensure all enrolled samples contain the same spoken word
        if len({sample["hashed_words"] for sample in enrollment_data}) != 1:
            self._logger.error(
                "Enrollment failed. Please try again with the same word."
            )
            return False

        # Load existing voiceprints and update the current user's data
        voiceprints = self._load_voiceprints()
        voiceprints[user_name] = enrollment_data
        self._save_voiceprints(voiceprints)

        self._logger.info(f"Enrollment complete for {user_name}.")
        return True

    def authenticate_user(self, user_name):
        """Authenticate an enrolled user based on a test audio sample."""
        test_filename = "test_sample.wav"
        self._record_audio(test_filename)
        test_features = self._extract_features(test_filename)
        test_hashed_words = self._hash_string(self._extract_words(test_filename))

        # Load voiceprints from central file
        voiceprints = self._load_voiceprints()
        # Show all enrolled users
        self._logger.info("Enrolled users:")
        for user in voiceprints:
            self._logger.info(user)
        if user_name not in voiceprints:
            self._logger.error("User not found!")
            os.remove(test_filename)
            return False

        enrolled_data = voiceprints[user_name]
        distances = [
            np.linalg.norm(test_features - sample["features"])
            for sample in enrolled_data
        ]
        similarities = [
            1 - cosine(test_features, sample["features"]) for sample in enrolled_data
        ]

        avg_distance = np.mean(distances)
        avg_similarity = np.mean(similarities)

        # Check if the test hashed word matches any enrolled sample's hash.
        hash_matches = [
            test_hashed_words == sample["hashed_words"] for sample in enrolled_data
        ]

        self._logger.info(f"Distance: {avg_distance}")
        self._logger.info(f"Similarity: {avg_similarity}")

        os.remove(test_filename)

        if (
            avg_distance < self._linear_threshold
            and avg_similarity > self._cos_threshold
            and any(hash_matches)
        ):
            self._logger.info("Authentication successful!")
            return True
        else:
            self._logger.error("Authentication failed.")
            return False

    def _record_audio(self, filename, duration=3, rate=44100, chunk=1024, channels=1):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
        )

        self._logger.info("Recording...")
        frames = []

        for _ in range(0, int(rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)

        self._logger.info("Recording finished.")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))


if __name__ == "__main__":
    USER1_TO_ENROLL = "User 1"
    USER2_TO_ENROLL = "User 2"
    USER_TO_AUTH = "User 1"

    # Create an instance of VoiceAuth
    va = VoiceAuth(linear_threshold=100)

    va._logger.info(f"Enrolling {USER1_TO_ENROLL} now")
    va.enroll_user(USER1_TO_ENROLL, ["ck_sample0.wav", "ck_sample1.wav"])

    va._logger.info(f"Enrolling {USER2_TO_ENROLL} now")
    va.enroll_user(USER2_TO_ENROLL, ["sample0.wav"])

    time.sleep(2)
    va._logger.info(f"Authenticating {USER_TO_AUTH} now")
    va.authenticate_user(USER_TO_AUTH)
