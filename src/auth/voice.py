import hashlib as hash
import logging
import os
import pickle
import time
import wave

import librosa
import numpy as np
import pyaudio
import speech_recognition as sr
from scipy.spatial.distance import cosine

from utils.logger_mixin import LoggerMixin


class VoiceAuth(LoggerMixin):
    """
    VoiceAuth class provides functionality for user enrollment and authentication
    using voice recognition techniques.

    This class leverages audio processing and feature extraction to facilitate
    a secure voice-based biometric system. It allows users to enroll their
    voiceprints and performs authentication by comparing test audio samples with
    enrolled voiceprints. The class ensures authentication by validating both
    acoustic features and spoken words. It also handles storing and retrieving
    voiceprints in a centralized file for persistent authentication mechanisms.

    :ivar _voiceprints_file: Path to the file where voiceprints are stored.
    :type _voiceprints_file: str
    :ivar _sr_rate: Sample rate for audio processing.
    :type _sr_rate: int
    :ivar _num_mfcc: Number of Mel Frequency Cepstral Coefficients for feature
        extraction.
    :type _num_mfcc: int
    :ivar _linear_threshold: Threshold value to evaluate the linear distance
        between feature vectors during authentication.
    :type _linear_threshold: float
    :ivar _cos_threshold: Threshold value for cosine similarity during
        authentication.
    :type _cos_threshold: float
    :ivar _logger: Logger instance for logging class activities and events.
    :type _logger: logging.Logger
    """

    def __init__(
            self,
            voice_auth_config,
            logging_level=logging.INFO,
            recognizer=sr.Recognizer(),
    ):
        """
        Represents the initialization of a voice authentication system that configures
        parameters related to audio processing, such as sample rate, MFCC features, and
        similarity thresholds, along with setting up logging mechanisms.

        :param voice_auth_config: Configuration dictionary containing parameters for voice
            authentication. It includes file details, sampling rate, MFCC count,
            and similarity thresholds.
        :type voice_auth_config: dict

        :param logging_level: Logging level for recording activities. Defaults to
            logging.INFO if not explicitly specified.
        :type logging_level: int
        """
        self._voiceprints_file = voice_auth_config["voiceprints_file"]
        self._sr_rate = voice_auth_config["sr_rate"]
        self._num_mfcc = voice_auth_config["num_mfcc"]
        self._linear_threshold = voice_auth_config["linear_threshold"]
        self._cos_threshold = voice_auth_config["cos_threshold"]
        self._logger = self.setup_logger(__name__, logging_level)
        self._recognizer = recognizer

    def _adjust_for_ambient_noise(self, source):
        """
        Adjusts the recognizer's energy threshold based on the surrounding noise level.

        This method listens for the specified audio source and adjusts the recognizer's energy
        threshold based on the surrounding noise level. It helps in reducing false positives and
        improving the accuracy of speech recognition.
        """

        self._recognizer.adjust_for_ambient_noise(self._recognizer.listen(source))

    def _extract_features(self, filename) -> np.ndarray:
        """
        Extracts a fixed-size feature vector from the provided audio file using
        MFCC (Mel-Frequency Cepstral Coefficients) technique. Loads the audio file
        and applies pre-emphasis filtering prior to extracting the MFCC features.
        The result is averaged to ensure a consistent fixed-size vector output.

        :param filename: Path to the input audio file.
        :type filename: str
        :return: A 1D numpy array representing the fixed-size MFCC feature vector.
        :rtype: np.ndarray
        """
        y, sr = librosa.load(filename, sr=self._sr_rate)  # Load the audio file
        y = librosa.effects.preemphasis(y)  # Pre-emphasis filter
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self._num_mfcc
        )  # Extract MFCC features
        return np.mean(mfccs.T, axis=0)  # Average to get a fixed-size feature vector

    def _extract_words(self, filename):
        """
        Extracts words from an audio file using speech recognition.

        This method processes the specified audio file using the `speech_recognition`
        library and performs speech-to-text conversion. If the recognition is successful,
        the recognized text is returned. In case of an error during recognition or a
        failure to understand the audio content, appropriate error messages are logged
        through the internal logger, and an empty string is returned.

        :param filename: The path to the audio file to be processed.
        :type filename: str

        :return: The recognized text from the audio file. If recognition fails,
                 an empty string is returned.
        :rtype: str
        """
        
        with sr.AudioFile(filename) as source:
            audio = self._recognizer.record(source)
        try:
            rec = self._recognizer.recognize_google(audio)
            self._logger.debug(
                "The audio file contains: " + rec
            )  # Changed from info to debug
            return rec
        except sr.UnknownValueError:
            self._logger.error("Could not understand the audio")
            return ""
        except sr.RequestError as e:
            self._logger.error("Could not request results: %s", e)
            return ""

    def _hash_string(self, string: str) -> str:
        """
        Hashes a given string using the SHA-256 hashing algorithm and returns the
        result as a hexadecimal string.

        :param string: A string to be hashed.
        :type string: str

        :return: A hexadecimal representation of the SHA-256 hash of the input string.
        :rtype: str
        """
        return hash.sha256(string.encode()).hexdigest()

    def _load_voiceprints(self):
        """
        Loads voiceprint data from a specified file.

        This method attempts to open a file in binary read mode and loads
        voiceprint data using the pickle module. If the file does not exist or
        is empty, an empty dictionary is returned.

        :raises FileNotFoundError: raised if the specified file cannot be found.
        :raises EOFError: raised if the end of the file is encountered while
            attempting to load data.

        :return: A dictionary containing the loaded voiceprint data.
        :rtype: dict
        """
        try:
            with open(self._voiceprints_file, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {}  # Return empty dict if file doesn't exist or is empty

    def _save_voiceprints(self, voiceprints):
        """
        Saves the provided voiceprints dictionary to a specified file using pickle serialization.

        This method serializes the `voiceprints` dictionary and writes it to the file specified
        by the `_voiceprints_file` attribute. It ensures that the voiceprints are stored in a
        binary format for later retrieval. The file will be overwritten if it already exists.

        :param voiceprints: Dictionary containing the voiceprints to be saved. The keys represent
            unique identifiers for individuals, and the values correspond to their respective
            voiceprint data.
        :return: None
        """
        with open(self._voiceprints_file, "wb") as f:
            pickle.dump(voiceprints, f)

    def enroll_user(self, user_name, wav_files=[]):
        """
        Enrolls a user by processing and storing the given audio files. The method extracts
        audio features and hashed representations of spoken words from the provided list of
        audio files. All provided files must contain the same spoken word to complete the
        enrollment successfully. After processing, the extracted features and word hashes are
        stored for the given user.

        :param user_name: The name of the user being enrolled.
        :type user_name: str
        :param wav_files: A list of paths to WAV audio files to be used for enrollment.
        :type wav_files: list[str]
        :return: Returns True if enrollment is successful, otherwise False.
        :rtype: bool
        """
        if len(wav_files) < 1:
            self._logger.error("No audio file provided!")
            return False

        enrollment_data = []
        for wav in wav_files:
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
        """
        Authenticates a user by comparing their voice input with saved voiceprint data.

        This method captures audio input, extracts features and hashed words from it,
        and compares it against pre-enrolled voiceprint data for the specified user.
        The authentication process analyzes the audio by computing Euclidean distances,
        cosine similarities, and verifying the hashed words. Results are validated
        against defined thresholds to determine if the user's voice matches.

        :param user_name: The name of the user whose voiceprint is being authenticated.
        :type user_name: str

        :return: Returns True if the authentication is successful, otherwise False.
        :rtype: bool
        """
        test_filename = "test_sample.wav"
        self._record_audio(test_filename)
        test_features = self._extract_features(test_filename)
        test_hashed_words = self._hash_string(self._extract_words(test_filename))

        # Load voiceprints from central file
        voiceprints = self._load_voiceprints()
        # Show all enrolled users
        self._logger.debug("Enrolled users:")  # Changed from info to debug
        for user in voiceprints:
            self._logger.debug(user)  # Changed from info to debug
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

        self._logger.debug(f"Distance: {avg_distance}")  # Changed from info to debug
        self._logger.debug(
            f"Similarity: {avg_similarity}"
        )  # Changed from info to debug

        os.remove(test_filename)

        if (
                avg_distance < self._linear_threshold
                and avg_similarity > self._cos_threshold
                and any(hash_matches)
        ):
            self._logger.info(
                "Authentication successful!"
            )  # Kept as info - important result
            return True
        else:
            self._logger.error("Authentication failed.")
            return False

    def _record_audio(self, filename, duration=3, rate=44100, chunk=1024, channels=1):
        """
        Records audio using the PyAudio library and saves it to a specified file.

        This method captures audio input from the device's microphone. The recording
        duration, audio format, sampling rate, buffer size, and channel settings can
        be customized using the provided parameters. The audio is then saved as a
        .wav file at the specified location. Debug-level logs provide insights into
        when the recording starts and stops.

        :param filename: The path to the output file where the recorded audio will
                         be saved.
        :type filename: str
        :param duration: The duration of the audio recording in seconds. Defaults to 3.
        :type duration: int, optional
        :param rate: The sampling rate for the audio recording in Hz. Defaults to 44100.
        :type rate: int, optional
        :param chunk: The size of each audio chunk for buffering. Defaults to 1024.
        :type chunk: int, optional
        :param channels: The number of audio channels to record. Defaults to 1.
        :type channels: int, optional
        :return: None
        """
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            frames_per_buffer=chunk,
        )

        self._logger.debug("Recording...")  # Changed from info to debug
        frames = []

        for _ in range(0, int(rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)

        self._logger.debug("Recording finished.")  # Changed from info to debug

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

    voice_auth_config = {
        "voiceprints_file": "voiceprints.pkl",
        "sr_rate": 44100,
        "num_mfcc": 20,
        "linear_threshold": 100,
        "cos_threshold": 0.95,
    }

    # Create an instance of VoiceAuth
    va = VoiceAuth(voice_auth_config)

    print(f"Enrolling {USER1_TO_ENROLL} now")  # Changed from info to debug
    va.enroll_user(USER1_TO_ENROLL, ["ck_sample0.wav", "ck_sample1.wav"])

    print(f"Enrolling {USER2_TO_ENROLL} now")  # Changed from info to debug
    va.enroll_user(USER2_TO_ENROLL, ["sample0.wav"])

    time.sleep(2)
    print(f"Authenticating {USER_TO_AUTH} now")  # Changed from info to debug
    va.authenticate_user(USER_TO_AUTH)
