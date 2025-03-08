import logging
import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
from scipy.spatial.distance import cosine
import time
import mediapipe as mp
from utils.logger_mixin import LoggerMixin


class FaceVerification(LoggerMixin):
    def __init__(
        self,
        face_verification_config,
        logging_level=logging.INFO,
    ):
        """
        Initialize the FaceVerification class with configurable parameters.

        Args:
            face_verification_config (dict): Dictionary of configuration parameters.
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        self.model_path = face_verification_config["model_path"]
        self.database_path = face_verification_config["database_path"]

        # Set configurable properties as instance attributes.
        self.model_selection = face_verification_config["model_selection"]
        self.min_detection_confidence = face_verification_config[
            "min_detection_confidence"
        ]
        self.padding = face_verification_config["padding"]
        self.face_required_size = face_verification_config["face_required_size"]
        self.target_size = face_verification_config["target_size"]
        self.verification_threshold = face_verification_config["verification_threshold"]
        self.verification_timeout = face_verification_config["verification_timeout"]
        self.verification_max_attempts = face_verification_config[
            "verification_max_attempts"
        ]
        self.camera_id = face_verification_config["camera_id"]
        self.logger = self._setup_logger(__name__, logging_level)
        self._interpreter = None

    def get_tflite_interpreter(self):
        """
        Load the MobileFaceNet TFLite model (lazily initialized).

        Returns:
            TFLite interpreter instance.
        """
        if self._interpreter is None:
            self.logger.info("Loading MobileFaceNet TFLite model...")
            self._interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self._interpreter.allocate_tensors()
            self.logger.info("Model loaded successfully.")
        return self._interpreter

    def detect_face(self, frame):
        """
        Detect faces in a frame using MediaPipe.

        Args:
            frame (ndarray): Input image frame (BGR format).

        Returns:
            Tuple (x, y, width, height) of detected face or None if no face detected.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self.mp_face_detection.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.min_detection_confidence,
        ) as face_detection:
            results = face_detection.process(frame_rgb)

            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box

                h, w, _ = frame.shape
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = min(int(bbox.width * w), w - x)
                height = min(int(bbox.height * h), h - y)

                return (x, y, width, height)
        return None

    def extract_face(self, frame, face_location):
        """
        Extract a face from the frame using padding and resize it.

        Args:
            frame (ndarray): Input image frame.
            face_location (tuple): (x, y, width, height) of detected face.

        Returns:
            Extracted face image resized to the required size.
        """
        x, y, width, height = face_location
        h, w, _ = frame.shape

        padding_x = int(width * self.padding)
        padding_y = int(height * self.padding)

        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(w, x + width + padding_x)
        y2 = min(h, y + height + padding_y)

        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, self.face_required_size)

        return face

    def preprocess_face(self, face_img):
        """
        Preprocess the face image for model input.

        Args:
            face_img (ndarray): Face image in BGR format.

        Returns:
            Preprocessed face image suitable for model input.
        """
        if face_img.shape[:2] != self.target_size:
            face_img = cv2.resize(face_img, self.target_size)

        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype("float32")
        face_normalized = (face_normalized - 127.5) / 128.0
        face_normalized = np.expand_dims(face_normalized, axis=0)

        return face_normalized

    def get_face_embedding(self, face_img):
        """
        Get embedding from a face image.

        Args:
            face_img (ndarray): Preprocessed face image.

        Returns:
            Face embedding vector (L2 normalized).
        """
        interpreter = self.get_tflite_interpreter()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], face_img)
        interpreter.invoke()
        embedding = interpreter.get_tensor(output_details[0]["index"])

        embedding = embedding / np.linalg.norm(embedding)
        return embedding.flatten()

    def save_face_embeddings(self, embeddings_dict):
        """
        Save face embeddings to a database file.

        Args:
            embeddings_dict (dict): Dictionary of {name: embedding_vector}.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            existing_embeddings = {}
            if os.path.exists(self.database_path):
                with open(self.database_path, "rb") as f:
                    existing_embeddings = pickle.load(f)

            existing_embeddings.update(embeddings_dict)

            with open(self.database_path, "wb") as f:
                pickle.dump(existing_embeddings, f)

            self.logger.info(
                f"Saved {len(embeddings_dict)} embeddings to {self.database_path}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {str(e)}")
            return False

    def load_face_embeddings(self):
        """
        Load face embeddings from the database file.

        Returns:
            Dictionary of {name: embedding_vector} or empty dict if file not found.
        """
        if not os.path.exists(self.database_path):
            self.logger.warning(f"Embedding database not found: {self.database_path}")
            return {}

        try:
            with open(self.database_path, "rb") as f:
                embeddings_db = pickle.load(f)
            return embeddings_db
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {str(e)}")
            return {}

    def build_embedding_from_image(self, img_path, person_name):
        """
        Build an embedding from a single image and save it to the database.

        Args:
            img_path (str): Path to the input image.
            person_name (str): Name of the person in the image.

        Returns:
            True if embedding was successfully built and saved, False otherwise.
        """
        interpreter = self.get_tflite_interpreter()

        try:
            self.logger.info(f"Processing {img_path}...")
            img = cv2.imread(img_path)
            if img is None:
                self.logger.error(f"Could not load image {img_path}")
                return False

            face_location = self.detect_face(img)
            if not face_location:
                self.logger.warning(f"No face detected in {img_path}")
                return False

            face = self.extract_face(img, face_location)
            preprocessed_face = self.preprocess_face(face)
            embedding = self.get_face_embedding(preprocessed_face)
            embeddings_dict = {person_name: embedding}

            self.save_face_embeddings(embeddings_dict)
            self.logger.info(f"Successfully processed {person_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing {img_path}: {str(e)}")
            return False

    def verify_face(self, frame=None):
        """
        Verify a face against the database.

        Args:
            frame (ndarray): Input image frame (if None, capture from the default camera).

        Returns:
            Tuple (name, similarity) if recognized, or None.
        """
        embeddings_db = self.load_face_embeddings()
        if not embeddings_db:
            self.logger.warning("No faces in database to compare against")
            return None

        need_release = False
        if frame is None:
            self.logger.info(f"Initializing camera {self.camera_id}...")
            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                self.logger.error(f"Error: Could not open camera {self.camera_id}")
                return None

            ret, frame = cap.read()
            if not ret:
                self.logger.error("Error capturing frame")
                cap.release()
                return None
            need_release = True

        try:
            face_location = self.detect_face(frame)
            if not face_location:
                self.logger.debug("No face detected in frame")
                if need_release:
                    cap.release()
                return None

            face = self.extract_face(frame, face_location)
            preprocessed_face = self.preprocess_face(face)
            query_embedding = self.get_face_embedding(preprocessed_face)

            best_match = None
            best_similarity = -1

            for name, stored_embedding in embeddings_db.items():
                try:
                    similarity = 1 - cosine(query_embedding, stored_embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name
                except Exception as e:
                    self.logger.error(f"Error comparing with {name}: {str(e)}")
                    continue

            if best_similarity >= self.verification_threshold:
                self.logger.info(
                    f"Face recognized as {best_match} with similarity {best_similarity:.4f}"
                )
                result = (best_match, best_similarity)
            else:
                self.logger.info(
                    f"No match found (best similarity: {best_similarity:.4f})"
                )
                result = None

        except Exception as e:
            self.logger.error(f"Error during face verification: {str(e)}")
            result = None

        if need_release:
            cap.release()

        return result

    def wait_for_face_and_verify(self):
        """
        Wait for a face to appear, then verify it with multiple attempts.

        Returns:
            Tuple (name, similarity) if recognized, or None.
        """
        self.logger.info(f"Initializing camera {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            self.logger.error(f"Error: Could not open camera {self.camera_id}")
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.logger.info("Waiting for face...")
        start_time = time.time()
        attempts = 0

        try:
            while (
                time.time() - start_time < self.verification_timeout
                and attempts < self.verification_max_attempts
            ):
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Error capturing frame")
                    continue

                face_location = self.detect_face(frame)
                if face_location:
                    attempts += 1
                    self.logger.info(
                        f"Face detected! Verification attempt {attempts}/{self.verification_max_attempts}..."
                    )
                    result = self.verify_face(frame)
                    if result:
                        self.logger.info("Verification successful!")
                        self.logger.info(
                            f"Face recognized as {result[0]} with similarity {result[1]:.4f}"
                        )
                        return result[0]
                    elif attempts == self.verification_max_attempts:
                        self.logger.warning("Max attempts reached.")
                        return None
                    else:
                        self.logger.info("Verification failed. Trying again...")
                        time.sleep(0.5)
                time.sleep(0.1)
        finally:
            cap.release()

        if attempts == 0:
            self.logger.warning("No face detected within the timeout period")
        else:
            self.logger.warning(f"Failed to verify face after {attempts} attempts")

        return None


if __name__ == "__main__":
    face_verification_config = {
        "model_path": "mobilefacenet.tflite",
        "database_path": "face_embeddings.pkl",
        # Face detection & preprocessing settings:
        "model_selection": 0,
        "min_detection_confidence": 0.7,
        "padding": 0.2,
        "face_required_size": (512, 512),
        "target_size": (112, 112),
        # Verification settings:
        "verification_threshold": 0.7,
        "verification_timeout": 30,
        "verification_max_attempts": 3,
        # Camera settings:
        "camera_id": 0,
    }

    face_verification = FaceVerification(face_verification_config)

    def detect_and_capture_face(face_verifier, output_folder, filename=None):
        """
        Capture an image using the webcam and save it.

        Args:
            face_verifier (FaceVerification): Instance of FaceVerification class.
            output_folder (str): Folder to save the captured image.
            filename (str): Name for the saved file (without extension).

        Returns:
            Path to saved image or None if capture failed.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if not filename:
            filename = f"image_{int(time.time())}"
        output_path = os.path.join(output_folder, f"{filename}.jpg")

        face_verifier.logger.info(f"Initializing camera {face_verifier.camera_id}...")
        cap = cv2.VideoCapture(face_verifier.camera_id)
        if not cap.isOpened():
            face_verifier.logger.error(
                f"Error: Could not open camera {face_verifier.camera_id}"
            )
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        face_verifier.logger.info("Capturing image...")
        # Give a little time for camera to initialize
        time.sleep(1)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            face_verifier.logger.error("Error capturing frame")
            return None

        cv2.imwrite(output_path, frame)
        face_verifier.logger.info(f"Image saved to {output_path}")

        return output_path

    while True:
        if input("Press Enter to capture face or 'q' to quit...").lower() == "q":
            break

        name = input("Enter your name: ")

        face_verification.logger.info("Capturing face image...")
        img_path = detect_and_capture_face(face_verification, "faces", name)
        if img_path:
            face_verification.logger.info(f"Face image saved to {img_path}")
            face_verification.logger.info("Building embedding from image...")
            face_verification.build_embedding_from_image(img_path, name)

    face_verification.logger.info("Running face verification demo...")
    result = face_verification.wait_for_face_and_verify()
    face_verification.logger.info("Face verification result:", result)
