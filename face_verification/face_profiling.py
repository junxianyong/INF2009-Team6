import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

class FaceProfiler:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.target_size = (112, 112)
        self.face_required_size = (512, 512)
        self.padding = 0.2
        self._interpreter = None
        self.model_path = "mobilefacenet.tflite"
        # Add OpenCV face detector
        self.opencv_face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def get_tflite_interpreter(self):
        if self._interpreter is None:
            print("Loading MobileFaceNet TFLite model...")
            self._interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self._interpreter.allocate_tensors()
            print("Model loaded successfully.")
        return self._interpreter

    def detect_face_mediapipe(self, frame):
        """MediaPipe face detection implementation"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.7,
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

    def detect_face_opencv(self, frame):
        """OpenCV face detection implementation"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.opencv_face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        if len(faces) > 0:
            x, y, width, height = faces[0]
            return (x, y, width, height)
        return None

    def extract_face(self, frame, face_location):
        x, y, width, height = face_location
        h, w, _ = frame.shape
        
        # Add padding
        padding_x = int(width * self.padding)
        padding_y = int(height * self.padding)
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(w, x + width + padding_x)
        y2 = min(h, y + height + padding_y)
        
        # Extract face with padding
        face = frame[y1:y2, x1:x2]
                
        return face

    def preprocess_face(self, face_img):
        if face_img.shape[:2] != self.target_size:
            face_img = cv2.resize(face_img, self.target_size)
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype("float32")
        face_normalized = (face_normalized - 127.5) / 128.0
        face_normalized = np.expand_dims(face_normalized, axis=0)
        return face_normalized

    def get_face_embedding(self, face_img):
        interpreter = self.get_tflite_interpreter()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], face_img)
        interpreter.invoke()
        embedding = interpreter.get_tensor(output_details[0]["index"])
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.flatten()

def profile_pipeline(profiler, image_path, detector='mediapipe'):
    """Profile each step of the face processing pipeline."""
    timings = {}
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Face detection timing
    start_time = time.time()
    if detector == 'mediapipe':
        face_location = profiler.detect_face_mediapipe(frame)
    else:
        face_location = profiler.detect_face_opencv(frame)
    timings['detection'] = time.time() - start_time
    
    if not face_location:
        print(f"No face detected in image using {detector}")
        return None
    
    # Face extraction timing
    start_time = time.time()
    face = profiler.extract_face(frame, face_location)
    timings['extraction'] = time.time() - start_time
    
    # Preprocessing timing
    start_time = time.time()
    preprocessed_face = profiler.preprocess_face(face)
    timings['preprocessing'] = time.time() - start_time
    
    # Embedding generation timing
    start_time = time.time()
    embedding = profiler.get_face_embedding(preprocessed_face)
    timings['embedding'] = time.time() - start_time
    
    return timings

def main():
    print("\nInitializing face profiler...")
    start_time = time.time()
    profiler = FaceProfiler()
    interpreter = profiler.get_tflite_interpreter()
    model_load_time = time.time() - start_time
    print(f"Model loading time: {model_load_time:.4f} seconds\n")

    test_images = [
        "saved_faces/billgates.jpg",
        "saved_faces/justinbieber.jpg",
        "saved_faces/selenagomez.jpg",
        "saved_faces/shawnmendes.jpg",
        "saved_faces/taylor.jpg",
    ]
    
    detectors = ['mediapipe', 'opencv']
    
    for detector in detectors:
        print(f"\n=== Testing {detector.upper()} Detector ===")
        
        # Initial run
        print(f"\nPerforming initial {detector} run...")
        initial_timings = profile_pipeline(profiler, test_images[0], detector)
        if initial_timings:
            print(f"\nInitial {detector.upper()} Results:")
            print("-" * 40)
            for step, time_taken in initial_timings.items():
                print(f"{step.replace('_', ' ').title():20}: {time_taken:.4f} seconds")
        
        # Multiple runs for average
        print(f"\nPerforming multiple {detector} runs...")
        all_timings = []
        for i, image_path in enumerate(test_images[1:], 1):
            print(f"\nProcessing image {i}/4...")
            timings = profile_pipeline(profiler, test_images[i], detector)
            if timings:
                all_timings.append(timings)
        
        if all_timings:
            avg_timings = {
                key: np.mean([t[key] for t in all_timings]) 
                for key in all_timings[0].keys()
            }
            std_timings = {
                key: np.std([t[key] for t in all_timings]) 
                for key in all_timings[0].keys()
            }
            
            print(f"\n{detector.upper()} Average Results (over 4 runs):")
            print("-" * 60)
            total_avg = sum(avg_timings.values())
            for step in avg_timings.keys():
                avg = avg_timings[step]
                std = std_timings[step]
                print(f"{step.replace('_', ' ').title():20}: {avg:.4f} ± {std:.4f} seconds")
            print("-" * 60)
            print(f"Total Pipeline Time:    {total_avg:.4f} seconds")

if __name__ == "__main__":
    main()