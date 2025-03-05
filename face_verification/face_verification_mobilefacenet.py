# Description: Face verification system using only TFlite model organized as callable functions

import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
from scipy.spatial.distance import cosine
import time
import mediapipe as mp

# Initialize MediaPipe models
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Global variable to store the TFLite interpreter - initialized lazily
_interpreter = None

def get_tflite_interpreter(model_path='mobilefacenet.tflite'):
    """
    Load the MobileFaceNet TFLite model (lazily initialized)
    
    Args:
        model_path: Path to the TFLite model file
        
    Returns:
        TFLite interpreter instance
    """
    global _interpreter
    if _interpreter is None:
        print("Loading MobileFaceNet TFLite model...")
        _interpreter = tf.lite.Interpreter(model_path=model_path)
        _interpreter.allocate_tensors()
        print("Model loaded successfully.")
    return _interpreter

def detect_face(frame, model_selection=0, min_detection_confidence=0.7):
    """
    Detect faces in a frame using MediaPipe
    
    Args:
        frame: Input image frame (BGR format)
        model_selection: MediaPipe model selection (0=close range, 1=full range)
        min_detection_confidence: Minimum confidence threshold for detection
        
    Returns:
        (x, y, width, height) of detected face or None if no face detected
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    with mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence) as face_detection:
        results = face_detection.process(frame_rgb)
        
        if results.detections:
            # Take the first face detected
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute
            h, w, _ = frame.shape
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            width = min(int(bbox.width * w), w - x)
            height = min(int(bbox.height * h), h - y)
            
            return (x, y, width, height)
    
    return None

def extract_face(frame, face_location, padding=0.2, required_size=(512, 512)):
    """
    Extract a face from frame with padding
    
    Args:
        frame: Input image frame
        face_location: Tuple of (x, y, width, height)
        padding: Percentage of padding to add around face
        required_size: Size to resize the face image
        
    Returns:
        Extracted face image resized to required_size
    """
    x, y, width, height = face_location
    h, w, _ = frame.shape
    
    # Add padding
    padding_x = int(width * padding)
    padding_y = int(height * padding)
    
    # Calculate padded coordinates with bounds checking
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(w, x + width + padding_x)
    y2 = min(h, y + height + padding_y)
    
    # Extract face with padding
    face = frame[y1:y2, x1:x2]
    
    # Resize to required size
    face = cv2.resize(face, required_size)
    
    return face

def preprocess_face(face_img, target_size=(112, 112)):
    """
    Preprocess face image for model input
    
    Args:
        face_img: Face image in BGR format
        target_size: Size for model input
        
    Returns:
        Preprocessed face suitable for model input
    """
    # Resize if needed
    if face_img.shape[:2] != target_size:
        face_img = cv2.resize(face_img, target_size)
    
    # Convert to RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [-1, 1]
    face_normalized = face_rgb.astype('float32')
    face_normalized = (face_normalized - 127.5) / 128.0
    
    # Expand dimensions for model input
    face_normalized = np.expand_dims(face_normalized, axis=0)
    
    return face_normalized

def get_face_embedding(face_img, interpreter=None):
    """
    Get embedding from a face image
    
    Args:
        face_img: Preprocessed face image
        interpreter: TFLite interpreter (or None to use default)
        
    Returns:
        Face embedding vector (L2 normalized)
    """
    if interpreter is None:
        interpreter = get_tflite_interpreter()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], face_img)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    embedding = interpreter.get_tensor(output_details[0]['index'])
    
    # L2 normalization
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.flatten()

def save_face_embeddings(embeddings_dict, database_path="face_embeddings.pkl"):
    """
    Save face embeddings to a database file
    
    Args:
        embeddings_dict: Dictionary of {name: embedding_vector}
        database_path: Path to save the database file
        
    Returns:
        True if saved successfully
    """
    try:
        # Load existing database if it exists
        existing_embeddings = {}
        if os.path.exists(database_path):
            with open(database_path, 'rb') as f:
                existing_embeddings = pickle.load(f)
        
        # Update with new embeddings
        existing_embeddings.update(embeddings_dict)
        
        # Save updated database
        with open(database_path, 'wb') as f:
            pickle.dump(existing_embeddings, f)
            
        print(f"Saved {len(embeddings_dict)} embeddings to {database_path}")
        return True
    except Exception as e:
        print(f"Error saving embeddings: {str(e)}")
        return False

def load_face_embeddings(database_path="face_embeddings.pkl"):
    """
    Load face embeddings from database file
    
    Args:
        database_path: Path to the embeddings database file
        
    Returns:
        Dictionary of {name: embedding_vector} or empty dict if file not found
    """
    if not os.path.exists(database_path):
        print(f"Embedding database not found: {database_path}")
        return {}
    
    try:
        with open(database_path, 'rb') as f:
            embeddings_db = pickle.load(f)
        return embeddings_db
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        return {}

def build_embeddings_from_folder(folder_path, output_db_path="face_embeddings.pkl", 
                                model_path='mobilefacenet.tflite'):
    """
    Build embeddings from images in a folder
    
    Args:
        folder_path: Path to folder with face images
        output_db_path: Path to save the embeddings database
        model_path: Path to TFLite model file
        
    Returns:
        Number of successfully processed images
    """
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} not found")
        return 0
    
    # Ensure the interpreter is loaded
    interpreter = get_tflite_interpreter(model_path)
    
    processed = 0
    errors = 0
    embeddings_dict = {}
    
    print(f"Building embeddings from images in {folder_path}...")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            person_name = os.path.splitext(filename)[0]
            
            try:
                print(f"Processing {filename}...")
                
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Could not load image {img_path}")
                    errors += 1
                    continue
                
                # Detect face
                face_location = detect_face(img)
                if not face_location:
                    print(f"No face detected in {img_path}")
                    errors += 1
                    continue
                
                # Extract and preprocess face
                face = extract_face(img, face_location)
                preprocessed_face = preprocess_face(face)
                
                # Get embedding
                embedding = get_face_embedding(preprocessed_face, interpreter)
                
                # Add to dictionary
                embeddings_dict[person_name] = embedding
                processed += 1
                print(f"Successfully processed {person_name}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                errors += 1
    
    # Save all embeddings at once
    if processed > 0:
        save_face_embeddings(embeddings_dict, output_db_path)
    
    print(f"\nBuilding embeddings complete:")
    print(f"Successfully processed: {processed} images")
    print(f"Errors encountered: {errors} images")
    
    return processed

def detect_and_capture_face(output_folder, filename=None, camera_id=0, 
                          timeout=30):
    """
    Detect a face using webcam and save the full frame when face is detected
    
    Args:
        output_folder: Folder to save the captured image
        filename: Name for the saved file (without extension)
        camera_id: Camera device ID
        timeout: Maximum time to wait in seconds
        
    Returns:
        Path to saved image or None if no face captured
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generate filename if not provided
    if not filename:
        filename = f"face_{int(time.time())}"
    output_path = os.path.join(output_folder, f"{filename}.jpg")
    
    print(f"Initializing camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Looking for face...")
    face_detected = False
    start_time = time.time()
    
    # Initialize face detection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret:
                print("Error capturing frame")
                continue
            
            # Detect face
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)
            
            if results.detections:
                # Save the full frame when face is detected
                cv2.imwrite(output_path, frame)
                face_detected = True
                print(f"Face detected and frame saved to {output_path}")
                break
            
            # Small delay to reduce CPU usage
            time.sleep(0.1)
    
    cap.release()
    
    if face_detected:
        return output_path
    else:
        print("No face detected within the timeout period")
        return None

def verify_face(frame=None, camera_id=0, database_path="face_embeddings.pkl", 
                threshold=0.7, model_path='mobilefacenet.tflite'):
    """
    Verify a face against the database
    
    Args:
        frame: Input image frame (if None, capture from camera)
        camera_id: Camera device ID (used if frame is None)
        database_path: Path to embeddings database
        threshold: Similarity threshold for recognition
        model_path: Path to TFLite model
        
    Returns:
        (name, similarity) tuple if recognized, None otherwise
    """
    # Ensure the interpreter is loaded
    interpreter = get_tflite_interpreter(model_path)
    
    # Load embeddings database
    embeddings_db = load_face_embeddings(database_path)
    if not embeddings_db:
        print("No faces in database to compare against")
        return None
    
    # Capture frame if needed
    need_release = False
    if frame is None:
        print(f"Initializing camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return None
        
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame")
            cap.release()
            return None
            
        need_release = True
    
    try:
        # Detect face
        face_location = detect_face(frame)
        if not face_location:
            print("No face detected in frame")
            if need_release:
                cap.release()
            return None
        
        # Extract and preprocess face
        face = extract_face(frame, face_location)
        preprocessed_face = preprocess_face(face)
        
        # Get embedding
        query_embedding = get_face_embedding(preprocessed_face, interpreter)
        
        # Compare with database
        best_match = None
        best_similarity = -1
        
        for name, stored_embedding in embeddings_db.items():
            try:
                similarity = 1 - cosine(query_embedding, stored_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = name
            except Exception as e:
                print(f"Error comparing with {name}: {str(e)}")
                continue
        
        # Check if the best match passes the threshold
        if best_similarity >= threshold:
            print(f"Face recognized as {best_match} with similarity {best_similarity:.4f}")
            result = (best_match, best_similarity)
        else:
            print(f"No match found (best similarity: {best_similarity:.4f})")
            result = None
            
    except Exception as e:
        print(f"Error during face verification: {str(e)}")
        result = None
    
    # Release camera if we opened it
    if need_release:
        cap.release()
        
    return result

def wait_for_face_and_verify(camera_id=0, database_path="face_embeddings.pkl", 
                           threshold=0.7, timeout=30, max_attempts=3):
    """
    Wait for a face to appear, then verify it against the database with multiple attempts
    
    Args:
        camera_id: Camera device ID
        database_path: Path to embeddings database
        threshold: Similarity threshold for recognition
        timeout: Maximum time to wait in seconds
        max_attempts: Maximum number of verification attempts before giving up
        
    Returns:
        (name, similarity) tuple if recognized, None otherwise
    """
    print(f"Initializing camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Waiting for face...")
    start_time = time.time()
    attempts = 0
    
    try:
        while time.time() - start_time < timeout and attempts < max_attempts:
            ret, frame = cap.read()
            if not ret:
                print("Error capturing frame")
                continue
            
            # Check if a face is in the frame
            face_location = detect_face(frame)
            if face_location:
                attempts += 1
                print(f"Face detected! Verification attempt {attempts}/{max_attempts}...")
                result = verify_face(frame, database_path=database_path, threshold=threshold)
                
                if result:
                    print("Verification successful!")
                    return result
                elif attempts == max_attempts:
                    print("Max attempts reached.")
                    return None
                else:
                    print(f"Verification failed. {'Trying again...' if attempts < max_attempts else 'Max attempts reached.'}")
                    # Small delay before next attempt
                    time.sleep(0.5)
            
            # Small delay to reduce CPU usage
            time.sleep(0.1)
    
    finally:
        cap.release()
    
    if attempts == 0:
        print("No face detected within the timeout period")
    else:
        print(f"Failed to verify face after {attempts} attempts")
    
    return None

# Example usage demo function (not directly called)
def demo():
    """Demo function showing how to use the module"""
    # Build embeddings from a folder of images
    build_embeddings_from_folder("saved_faces")
    
    while True:
        # Wait for a face and verify it
        result = wait_for_face_and_verify()
        if result:
            name, similarity = result
            print(f"Recognized: {name} ({similarity:.4f})")
        else:
            print("Face not recognized")

# Example usage for capturing a face and saving it to a file
def capture_face_demo():
    """Demo function to capture a face and save it to a file"""
    detect_and_capture_face("saved_faces", filename="junxian")

# The module can be imported without running the demo
if __name__ == "__main__":
    # Capture a face and save it to a file called "junxian.jpg"
    capture_face_demo()

    # Run the demo function
    # Which will build embeddings from the folder "saved_faces"
    # And then wait for a face to appear and verify it
    demo()