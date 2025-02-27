# Description: Face verification system using only TFlite model
# Make sure you have the tflite model (generate using convert_to_tflite.py)
# OR download from https://drive.google.com/file/d/1Ry7pJ3tz78orypZg3OqLMtDDIf1u3bUU/view?usp=drive_link

import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
from scipy.spatial.distance import cosine
import time
import mediapipe as mp

def get_tflite_model(model_path='mobilefacenet.tflite'):
    """Load the MobileFaceNet TFLite model"""
    print("Loading MobileFaceNet TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("Model loaded successfully.")
    return interpreter

# Add MediaPipe initialization
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_and_extract_face(img_path, required_size=(224, 224), debug=False):
    """Detect face in image and extract it using MediaPipe"""
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Could not load image")
    
    # Convert to RGB (MediaPipe requires RGB input)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Try both model selections if needed
    detection_params = [
        {'model_selection': 0, 'min_detection_confidence': 0.5},  # Close-range
        {'model_selection': 1, 'min_detection_confidence': 0.5}   # Full-range
    ]
    
    face = None
    for params in detection_params:
        with mp_face_detection.FaceDetection(**params) as face_detection:
            results = face_detection.process(img_rgb)
            
            if results.detections:
                if len(results.detections) > 1:
                    if debug:
                        print(f"Warning: Multiple faces detected ({len(results.detections)}), using the first one")
                
                # Get face bounding box
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                h, w, _ = img.shape
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = min(int(bbox.width * w), w - x)
                height = min(int(bbox.height * h), h - y)
                
                # Add padding (20% on each side)
                padding_x = int(width * 0.2)
                padding_y = int(height * 0.2)
                
                # Calculate padded coordinates with bounds checking
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(w, x + width + padding_x)
                y2 = min(h, y + height + padding_y)
                
                if debug:
                    # Draw bounding box on debug image
                    debug_img = img.copy()
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.imshow("Detected Face", debug_img)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()
                
                # Extract face with padding
                face = img[y1:y2, x1:x2]
                # Resize to required size
                face = cv2.resize(face, required_size)
                break
    
    if face is None:
        raise ValueError("No face detected in the image")
    
    return face

def preprocess_face_image(img_path, target_size=(112, 112)):
    """Preprocess image for MobileFaceNet input"""
    try:
        # Detect and extract face
        face = detect_and_extract_face(img_path, target_size)
        
        # Convert to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [-1, 1]
        face = face.astype('float32')
        face = (face - 127.5) / 128.0
        
        # Expand dimensions
        face = np.expand_dims(face, axis=0)
        
        return face
    except ValueError as e:
        raise ValueError(f"Face preprocessing failed: {str(e)}")

def get_face_embedding(interpreter, img_path):
    """Get embedding from a face image using MobileFaceNet"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Debug info
    print(f"Input tensor shape: {input_details[0]['shape']}")
    print(f"Input tensor type: {input_details[0]['dtype']}")
    
    # Preprocess image to 112x112 (MobileFaceNet standard input size)
    preprocessed_img = preprocess_face_image(img_path)
    print(f"Preprocessed image shape: {preprocessed_img.shape}")
    
    try:
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor and normalize
        embedding = interpreter.get_tensor(output_details[0]['index'])
        print(f"Raw embedding shape: {embedding.shape}")
        
        # L2 normalization
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.flatten()
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise

def clean_database(database_path="face_embeddings.pkl"):
    """Remove the database file to start fresh"""
    if os.path.exists(database_path):
        os.remove(database_path)
        print("Existing database removed. Ready for fresh embeddings.")
    else:
        print("No existing database found.")

def save_embedding(embedding, person_name, database_path="face_embeddings.pkl"):
    """Save a face embedding to the database with the person's name as label"""
    embeddings_db = {}
    if os.path.exists(database_path):
        with open(database_path, 'rb') as f:
            embeddings_db = pickle.load(f)
            # Check if we have mixed embeddings
            for name, stored_embedding in embeddings_db.items():
                if stored_embedding.shape != embedding.shape:
                    print("Warning: Database contains mixed embedding sizes.")
                    print("Cleaning database to ensure consistency...")
                    embeddings_db = {}
                    break
    
    # Add new embedding
    embeddings_db[person_name] = embedding
    
    # Save updated database
    with open(database_path, 'wb') as f:
        pickle.dump(embeddings_db, f)
    
    print(f"Embedding for {person_name} saved to database.")

def compare_face(interpreter, img_path, database_path="face_embeddings.pkl", threshold=0.7):
    """Compare face in image with database (adjusted threshold for MobileFaceNet)"""
    try:
        query_embedding = get_face_embedding(interpreter, img_path)
        print(f"Query embedding shape: {query_embedding.shape}")
    except ValueError as e:
        print(f"Error getting query embedding: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error during embedding generation: {str(e)}")
        return None
    
    if not os.path.exists(database_path):
        print("No database found.")
        return None
    
    try:
        with open(database_path, 'rb') as f:
            embeddings_db = pickle.load(f)
            
            # Check for embedding size mismatch
            invalid_entries = []
            for name, stored_embedding in embeddings_db.items():
                if stored_embedding.shape != query_embedding.shape:
                    invalid_entries.append(name)
            
            # Remove invalid entries
            for name in invalid_entries:
                print(f"Warning: Removing invalid embedding for {name} (wrong size)")
                del embeddings_db[name]
            
            if invalid_entries:
                print("Please re-add the removed embeddings.")
    except Exception as e:
        print(f"Error loading database: {str(e)}")
        return None
    
    if not embeddings_db:
        print("Database is empty.")
        return None
    
    print(f"Comparing against {len(embeddings_db)} faces in database...")
    
    # Compare with each embedding using cosine similarity
    results = {}
    for person_name, stored_embedding in embeddings_db.items():
        print(f"Stored embedding shape for {person_name}: {stored_embedding.shape}")
        try:
            similarity = 1 - cosine(query_embedding, stored_embedding)
            results[person_name] = similarity
            print(f"Similarity with {person_name}: {similarity:.4f}")
        except Exception as e:
            print(f"Error comparing with {person_name}: {str(e)}")
            continue
    
    if not results:
        print("No valid comparisons could be made.")
        return None
    
    best_match = max(results.items(), key=lambda x: x[1])
    print(f"Best match found: {best_match[0]} with similarity {best_match[1]:.4f}")
    
    return best_match if best_match[1] > threshold else None

def capture_image():
    """Capture image from webcam and save it"""
    output_filename = input("Enter the filename to save the captured image (e.g., capture1.jpg): ")
    
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Webcam initialized. Press SPACE to capture or ESC to cancel.")
    print("Note: A bounding box will appear when a face is detected.")
    
    with mp_face_detection.FaceDetection(
        model_selection=0,  # Use 0 for webcam/close-range detection
        min_detection_confidence=0.5
    ) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error capturing frame.")
                break
            
            # Flip the frame horizontally for a more natural view
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect faces
            results = face_detection.process(frame_rgb)
            
            # Draw detection annotations on a copy of the frame
            display_frame = frame.copy()
            
            # Count detected faces
            num_faces = 0
            
            if results.detections:
                num_faces = len(results.detections)
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = display_frame.shape
                    
                    # Convert relative coordinates to absolute
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Draw rectangle
                    cv2.rectangle(display_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    
                    # Draw confidence score
                    confidence = detection.score[0]
                    cv2.putText(display_frame, f"{confidence:.2f}", (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display number of faces detected
            cv2.putText(display_frame, f"Faces detected: {num_faces}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow("Webcam Capture", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE key
                if num_faces == 1:
                    cv2.imwrite(output_filename, frame)
                    print(f"Image saved as {output_filename}")
                    break
                else:
                    print(f"Please ensure exactly one face is in frame (detected: {num_faces})")
            elif key == 27:  # ESC key
                print("Capture cancelled.")
                break
    
    cap.release()
    cv2.destroyAllWindows()
    return os.path.exists(output_filename)

def batch_process_saved_faces(interpreter, directory="saved_faces"):
    """Process all images in the saved_faces directory and save their embeddings"""
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} not found.")
        return
    
    processed = 0
    errors = 0
    
    print("Processing images...")
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            person_name = os.path.splitext(filename)[0]
            
            try:
                print(f"Processing image for {person_name}...")
                embedding = get_face_embedding(interpreter, img_path)
                save_embedding(embedding, person_name)
                processed += 1
                print(f"Successfully saved embedding for {person_name}.")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                errors += 1
    
    print(f"\nBatch processing complete:")
    print(f"Successfully processed: {processed} images")
    print(f"Errors encountered: {errors} images")

def process_save_embedding(interpreter):
    """Process an image and save its embedding"""
    print("\nChoose processing mode:")
    print("1. Process single image")
    print("2. Batch process saved_faces directory")
    print("3. Clean database and start fresh")
    mode = input("Enter your choice (1-3): ")
    
    if mode == "1":
        img_path = input("Enter the image filename (e.g., capture1.jpg): ")
        if not os.path.exists(img_path):
            print(f"Error: File {img_path} not found.")
            return
        
        person_name = input("Enter the person's name: ")
        try:
            print(f"Processing image: {img_path}")
            embedding = get_face_embedding(interpreter, img_path)
            save_embedding(embedding, person_name)
            print(f"Successfully saved embedding for {person_name}.")
        except Exception as e:
            print(f"Error: {str(e)}")
            return
    
    elif mode == "2":
        batch_process_saved_faces(interpreter)
    elif mode == "3":
        clean_database()
        print("Please re-add your face embeddings using option 1 or 2.")

def display_menu():
    """Display menu options"""
    print("\n===== Face Recognition System =====")
    print("1. Capture image from webcam")
    print("2. Save face embedding to database")
    print("3. Compare face against database")
    print("0. Exit")
    print("===================================")
    return input("Enter your choice (0-3): ")

def main():
    """Main program"""
    interpreter = get_tflite_model()
    
    while True:
        choice = display_menu()
        
        if choice == "1":
            capture_image()
        elif choice == "2":
            process_save_embedding(interpreter)
        elif choice == "3":
            img_path = input("Enter the image filename to compare: ")
            if not os.path.exists(img_path):
                print(f"Error: File {img_path} not found.")
                continue
            
            start_time = time.time()
            result = compare_face(interpreter, img_path)
            process_time = time.time() - start_time
            
            print(f"\nProcessing time: {process_time:.4f} seconds")
            if result:
                person_name, similarity = result
                print(f"\nMatch found: {person_name} (similarity: {similarity:.4f})")
            else:
                print("\nNo matching face found in the database.")
        elif choice == "0":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()