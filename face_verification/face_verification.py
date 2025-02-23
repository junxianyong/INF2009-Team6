import time
import cv2
from deepface import DeepFace
import os
import numpy as np
import pickle

# Directory to store saved faces
SAVED_FACES_DIR = "saved_faces"
EMBEDDINGS_FILE = "face_embeddings.pkl"

# Create the directory if it doesn't exist
if not os.path.exists(SAVED_FACES_DIR):
    os.makedirs(SAVED_FACES_DIR)

def load_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_embeddings(embeddings):
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)

def capture_face():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 's' to capture and exit.")

    saved_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        cv2.imshow("Capture Face - Press 's' to Save", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') or key == ord('S'):
            # Store the last frame in memory
            saved_frame = frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()

    # Once the window is closed, THEN ask for the name and save if needed
    if saved_frame is not None:
        name = input("Enter the name for the face: ").strip()
        if name:
            image_path = os.path.join(SAVED_FACES_DIR, f"{name}.jpg")
            cv2.imwrite(image_path, saved_frame)
            print(f"Face saved as {image_path}")
        else:
            print("Name cannot be empty. Face not saved.")

def regenerate_embeddings():
    """Regenerate embeddings for all saved faces"""
    saved_faces = load_saved_faces()
    if not saved_faces:
        print("No saved faces found. Please save some faces first.")
        return

    new_embeddings = {}
    total_faces = len(saved_faces)
    processed = 0

    print(f"Found {total_faces} faces. Starting embedding generation...")
    
    for name, image_path in saved_faces.items():
        try:
            processed += 1
            print(f"Processing {name} ({processed}/{total_faces})...")
            
            # Read the image and generate embedding
            img = cv2.imread(image_path)
            embedding_obj = DeepFace.represent(img, model_name="VGG-Face", enforce_detection=False)[0]
            # Store just the embedding array, not the whole object
            new_embeddings[name] = embedding_obj['embedding'] if isinstance(embedding_obj, dict) else embedding_obj
            print(f"✓ Successfully generated embedding for {name}")
            
        except Exception as e:
            print(f"✗ Error generating embedding for {name}: {e}")
    
    if new_embeddings:
        save_embeddings(new_embeddings)
        print(f"\nSuccessfully generated and saved {len(new_embeddings)} embeddings")
    else:
        print("\nNo embeddings were generated")

def load_saved_faces():
    # Load saved faces and their names from the directory
    saved_faces = {}
    for file in os.listdir(SAVED_FACES_DIR):
        if file.endswith(".jpg"):
            name = file.split(".")[0]
            saved_faces[name] = os.path.join(SAVED_FACES_DIR, file)

    # Sort the dictionary by keys (names) in alphabetical order (Debugging purpose)
    sorted_faces = dict(sorted(saved_faces.items()))
    return sorted_faces

import time

def cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings - this is what DeepFace uses internally
    """
    a = np.array(embedding1).flatten()
    b = np.array(embedding2).flatten()
    
    cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # Convert to distance (0 to 1, where 0 means identical)
    distance = 1 - cosine
    return distance

def compare_embeddings(embedding1, embedding2, threshold=0.4):  # Changed threshold to match DeepFace
    print("\nDebug - Comparing embeddings:")
    print(f"Threshold: {threshold}")
    
    # Convert embeddings to numpy arrays if they aren't already
    emb1 = np.array(embedding1['embedding'] if isinstance(embedding1, dict) else embedding1)
    emb2 = np.array(embedding2['embedding'] if isinstance(embedding2, dict) else embedding2)
    
    # Calculate cosine distance (same as DeepFace)
    distance = cosine_similarity(emb1, emb2)
    match = distance < threshold
    
    print(f"Cosine distance: {distance:.4f}")
    print(f"Is match? {match} {'(Good match!)' if distance < 0.3 else '(Weak match)' if match else ''}")
    
    return match, distance

def get_confidence_score(distance, method="embedding"):
    """
    Convert distance to confidence score - now using same scale for both methods
    """
    # Both methods now use same scale (0-1 distance)
    return round((1 - distance) * 100, 2)

def capture_and_compare():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 's' to capture and compare.")

    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        cv2.imshow("Capture Face - Press 's' to Save and Compare", frame)
        key = cv2.waitKey(1) & 0xFF

        # Check if the user closes the window
        if cv2.getWindowProperty("Capture Face - Press 's' to Save and Compare", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed. Exiting capture.")
            break

        if key == ord('s') or key == ord('S'):
            # Store the last frame in memory
            captured_frame = frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is not None:
        embeddings = load_embeddings()
        if not embeddings:
            print("No face embeddings found. Please save some faces first.")
            return

        print("\nDebug - Loaded embeddings:")
        print(f"Number of stored embeddings: {len(embeddings)}")
        print("Stored names:", list(embeddings.keys()))

        try:
            faces = DeepFace.extract_faces(captured_frame, detector_backend="opencv")
            print(f"\nDebug - Found {len(faces)} faces in captured frame")
            
            for face_idx, face in enumerate(faces):
                print(f"\nProcessing face {face_idx + 1}:")
                x, y, w, h = (face["facial_area"]["x"],
                            face["facial_area"]["y"],
                            face["facial_area"]["w"],
                            face["facial_area"]["h"])
                
                face_img = captured_frame[y:y+h, x:x+w]
                print("Getting embedding for captured face...")
                captured_embedding_obj = DeepFace.represent(face_img, model_name="VGG-Face", enforce_detection=False)[0]
                
                print("\nDebug - Captured face embedding:")
                print(f"Type: {type(captured_embedding_obj)}")
                if isinstance(captured_embedding_obj, dict):
                    print("Keys:", captured_embedding_obj.keys())
                
                captured_embedding = captured_embedding_obj['embedding'] if isinstance(captured_embedding_obj, dict) else captured_embedding_obj
                print(f"Embedding shape: {np.array(captured_embedding).shape}")

                best_match = None
                best_distance = float('inf')
                start_time = time.time()

                print("\nComparing with stored faces...")
                distances = []  # Store all distances for analysis
                
                for name, stored_embedding in embeddings.items():
                    print(f"\nChecking against {name}:")
                    verified, distance = compare_embeddings(captured_embedding, stored_embedding)
                    distances.append((name, distance))
                    confidence = get_confidence_score(distance)
                    print(f"Confidence: {confidence}%")
                    if verified and distance < best_distance:
                        best_match = name
                        best_distance = distance
                        print(f"New best match: {name} (distance: {distance:.4f}, confidence: {confidence}%)")
                
                # Sort and analyze distances
                distances.sort(key=lambda x: x[1])
                print("\nAll distances from closest to furthest:")
                for name, dist in distances:
                    print(f"{name}: {dist:.4f}")
                
                # If closest matches are very close to each other, show both
                if len(distances) >= 2:
                    closest, second_closest = distances[:2]
                    distance_difference = second_closest[1] - closest[1]
                    print(f"\nDistance difference between top 2 matches: {distance_difference:.4f}")
                    
                    if distance_difference < 0.1:  # If very close, show both
                        print(f"Note: Top 2 matches ({closest[0]} and {second_closest[0]}) are very close!")
                
                end_time = time.time()
                inference_time = end_time - start_time

                # Draw results with new confidence calculation
                if best_match:
                    confidence = get_confidence_score(best_distance)
                    label = f"{best_match} ({confidence:.1f}%)"
                    
                    # Adjust color based on confidence
                    if confidence >= 90:
                        color = (0, 255, 0)  # Green for high confidence
                    elif confidence >= 70:
                        color = (0, 255, 255)  # Yellow for medium confidence
                    else:
                        color = (0, 165, 255)  # Orange for low confidence
                    
                    print(f"\nFinal match: {best_match} (confidence: {confidence:.1f}%) in {inference_time:.2f} seconds")
                    print(f"Distance: {best_distance:.4f} - " + 
                          ("Very high similarity" if confidence >= 95 else
                           "High similarity" if confidence >= 85 else
                           "Good similarity" if confidence >= 75 else
                           "Moderate similarity" if confidence >= 65 else
                           "Low similarity"))
                else:
                    label = "Unknown"
                    color = (0, 0, 255)  # Red for no match
                    print("\nNo match found")

                cv2.rectangle(captured_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(captured_frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Display the frame with bounding box and label
            cv2.imshow("Captured Image - Detected Face", captured_frame)

            # Wait for the user to close the window or press any key
            while cv2.getWindowProperty("Captured Image - Detected Face", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.waitKey(1)

        except Exception as e:
            print(f"Error during face detection or verification: {e}")
            import traceback
            print("Full error:")
            print(traceback.format_exc())

        cv2.destroyAllWindows()

def capture_and_compare_traditional():
    """Traditional method that compares against each saved image directly"""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 's' to capture and compare (Traditional Method).")

    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        cv2.imshow("Capture Face - Press 's' to Save and Compare", frame)
        key = cv2.waitKey(1) & 0xFF

        if cv2.getWindowProperty("Capture Face - Press 's' to Save and Compare", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed. Exiting capture.")
            break

        if key == ord('s') or key == ord('S'):
            captured_frame = frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is not None:
        saved_faces = load_saved_faces()
        if not saved_faces:
            print("No faces found in the directory. Please save a face first.")
            return

        try:
            faces = DeepFace.extract_faces(captured_frame, detector_backend="opencv")
            print(f"\nFound {len(faces)} faces in captured frame")
            total_time = 0
            
            for face_idx, face in enumerate(faces):
                x, y, w, h = (face["facial_area"]["x"],
                             face["facial_area"]["y"],
                             face["facial_area"]["w"],
                             face["facial_area"]["h"])
                
                print(f"\nProcessing face {face_idx + 1}")
                face_img = captured_frame[y:y+h, x:x+w]
                
                # Compare with each saved face
                best_match = None
                best_confidence = 0
                face_times = []

                for name, saved_face_path in saved_faces.items():
                    print(f"\nComparing with {name}...")
                    start_time = time.time()
                    
                    try:
                        result = DeepFace.verify(face_img,
                                               saved_face_path,
                                               enforce_detection=False)
                        
                        end_time = time.time()
                        comparison_time = end_time - start_time
                        face_times.append(comparison_time)
                        total_time += comparison_time
                        
                        print(f"Time taken: {comparison_time:.2f} seconds")
                        print(f"Raw distance: {result['distance']:.4f}")
                        confidence = get_confidence_score(result['distance'], "traditional")
                        print(f"Confidence: {confidence}%")
                        print(f"Verified: {result['verified']}")
                        
                        if result["verified"]:
                            if confidence > best_confidence:
                                best_match = name
                                best_confidence = confidence
                    
                    except Exception as e:
                        print(f"Error comparing with {name}: {e}")

                # Draw results
                if best_match:
                    label = f"{best_match} ({best_confidence}%)"
                    color = (0, 255, 0)
                    print(f"\nBest match: {best_match} with {best_confidence}% confidence")
                else:
                    label = "Unknown"
                    color = (0, 0, 255)
                    print("\nNo match found")

                cv2.rectangle(captured_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(captured_frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Print timing statistics
                if face_times:
                    avg_time = sum(face_times) / len(face_times)
                    print(f"\nTiming Statistics:")
                    print(f"Average time per comparison: {avg_time:.2f} seconds")
                    print(f"Total time for all comparisons: {total_time:.2f} seconds")
                    print(f"Number of comparisons: {len(face_times)}")

            cv2.imshow("Captured Image - Detected Face (Traditional Method)", captured_frame)
            while cv2.getWindowProperty("Captured Image - Detected Face (Traditional Method)", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.waitKey(1)

        except Exception as e:
            print(f"Error during face detection or verification: {e}")
            import traceback
            print("Full error:")
            print(traceback.format_exc())

        cv2.destroyAllWindows()

def main():
    while True:
        print("\nOptions:")
        print("1. Capture and save face image")
        print("2. Generate/regenerate embeddings for saved faces")
        print("3. Verify face (compare with saved faces using embeddings)")
        print("4. Exit")
        print("9. Verify face (traditional method - slower but more accurate)")
        choice = input("Enter your choice (1-4, or 9): ")

        if choice == "1":
            capture_face()
        elif choice == "2":
            regenerate_embeddings()
        elif choice == "3":
            capture_and_compare()
        elif choice == "4":
            print("Exiting...")
            break
        elif choice == "9":
            capture_and_compare_traditional()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

