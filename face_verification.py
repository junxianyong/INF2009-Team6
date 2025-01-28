import time
import cv2
from deepface import DeepFace
import os

# Directory to store saved faces
SAVED_FACES_DIR = "saved_faces"

# Create the directory if it doesn't exist
if not os.path.exists(SAVED_FACES_DIR):
    os.makedirs(SAVED_FACES_DIR)

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
        # Load saved faces
        saved_faces = load_saved_faces()
        if not saved_faces:
            print("No faces found in the directory. Please save a face first.")
            return

        try:
            # Detect faces in the captured frame
            faces = DeepFace.extract_faces(captured_frame, detector_backend="opencv")
            for face in faces:
                x, y, w, h = (face["facial_area"]["x"],
                              face["facial_area"]["y"],
                              face["facial_area"]["w"],
                              face["facial_area"]["h"])

                # Compare the detected face with saved faces
                label = "Unknown"
                color = (0, 0, 255)  # Default: Red for no match
                for name, saved_face_path in saved_faces.items():
                    print(f"Checking for {name}...")

                    # Start timing
                    start_time = time.time()

                    # Perform face verification
                    result = DeepFace.verify(captured_frame[y:y+h, x:x+w],
                                             saved_face_path,
                                             enforce_detection=False)

                    # End timing
                    end_time = time.time()

                    # Calculate inference time
                    inference_time = end_time - start_time
                    print(f"Time taken to match with {name}: {inference_time:.2f} seconds")

                    if result["verified"]:
                        confidence = round((1 - result["distance"]) * 100, 2)
                        label = f"{name} ({confidence}%)"
                        color = (0, 255, 0)  # Green for match
                        print(f"Detected {name} ({confidence}%)")
                        break

                # Draw rectangle and label on the frame
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

        cv2.destroyAllWindows()

def main():
    while True:
        print("\nOptions:")
        print("1. Capture and save your face")
        print("2. Capture and compare with saved faces")
        print("3. Exit")
        choice = input("Enter your choice (1, 2, or 3): ")

        if choice == "1":
            capture_face()
        elif choice == "2":
            capture_and_compare()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

