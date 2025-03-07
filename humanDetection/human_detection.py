import cv2
import numpy as np

# Open a connection to the first webcam
cap = cv2.VideoCapture(0)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optionally convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect people
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    # Draw bounding boxes
    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Display the number of detections
    num_detections = len(boxes)
    cv2.putText(frame,
                f"Detections: {num_detections}",
                (10, 30),  # position (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,
                1,         # font scale
                (255, 0, 0),  # color (B, G, R) – here, blue text
                2,         # thickness
                cv2.LINE_AA)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
