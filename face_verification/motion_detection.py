import cv2
import numpy as np
import time
from datetime import datetime

class MotionDetector:
    def __init__(self, camera_id=0, resolution=(320, 240), threshold=25, 
                 min_area=500, blur_size=5, check_interval=1.0):
        self.camera_id = camera_id
        self.resolution = resolution
        self.threshold = threshold
        self.min_area = min_area
        self.blur_size = blur_size
        self.check_interval = check_interval
        
        self.camera = None
        self.previous_frame = None
        self.motion_detected = False
        self.running = False
        
    def start(self):
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise Exception(f"Could not open camera {self.camera_id}")
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        ret, frame = self.camera.read()
        if not ret:
            raise Exception("Failed to capture initial frame")
            
        self.previous_frame = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (self.blur_size, self.blur_size), 0
        )
        return self.camera
        
    def check_for_motion(self, frame):
        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (self.blur_size, self.blur_size), 0
        )
        
        thresh = cv2.dilate(
            cv2.threshold(
                cv2.absdiff(self.previous_frame, gray),
                self.threshold, 255, cv2.THRESH_BINARY
            )[1],
            None, iterations=2
        )
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.previous_frame = gray
        
        return any(cv2.contourArea(contour) >= self.min_area for contour in contours)
        
    def start_monitoring(self):
        if self.camera is None:
            self.start()
            
        self.running = True
        last_check = time.time()
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                current_time = time.time()
                if current_time - last_check >= self.check_interval:
                    if self.check_for_motion(frame):
                        print(f"Motion detected at {datetime.now()}")
                    last_check = current_time
                
                time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("Monitoring interrupted")
        finally:
            self.stop()
            
    def stop(self):
        self.running = False
        if self.camera:
            self.camera.release()
            self.camera = None

def main():
    detector = MotionDetector(
        resolution=(320, 240),
        threshold=30,
        min_area=1000,
        blur_size=5,
        check_interval=1.0
    )
    
    try:
        detector.start_monitoring()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
