# main.py
import os
import time
from typing import Dict, Tuple
import cv2
import numpy as np

from detect import Detect
from streams import VideoStream
from tracker import Tracker
from utils import direction_config

def _detect_person(detect: Detect, frame: np.ndarray, confidence: float, iou_threshold: float) -> np.ndarray:
    # [Your detection code remains the same...]
    boxes, scores, class_idx = detect.detect(frame)
    idx = cv2.dnn.NMSBoxes(boxes, scores, confidence, iou_threshold)
    boxes = boxes[idx]
    scores = scores[idx]
    class_idx = class_idx[idx]
    person_idx = np.where(class_idx == 0)[0]
    boxes = boxes[person_idx]
    scores = scores[person_idx]
    H, W = frame.shape[:2]
    boxes = detect.to_xyxy(boxes) * np.array([W, H, W, H])
    dets = np.concatenate([boxes.astype(int), scores.reshape(-1, 1)], axis=1)
    return dets

def run_tracking(src: str, dest: str, model: str, video_fmt: str, confidence: float,
                 iou_threshold: float, directions: Dict[str, Tuple[bool]], duration: int = 30) -> float:
    if not os.path.exists(dest):
        os.mkdir(dest)
    
    try:
        src = int(src)
    except ValueError:
        pass

    # Setup tracking components.
    border = [(0, 500), (1920, 500)]
    directions = {key: direction_config.get(d_str) for key, d_str in directions.items()}
    tracker = Tracker(border, directions)
    detect = Detect(model, confidence)
    stream = VideoStream(src)
    writer = None
    
    # Variables to track detections.
    human_counts = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        is_finish, frame = stream.next()
        if not is_finish:
            break
        
        dets = _detect_person(detect, frame, confidence, iou_threshold)
        human_count = dets.shape[0]
        human_counts.append(human_count)
        
        # Overlay the count on the frame.
        cv2.putText(frame, f"Humans detected: {human_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"Detected {human_count} humans in this frame.")
        
        # Update tracker (you can modify tracker.update as needed).
        frame = tracker.update(frame, dets)
        
        # Optionally write to file (if needed).
        if writer is None:
            model_name = os.path.basename(model).split(".")[0]
            video_name = "webcam_live"
            codecs = {"mp4": "MP4V", "avi": "MJPG"}
            basename = f"{video_name}_{model_name}"
            output_video = os.path.join(dest, f"{basename}.{video_fmt}")
            fourcc = cv2.VideoWriter_fourcc(*codecs[video_fmt])
            writer = cv2.VideoWriter(output_video, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
            if not writer.isOpened():
                print("Error: VideoWriter failed to open.")
        if writer is not None:
            writer.write(frame)
        
        # Optionally display the live video.
        cv2.imshow("Live Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if writer is not None:
        writer.release()
    stream.release()
    cv2.destroyAllWindows()
    
    # Compute average human count.
    avg_count = sum(human_counts) / len(human_counts) if human_counts else 0
    print("Done! Average human count:", avg_count)
    return avg_count

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="Path to video source.", default="./data/TownCentreXVID.mp4")
    parser.add_argument("--dest", help="Path to output directory", default="./outputs/")
    parser.add_argument("--model", help="Path to YOLOv5 tflite file", default="./models/yolov5n6-fp16.tflite")
    parser.add_argument("--video-fmt", help="Format of output video file.", choices=["mp4", "avi"], default="mp4")
    parser.add_argument("--confidence", type=float, default=0.2, help="Confidence threshold.")
    parser.add_argument("--iou-threshold", type=float, default=0.2, help="IoU threshold for NMS.")
    parser.add_argument("--directions", default={"total": None}, type=eval, help="Directions")
    parser.add_argument("--duration", type=int, default=30, help="Duration (in seconds) to run tracking.")
    
    args = vars(parser.parse_args())
    run_tracking(**args)
