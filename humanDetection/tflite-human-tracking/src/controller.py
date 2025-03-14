import time
from main import run_tracking 

def perform_tracking(duration=30) -> bool:

    avg = run_tracking(
        src=0,
        dest="../outputs/",
        model="../models/yolov5/yolov5n-fp16.tflite",
        video_fmt="mp4",
        confidence=0.2,
        iou_threshold=0.2,
        directions={"total": None, "inside": "bottom", "outside": "top"},
        duration=duration
    )
    print(f"Average humans detected over {duration} seconds: {avg:.2f}")
    return avg == 1

def main():
    while True:
        cmd = input("Type 'start' to begin human tracking (or 'exit' to quit): ")
        if cmd.lower() == "start":
            print("Starting tracking for 30 seconds...")
            result = perform_tracking(30)
            print("Tracking result (True if exactly 1 human detected on average):", result)

        elif cmd.lower() == "exit":
            print("Exiting.")
            break
        else:
            print("Unrecognized command. Please type 'start' or 'exit'.")

if __name__ == "__main__":
    main()
