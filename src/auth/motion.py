import logging
import cv2
import time
from datetime import datetime
from utils.logger_mixin import LoggerMixin


class MotionDetector(LoggerMixin):
    def __init__(self, motion_detector_config, logging_level=logging.INFO):
        self._camera_id = motion_detector_config["camera_id"]
        self._resolution = motion_detector_config["resolution"]
        self._threshold = motion_detector_config["threshold"]
        self._min_area = motion_detector_config["min_area"]
        self._blur_size = motion_detector_config["blur_size"]
        self._check_interval = motion_detector_config["check_interval"]
        self._fps = motion_detector_config["fps"]  # Target frames per second
        self.logger = self._setup_logger(__name__, logging_level)
        self._camera = None
        self._previous_frame = None
        self._running = False

    def _start(self):
        self._camera = cv2.VideoCapture(self._camera_id)
        if not self._camera.isOpened():
            raise Exception(f"Could not open camera {self._camera_id}")

        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])

        ret, frame = self._camera.read()
        if not ret:
            raise Exception("Failed to capture initial frame")

        self._previous_frame = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (self._blur_size, self._blur_size),
            0,
        )
        return self._camera

    def _check_for_motion(self, frame):
        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (self._blur_size, self._blur_size),
            0,
        )

        thresh = cv2.dilate(
            cv2.threshold(
                cv2.absdiff(self._previous_frame, gray),
                self._threshold,
                255,
                cv2.THRESH_BINARY,
            )[1],
            None,
            iterations=2,
        )

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self._previous_frame = gray

        return any(cv2.contourArea(contour) >= self._min_area for contour in contours)

    def start_monitoring(self):
        if self._camera is None:
            self._start()

        self._running = True
        last_check = time.time()

        try:
            while self._running:
                ret, frame = self._camera.read()
                if not ret:
                    break

                current_time = time.time()
                if current_time - last_check >= self._check_interval:
                    if self._check_for_motion(frame):
                        self.logger.info(f"Motion detected at {datetime.now()}")
                    last_check = current_time

                time.sleep(0.01)

        except KeyboardInterrupt:
            self.logger.debug("Monitoring interrupted")
        finally:
            self._stop()

    def _stop(self):
        self._running = False
        if self._camera:
            self._camera.release()
            self._camera = None

    def wait_for_motion(self, timeout=None, warmup_time=1.0):
        """
        Wait for motion to be detected and return True when detected.

        Args:
            timeout: Maximum time to wait for motion (None for infinite)
            warmup_time: Time in seconds to let the camera stabilize before detecting motion

        Returns:
            True when motion is detected, False if timeout is reached
        """
        try:
            self.logger.debug(
                f"Motion detection initialized at {datetime.now()} with {self._fps} fps"
            )
            if self._camera is None:
                self._start()

            # Warm-up period: capture frames but don't detect motion
            self.logger.debug(f"Warming up camera for {warmup_time} seconds...")
            warmup_start = time.time()
            while time.time() - warmup_start < warmup_time:
                ret, frame = self._camera.read()
                if not ret:
                    self.logger.error("Failed to read frame during warm-up")
                    return False

                # Update previous_frame during warm-up to avoid false positives
                gray = cv2.GaussianBlur(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    (self._blur_size, self._blur_size),
                    0,
                )
                self._previous_frame = gray

                # Sleep to maintain the desired frame rate
                time.sleep(1.0 / self._fps)

            self.logger.debug(
                f"Warm-up complete. Motion detection actively started at {datetime.now()}"
            )

            start_time = time.time()
            last_check = start_time
            frame_time = start_time

            while True:
                current_time = time.time()

                # Limit frame capture rate to specified fps
                if current_time - frame_time >= 1.0 / self._fps:
                    ret, frame = self._camera.read()
                    if not ret:
                        self.logger.error("Failed to read frame from camera")
                        return False

                    frame_time = current_time

                    # Check for motion at the check_interval
                    if current_time - last_check >= self._check_interval:
                        if self._check_for_motion(frame):
                            motion_time = datetime.now()
                            self.logger.info(f"Motion detected at {motion_time}")
                            return True
                        last_check = current_time

                # Check for timeout
                if timeout and (current_time - start_time > timeout):
                    self.logger.debug(f"Timeout reached after {timeout} seconds")
                    return False

                # Small sleep to prevent CPU hogging
                time.sleep(0.01)

        except KeyboardInterrupt:
            self.logger.debug("Motion detection interrupted")
            return False
        except Exception as e:
            self.logger.error(f"Error during motion detection: {e}")
            return False
        finally:
            self._stop()


if __name__ == "__main__":
    print("Waiting for motion...")
    motion_detector_config = {
        "camera_id": 0,
        "resolution": (320, 240),
        "threshold": 25,
        "min_area": 500,
        "blur_size": 5,
        "check_interval": 1.0,
        "fps": 1,  # Using 1 fps for low power consumption
    }
    detector = MotionDetector(motion_detector_config)
    if detector.wait_for_motion(warmup_time=1.0):
        print("Motion detected! Taking action...")
    else:
        print("Motion detection stopped or failed.")
