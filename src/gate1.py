import logging

from gate.enum.gate_types import GateType
from gate.gate import Gate

mqtt_config = {
    "broker": "localhost",
    "port": 1883,
    "username": "mosquitto",
    "password": "mosquitto",
}

update_config = {
    "url": "https://files.testfile.org/PDF/",
    "save_path": "gate1/update",
}

motion_detector_config = {
    "camera_id": 0,
    "resolution": (320, 240),
    "threshold": 25,
    "min_area": 500,
    "blur_size": 5,
    "check_interval": 1.0,
    "fps": 1,  # Using 1 fps for low power consumption
}

face_verification_config = {
    "model_path": "mobilefacenet.tflite",
    "database_path": "face_embeddings.pkl",
    # Face detection & preprocessing settings:
    "model_selection": 0,
    "min_detection_confidence": 0.7,
    "padding": 0.2,
    "face_required_size": (512, 512),
    "target_size": (112, 112),
    # Verification settings:
    "verification_threshold": 0.7,
    "verification_timeout": 30,
    "verification_max_attempts": 3,
    # Camera settings:
    "camera_id": 0,
}

gate1 = Gate(
    gate_type=GateType.GATE1,
    mqtt_config=mqtt_config,
    update_config=update_config,
    voice_auth_config=None,  # No voice auth for gate1
    motion_detector_config=motion_detector_config,
    face_verification_config=face_verification_config,
    logging_level=logging.DEBUG,
)
gate1.run()
