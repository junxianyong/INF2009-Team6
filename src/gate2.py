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

voice_auth_config = {
    "voiceprints_file": "voiceprints.pkl",
    "sr_rate": 44100,
    "num_mfcc": 20,
    "linear_threshold": 100,
    "cos_threshold": 0.95,
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

gate2 = Gate(
    gate_type=GateType.GATE2,
    mqtt_config=mqtt_config,
    update_config=update_config,
    voice_auth_config=voice_auth_config,
    motion_detector_config=None,  # No motion detection for gate2
    face_verification_config=face_verification_config,
    intruder_detection_config=None,
    logging_level=logging.DEBUG,
)
gate2.run()
