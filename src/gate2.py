from gate.gate import Gate
import logging
from gate.enum.gate_types import GateType

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

gate2 = Gate(
    gate_type=GateType.GATE2,
    mqtt_config=mqtt_config,
    update_config=update_config,
    voice_auth_config=voice_auth_config,
    logging_level=logging.DEBUG,
)
gate2.run()
