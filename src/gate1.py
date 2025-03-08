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


gate1 = Gate(
    gate_type=GateType.GATE1,
    mqtt_config=mqtt_config,
    update_config=update_config,
    voice_auth_config=None,  # No voice auth for gate1
    logging_level=logging.DEBUG,
)
gate1.run()
