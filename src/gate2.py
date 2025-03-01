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

gate2 = Gate(
    gate_type=GateType.GATE2,
    mqtt_broker=mqtt_config,
    update_config=update_config,
    logging_level=logging.DEBUG,
)
gate2.run()
