from gate.gate import Gate
import logging

from gate.gate_types import GateType

gate1 = Gate(
    gate_type=GateType.GATE1,
    mqtt_broker="localhost",
    mqtt_port=1883,
    mqtt_username="mosquitto",
    mqtt_password="mosquitto",
    update_url="https://files.testfile.org/PDF/",
    update_save_path="gate1/update",
    logging_level=logging.DEBUG,
)
gate1.run()
