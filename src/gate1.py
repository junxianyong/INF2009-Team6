from gate.gate import Gate
import logging

gate1 = Gate(
    type="gate1",
    mqtt_broker="localhost",
    mqtt_port=1883,
    mqtt_username="mosquitto",
    mqtt_password="mosquitto",
    logging_level=logging.INFO,
)
gate1.run()
