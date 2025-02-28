from gate.gate import Gate
import logging

gate2 = Gate(
    type="gate2",
    mqtt_broker="localhost",
    mqtt_port=1883,
    mqtt_username="mosquitto",
    mqtt_password="mosquitto",
    logging_level=logging.INFO,
)
gate2.run()
