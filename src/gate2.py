from gate.gate import Gate
import logging

gate2 = Gate(
    type="gate2",
    mqtt_broker="localhost",
    mqtt_port=1883,
    mqtt_username="mosquitto",
    mqtt_password="mosquitto",
    update_url="https://files.testfile.org/PDF/",
    update_save_path="gate2/update",
    logging_level=logging.DEBUG,
)
gate2.run()
