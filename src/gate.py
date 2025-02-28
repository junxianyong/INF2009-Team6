from mqtt.publisher import Publisher
from mqtt.subscriber import Subscriber
from shared.enum import GateStatus, MantrapStatus
import logging
import time
from datetime import datetime
import json


def face_detected():
    """
    TODO: This function is a infinte loop that waits for a face to be detected
    """
    while True:
        if input("Face detected? (y/n)").lower() == "y":
            return True


def open_gate(id):
    """
    TODO: This function will open the gate
    """
    print(f"Gate {id} opened")


def face_verified():
    """
    TODO: This function is a infinte loop that waits for a face to be verified (after few tries?),
    it should return personnel ID if the face is verified, None otherwise
    """
    while True:
        x = input("Face verified? (y/n)")
        if x.lower() == "y":
            return "personnel_id"
        else:
            return None


def face_verified_with_id(personnel_id):
    """
    TODO: This function is a infinte loop that waits for a face to be verified with personnel ID,
    it should return True if the face is verified
    """
    while True:
        x = input(f"Face verified with personnel ID {personnel_id}? (y/n)")
        if x.lower() == "y":
            return True
        else:
            return False


def voice_verified(personnel_id):
    """
    TODO: This function is a infinte loop that waits for a voice to be verified with personnel ID,
    it should return True if the face is verified
    """
    while True:
        x = input(f"Voice verified with personnel ID {personnel_id}? (y/n)")
        if x.lower() == "y":
            return True
        else:
            return False


def personnel_passed():
    """
    TODO: This function is a infinte loop that waits for the personnel to pass through the
    gate, it should return True if the personnel has passed
    """
    while True:
        x = input("Personnel passed? (y/n)")
        if x.lower() == "y":
            return True


def close_gate(id):
    """
    TODO: This function will close the gate
    """
    print(f"Gate {id} closed")


def mantrap_scan():
    """
    TODO: This function is a infinte loop that waits for the mantrap to scan the personnel,
    if there is multiple personnel, it should return False, True otherwise
    """
    while True:
        x = input("Mantrap scanned? (y/n)")
        if x.lower() == "y":
            return True
        else:
            return False


def capture_intruder():
    """
    TODO: This function will capture the intruder and return the image buffer
    """
    print("Intruder captured")
    return "image_buffer"


def alert_buzzer_and_led():
    """
    TODO: This function will alert the buzzer and LED (Beep and Blink once and delay for 1 second)
    """
    time.sleep(1)
    print("Buzzer and LED alerted")


class Gate:
    def __init__(
        self, type, mqtt_broker, mqtt_port, mqtt_username, mqtt_password, logging_level
    ):
        self.last_logged_state = None
        self.current_status = None
        self.status_action_dict = {
            (GateStatus.CLOSED, MantrapStatus.IDLE, GateStatus.CLOSED): self.state_1,
            (GateStatus.FACE, MantrapStatus.IDLE, GateStatus.CLOSED): self.state_2,
            (GateStatus.OPENED, MantrapStatus.IDLE, GateStatus.CLOSED): self.state_3,
            (GateStatus.CLOSED, MantrapStatus.SCAN, GateStatus.CLOSED): self.state_4,
            (
                GateStatus.OPENED,
                MantrapStatus.CHECKED,
                GateStatus.CLOSED,
            ): self.state_5,
            (GateStatus.CLOSED, MantrapStatus.ALERT, GateStatus.CLOSED): self.state_6,
            (GateStatus.CLOSED, MantrapStatus.MULTI, GateStatus.CLOSED): self.state_7,
            (GateStatus.CLOSED, MantrapStatus.IDLE, GateStatus.FACE): self.state_8,
            (GateStatus.CLOSED, MantrapStatus.IDLE, GateStatus.DIFF): self.state_9,
            (GateStatus.CLOSED, MantrapStatus.IDLE, GateStatus.VOICE): self.state_10,
            (GateStatus.CLOSED, MantrapStatus.IDLE, GateStatus.OPENED): self.state_11,
        }
        self.type = type
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_username = mqtt_username
        self.mqtt_password = mqtt_password

        # Configure logging
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging_level)

        # Add console handler if none exists
        if not self._logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        self.publisher = Publisher(
            self.mqtt_broker, self.mqtt_port, self.mqtt_username, self.mqtt_password
        )
        self.subscriber = Subscriber(
            self.mqtt_broker, self.mqtt_port, self.mqtt_username, self.mqtt_password
        )
        self.personnel_id = None

    def _log_state(self, state_number):
        """Log state only if it has changed"""
        if state_number != self.last_logged_state:
            self._logger.info(f"State {state_number}")
            self.last_logged_state = state_number

    def state_1(self):
        self._log_state(1)
        if face_detected():
            self.current_status = (
                GateStatus.FACE,
                MantrapStatus.IDLE,
                GateStatus.CLOSED,
            )  # move to state 2

    def state_2(self):
        self._log_state(2)
        self.personnel_id = face_verified()
        if self.personnel_id:
            open_gate(1)
            self.publisher.publish(
                "gate_1/status",
                json.dumps({"opened": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}),
                2,
            )
            self.current_status = (
                GateStatus.OPENED,
                MantrapStatus.IDLE,
                GateStatus.CLOSED,
            )  # move to state 3
        else:
            self.current_status = (
                GateStatus.CLOSED,
                MantrapStatus.IDLE,
                GateStatus.CLOSED,
            )  # return to state 1

    def state_3(self):
        self._log_state(3)
        if personnel_passed():
            close_gate(1)
            self.current_status = (
                GateStatus.CLOSED,
                MantrapStatus.SCAN,
                GateStatus.CLOSED,
            )  # move to state 4

    def state_4(self):
        self._log_state(4)
        if self.type == "gate2":
            return
        self.publisher.publish(
            "gate_1/status",
            json.dumps({"closed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}),
            2,
        )
        if mantrap_scan():
            self.publisher.publish(
                "verified", json.dumps({"personnel_id": self.personnel_id}), 2
            )
            self.current_status = None  # Gate 2 will handle the rest
        else:
            self.current_status = (
                GateStatus.CLOSED,
                MantrapStatus.MULTI,
                GateStatus.CLOSED,
            )  # move to state 7

    def state_5(self):
        self._log_state(5)
        pass
        # Here has no manual transition to other states, it will wait for the subscriber to change the state

    def state_6(self):
        self._log_state(6)
        alert_buzzer_and_led()
        # Here has no manual transition to other states, it will wait for the subscriber to change the state

    def state_7(self):
        self._log_state(7)
        buffer = capture_intruder()
        self.publisher.publish(
            "alert",
            json.dumps(
                {
                    "type": "multi",
                    "image": buffer,
                }
            ),
            2,
        )
        self.current_status = (
            GateStatus.CLOSED,
            MantrapStatus.ALERT,
            GateStatus.CLOSED,
        )  # move to state 6

    def state_8(self):
        self._log_state(8)
        if face_verified_with_id(self.personnel_id):
            self.current_status = (
                GateStatus.CLOSED,
                MantrapStatus.IDLE,
                GateStatus.VOICE,
            )  # move to state 10
        else:
            self.current_status = (
                GateStatus.CLOSED,
                MantrapStatus.IDLE,
                GateStatus.DIFF,
            )  # move to state 9

    def state_9(self):
        self._log_state(9)
        buffer = capture_intruder()
        self.publisher.publish(
            "alert",
            json.dumps(
                {
                    "type": "diff",
                    "image": buffer,
                }
            ),
            2,
        )
        self.current_status = None  # Wait for gate 1 to handle the rest

    def state_10(self):
        self._log_state(10)
        if voice_verified(self.personnel_id):
            open_gate(2)
            self.publisher.publish(
                "gate_2/status",
                json.dumps({"opened": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}),
                2,
            )
            self.current_status = (
                GateStatus.CLOSED,
                MantrapStatus.IDLE,
                GateStatus.OPENED,
            )  # move to state 11
        else:
            self.current_status = (
                GateStatus.CLOSED,
                MantrapStatus.IDLE,
                GateStatus.DIFF,
            )  # move to state 9

    def state_11(self):
        self._log_state(11)
        if personnel_passed():
            close_gate(2)
            self.publisher.publish(
                "gate_2/status",
                json.dumps({"closed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}),
                2,
            )
            self.current_status = None  # Let gate 1 handle the rest

    def gate_1_subscribe_callback(self, topic, text_payload, raw_payload):
        self._logger.info(f"Received message: {text_payload} from topic: {topic}")
        if topic == "command":
            if (
                text_payload is not None
                and json.loads(text_payload)["action"] == "open"
            ):
                open_gate(1)
                self.current_status = (
                    GateStatus.OPENED,
                    MantrapStatus.CHECKED,
                    GateStatus.CLOSED,
                )  # move to state 5
            if (
                text_payload is not None
                and json.loads(text_payload)["action"] == "close"
            ):
                close_gate(1)
                self.current_status = (
                    GateStatus.CLOSED,
                    MantrapStatus.IDLE,
                    GateStatus.CLOSED,
                )  # move to state 1

        if topic == "gate_2/status":
            if text_payload is not None:
                payload = json.loads(text_payload)
                if "closed" in payload and payload["closed"] is not None:
                    self.current_status = (
                        GateStatus.CLOSED,
                        MantrapStatus.IDLE,
                        GateStatus.CLOSED,
                    )  # move to state 1

        if topic == "alert":
            if text_payload is not None and json.loads(text_payload)["type"] == "diff":
                self.current_status = (
                    GateStatus.CLOSED,
                    MantrapStatus.ALERT,
                    GateStatus.CLOSED,
                )

    def gate_2_subscribe_callback(self, topic, text_payload, raw_payload):
        self._logger.info(f"Received message: {text_payload} from topic: {topic}")
        if topic == "verified":
            if text_payload is not None and json.loads(text_payload)["personnel_id"]:
                self.personnel_id = json.loads(text_payload)["personnel_id"]
                self._logger.info(f"Personnel ID: {self.personnel_id}")
                self.current_status = (
                    GateStatus.CLOSED,
                    MantrapStatus.IDLE,
                    GateStatus.FACE,
                )  # move to state 9

    def run(self):
        self.publisher.connect()
        self.subscriber.connect()

        if self.type == "gate1":
            self.current_status = (
                GateStatus.CLOSED,
                MantrapStatus.IDLE,
                GateStatus.CLOSED,
            )  # originally state 1
            self.subscriber.on_message_callback = self.gate_1_subscribe_callback
            self.subscriber.subscribe("gate_2/status", 2)
            self.subscriber.subscribe("command", 2)
            self.subscriber.subscribe("alert", 2)

        if self.type == "gate2":
            self.current_status = None  # wait for the message from gate 1
            self.subscriber.on_message_callback = self.gate_2_subscribe_callback
            self.subscriber.subscribe("verified", 2)

        while True:
            if self.current_status is not None:
                self.status_action_dict[self.current_status]()
