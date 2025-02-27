from mqtt.publisher import Publisher
from mqtt.subscriber import Subscriber
from shared.enum import GateStatus, MantrapStatus
import logging
import time


def face_detected():
    """
    TODO: This function is a infinte loop that waits for a face to be detected
    """
    while True:
        if input("Face detected? (y/n)").lower() == "y":
            return True


def open_gate():
    """
    TODO: This function will open the gate
    """
    print("Gate opened")


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


def personnel_passed():
    """
    TODO: This function is a infinte loop that waits for the personnel to pass through the
    gate, it should return True if the personnel has passed
    """
    while True:
        x = input("Test state 3? (y/n)")
        if x.lower() == "y":
            return True


def close_gate():
    """
    TODO: This function will close the gate
    """
    print("Gate closed")


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


def alert_buzzer_and_led():
    """
    TODO: This function will alert the buzzer and LED (Beep and Blink once and delay for 1 second)
    """
    print("Buzzer and LED alerted")


class Gate1:
    def __init__(self, logging_level=logging.INFO):
        self.current_status = (GateStatus.CLOSED, MantrapStatus.IDLE)
        self.status_action_dict = {
            (GateStatus.CLOSED, MantrapStatus.IDLE, GateStatus.CLOSED): self._state_1,
            (GateStatus.FACE, MantrapStatus.IDLE, GateStatus.CLOSED): self._state_2,
            (GateStatus.OPENED, MantrapStatus.IDLE, GateStatus.CLOSED): self._state_3,
            (GateStatus.CLOSED, MantrapStatus.SCAN, GateStatus.CLOSED): self._state_4,
            (
                GateStatus.OPENED,
                MantrapStatus.CHECKED,
                GateStatus.CLOSED,
            ): self._state_5,
            (GateStatus.CLOSED, MantrapStatus.ALERT, GateStatus.CLOSED): self._state_6,
            (GateStatus.CLOSED, MantrapStatus.MULTI, GateStatus.CLOSED): self._state_7,
        }
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging_level)
        self.publisher = Publisher(
            "gate_1", "test.mosquitto.org", 1884, "rw", "readwrite"
        ).connect()
        self.mantap_subscriber = Subscriber(
            "mantrap", "test.mosquitto.org", 1884, "rw", "readwrite", mantrap_callback
        ).connect()
        self.personnel_id = None

        def mantrap_callback(topic, text_payload, raw_payload):
            if text_payload is not None and text_payload == "open":
                self.current_status = (
                    GateStatus.OPENED,
                    MantrapStatus.CHECKED,
                    GateStatus.CLOSED,
                )  # move to state 5
            if text_payload is not None and text_payload == "close":
                self.current_status = (
                    GateStatus.CLOSED,
                    MantrapStatus.IDLE,
                    GateStatus.CLOSED,
                )  # move to state 1

    def _state_1(self):
        self._logger.info("State 1")
        if face_detected():
            self.current_status = (
                GateStatus.FACE,
                MantrapStatus.IDLE,
                GateStatus.CLOSED,
            )  # move to state 2

    def _state_2(self):
        self._logger.info("State 2")
        self.personnel_id = face_verified()
        if self.personnel_id:
            # This function will open the gate
            open_gate()
            # TODO self.publisher.publish(str({"opened": time.time()}))
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

    def _state_3(self):
        self._logger.info("State 3")
        if personnel_passed():
            close_gate()
            self.current_status = (
                GateStatus.CLOSED,
                MantrapStatus.SCAN,
                GateStatus.CLOSED,
            )  # move to state 4

    def _state_4(self):
        self._logger.info("State 4")
        # TODO self.publisher.publish(str({"closed": time.time()}))
        if mantrap_scan():
            self.current_status = (
                GateStatus.CLOSED,
                MantrapStatus.IDLE,
                GateStatus.FACE,
            )  # move to state 8 (Gate 2)
            # Simulate the gate 2 response
            time.sleep(5)
            self.current_status = (
                GateStatus.CLOSED,
                MantrapStatus.IDLE,
                GateStatus.CLOSED,
            )
        else:
            self.current_status = (
                GateStatus.CLOSED,
                MantrapStatus.MULTI,
                GateStatus.CLOSED,
            )  # move to state 7

    def _state_5(self):
        self._logger.info("State 5")
        pass
        # Here has manual transition to other states, it will wait for the subscriber to change the state

    def _state_6(self):
        self._logger.info("State 6")
        alert_buzzer_and_led()
        # Here has manual transition to other states, it will wait for the subscriber to change the state

    def _state_7(self):
        self._logger.info("State 7")
        buffer = capture_intruder()
        self.publisher.publish(
            "alert",
            {
                "alert": "multi",
                "image": buffer,
            },
        )
        self.current_status = (
            GateStatus.CLOSED,
            MantrapStatus.ALERT,
            GateStatus.CLOSED,
        )  # move to state 6

    def run(self):
        while True:
            self.status_action_dict[self.current_status]()


if __name__ == "__main__":
    gate = Gate1()
    gate.run()
