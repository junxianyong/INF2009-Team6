from mqtt.publisher import Publisher
from mqtt.subscriber import Subscriber
import logging
import time
from datetime import datetime
import json
from gate.update import UpdateDownloader
from queue import Queue
from gate.example import *


class Gate:
    def __init__(
        self,
        type,
        mqtt_broker,
        mqtt_port,
        mqtt_username,
        mqtt_password,
        update_url,
        update_save_path,
        logging_level,
    ):
        self.last_logged_state = None
        self.current_status = None
        self.status_action_dict = {
            1: self.state_1,
            2: self.state_2,
            3: self.state_3,
            4: self.state_4,
            5: self.state_5,
            6: self.state_6,
            7: self.state_7,
            8: self.state_8,
            9: self.state_9,
            10: self.state_10,
            11: self.state_11,
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
            self.mqtt_broker,
            self.mqtt_port,
            self.mqtt_username,
            self.mqtt_password,
            logging_level,
        )
        self.subscriber = Subscriber(
            self.mqtt_broker,
            self.mqtt_port,
            self.mqtt_username,
            self.mqtt_password,
            logging_level,
        )
        self.personnel_id = None
        self.is_busy = False
        self.update_thread = None
        self.update_url = update_url
        self.update_save_dir = update_save_path
        self.update_queue = Queue()

    def _log_state(self, state_number):
        """Log state only if it has changed"""
        if state_number != self.last_logged_state:
            self._logger.info(f"State {state_number}")
            self.last_logged_state = state_number
        # If the state is not None or 1, the system is busy
        self.is_busy = state_number not in [None, 1]

    def _handle_update(self, updates):
        """Handle multiple update processes in parallel"""
        if self.is_busy:
            self._logger.info(f"System is busy, queueing updates")
            self.update_queue.put(updates)
            return True

        if self.update_thread and self.update_thread.is_alive():
            self._logger.warning("Update already in progress")
            return False

        def update_callback(success, results=None):
            if success:
                self._logger.info("All updates completed successfully")
            else:
                self._logger.error("Some updates failed")
                if results:
                    for update_type, result in results.items():
                        self._logger.info(
                            f"{update_type} update: {'Success' if result else 'Failed'}"
                        )
            self.update_thread = None

        # Create URLs dictionary
        download_urls = {
            update_type: f"{self.update_url}{filename}"
            for update_type, filename in updates.items()
        }

        self.update_thread = UpdateDownloader(
            download_urls,
            self.update_save_dir,
            update_callback,
        )
        self.update_thread.start()
        return True

    def _process_pending_updates(self):
        """Process any pending updates in the queue"""
        if not self.update_queue.empty() and not self.is_busy:
            updates = self.update_queue.get()
            self._logger.info(f"Processing queued updates")
            self._handle_update(updates)

    def state_1(self):
        self._log_state(1)
        self._process_pending_updates()
        if face_detected():
            self.current_status = 2  # move to state 2

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
            self.current_status = 3  # move to state 3
        else:
            self.current_status = 1  # return to state 1

    def state_3(self):
        self._log_state(3)
        if personnel_passed():
            close_gate(1)
            self.current_status = 4  # move to state 4

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
            self.current_status = None  # gate 2 should handle the rest (State 8 to 11)
        else:
            self.current_status = 7  # move to state 7

    def state_5(self):
        self._log_state(5)
        # no manual transition to other states, it will wait for the subscriber to change the state

    def state_6(self):
        self._log_state(6)
        alert_buzzer_and_led()
        # no manual transition to other states, it will wait for the subscriber to change the state

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
        self.current_status = 6  # move to state 6

    def state_8(self):
        self._log_state(8)
        if face_verified_with_id(self.personnel_id):
            self.current_status = 10  # move to state 10
        else:
            self.current_status = 9  # move to state 9

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
        self.current_status = None  # gate 1 should handle the rest (State 1 to 7)

    def state_10(self):
        self._log_state(10)
        if voice_verified(self.personnel_id):
            open_gate(2)
            self.publisher.publish(
                "gate_2/status",
                json.dumps({"opened": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}),
                2,
            )
            self.current_status = 11  # move to state 11
        else:
            self.current_status = 9  # move to state 9

    def state_11(self):
        self._log_state(11)
        if personnel_passed():
            close_gate(2)
            self.publisher.publish(
                "gate_2/status",
                json.dumps({"closed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}),
                2,
            )
            self.current_status = None  # gate 1 should handle the rest (State 1 to 7)
            self._log_state(None)
            self._process_pending_updates()

    def gate_1_subscribe_callback(self, topic, text_payload, raw_payload):
        """
        Callback function for the subscriber to handle messages from Gate 2 and the Dashboard
        """
        self._logger.info(f"Received message: {text_payload} from topic: {topic}")
        if text_payload is None:
            return
        # This will be used to handle command from the Dashboard to open/close the gate
        if topic == "command":
            if json.loads(text_payload)["action"] == "open":
                open_gate(1)
                self.current_status = 5  # move to state 5
            if json.loads(text_payload)["action"] == "close":
                close_gate(1)
                self.current_status = 1  # move to state 1

        # This will be used to tell gate 1 that gate 2 has let the personnel in
        # (Stete 11 to 1)
        if topic == "gate_2/status":
            payload = json.loads(text_payload)
            if "closed" in payload and payload["closed"] is not None:
                self.current_status = 1  # move to state 1

        # This will be used to tell gate 1 that gate 2 has detected an intruder that is different
        # from the personnel entered in gate 1 (State 9 to 6)
        if topic == "alert":
            if json.loads(text_payload)["type"] == "diff":
                self.current_status = 6  # move to state 6

        # Handle update command
        if topic == "update/embedding":
            try:
                payload = json.loads(text_payload)
                updates = {k: v for k, v in payload.items() if k in ["face", "voice"]}
                if updates:
                    self._handle_update(updates)
            except json.JSONDecodeError:
                self._logger.error("Invalid update command format")

    def gate_2_subscribe_callback(self, topic, text_payload, raw_payload):
        """
        Callback function for the subscriber to handle messages from Gate 1
        """
        self._logger.info(f"Received message: {text_payload} from topic: {topic}")
        if text_payload is None:
            return

        # This will be used to tell gate 2 that the mantrap has only one personnel and
        # can proceed to face verification (State 4 to 8)
        if topic == "verified":
            if json.loads(text_payload)["personnel_id"]:
                self.personnel_id = json.loads(text_payload)["personnel_id"]
                self._logger.info(f"Personnel ID: {self.personnel_id}")
                self.current_status = 8  # move to state 8

        # Handle update command
        if topic == "update/embedding":
            try:
                payload = json.loads(text_payload)
                updates = {k: v for k, v in payload.items() if k in ["face", "voice"]}
                if updates:
                    self._handle_update(updates)
            except json.JSONDecodeError:
                self._logger.error("Invalid update command format")

    def run(self):
        self.publisher.connect()
        self.subscriber.connect()
        self.subscriber.subscribe("update/embedding", 2)

        while not self.publisher.connected or not self.subscriber.connected:
            time.sleep(1)

        if self.type == "gate1":
            self.current_status = 1  # originally state 1
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
