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
        self._last_logged_state = None
        self._current_status = None
        self._status_action_dict = {
            1: self._state_1,
            2: self._state_2,
            3: self._state_3,
            4: self._state_4,
            5: self._state_5,
            6: self._state_6,
            7: self._state_7,
            8: self._state_8,
            9: self._state_9,
            10: self._state_10,
            11: self._state_11,
        }
        self._type = type
        self._mqtt_broker = mqtt_broker
        self._mqtt_port = mqtt_port
        self._mqtt_username = mqtt_username
        self._mqtt_password = mqtt_password

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

        self._publisher = Publisher(
            self._mqtt_broker,
            self._mqtt_port,
            self._mqtt_username,
            self._mqtt_password,
            logging_level,
        )
        self._subscriber = Subscriber(
            self._mqtt_broker,
            self._mqtt_port,
            self._mqtt_username,
            self._mqtt_password,
            logging_level,
        )
        self._personnel_id = None
        self._is_busy = False
        self._update_thread = None
        self._update_url = update_url
        self._update_save_dir = update_save_path
        self._update_queue = Queue()

    def _log_state(self, state_number):
        """Log state only if it has changed"""
        if state_number != self._last_logged_state:
            self._logger.info(f"State {state_number}")
            self._last_logged_state = state_number
        # If the state is not None or 1, the system is busy
        self._is_busy = state_number not in [None, 1]

    def _handle_update(self, updates):
        """Handle multiple update processes in parallel"""
        if self._is_busy:
            self._logger.info(f"System is busy, queueing updates")
            self._update_queue.put(updates)
            return True

        if self._update_thread and self._update_thread.is_alive():
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
            self._update_thread = None

        # Create URLs dictionary
        download_urls = {
            update_type: f"{self._update_url}{filename}"
            for update_type, filename in updates.items()
        }

        self._update_thread = UpdateDownloader(
            download_urls,
            self._update_save_dir,
            update_callback,
            self._logger.level,
        )
        self._update_thread.start()
        return True

    def _process_pending_updates(self):
        """Process any pending updates in the queue"""
        if not self._update_queue.empty() and not self._is_busy:
            updates = self._update_queue.get()
            self._logger.info(f"Processing queued updates")
            self._handle_update(updates)

    def _state_1(self):
        self._log_state(1)
        self._process_pending_updates()
        if face_detected():
            self._current_status = 2  # move to state 2

    def _state_2(self):
        self._log_state(2)
        self._personnel_id = face_verified()
        if self._personnel_id:
            open_gate(1)
            self._publisher.publish(
                "gate_1/status",
                json.dumps({"opened": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}),
                2,
            )
            self._current_status = 3  # move to state 3
        else:
            self._current_status = 1  # return to state 1

    def _state_3(self):
        self._log_state(3)
        if personnel_passed():
            close_gate(1)
            self._current_status = 4  # move to state 4

    def _state_4(self):
        self._log_state(4)
        if self._type == "gate2":
            return
        self._publisher.publish(
            "gate_1/status",
            json.dumps({"closed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}),
            2,
        )
        if mantrap_scan():
            self._publisher.publish(
                "verified", json.dumps({"personnel_id": self._personnel_id}), 2
            )
            self._current_status = None  # gate 2 should handle the rest (State 8 to 11)
        else:
            self._current_status = 7  # move to state 7

    def _state_5(self):
        self._log_state(5)
        # no manual transition to other states, it will wait for the subscriber to change the state

    def _state_6(self):
        self._log_state(6)
        alert_buzzer_and_led()
        # no manual transition to other states, it will wait for the subscriber to change the state

    def _state_7(self):
        self._log_state(7)
        buffer = capture_intruder()
        self._publisher.publish(
            "alert",
            json.dumps(
                {
                    "type": "multi",
                    "image": buffer,
                }
            ),
            2,
        )
        self._current_status = 6  # move to state 6

    def _state_8(self):
        self._log_state(8)
        if face_verified_with_id(self._personnel_id):
            self._current_status = 10  # move to state 10
        else:
            self._current_status = 9  # move to state 9

    def _state_9(self):
        self._log_state(9)
        buffer = capture_intruder()
        self._publisher.publish(
            "alert",
            json.dumps(
                {
                    "type": "diff",
                    "image": buffer,
                }
            ),
            2,
        )
        self._current_status = None  # gate 1 should handle the rest (State 1 to 7)

    def _state_10(self):
        self._log_state(10)
        if voice_verified(self._personnel_id):
            open_gate(2)
            self._publisher.publish(
                "gate_2/status",
                json.dumps({"opened": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}),
                2,
            )
            self._current_status = 11  # move to state 11
        else:
            self._current_status = 9  # move to state 9

    def _state_11(self):
        self._log_state(11)
        if personnel_passed():
            close_gate(2)
            self._publisher.publish(
                "gate_2/status",
                json.dumps({"closed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}),
                2,
            )
            self._current_status = None  # gate 1 should handle the rest (State 1 to 7)
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
                self._current_status = 5  # move to state 5
            if json.loads(text_payload)["action"] == "close":
                close_gate(1)
                self._current_status = 1  # move to state 1

        # This will be used to tell gate 1 that gate 2 has let the personnel in
        # (Stete 11 to 1)
        if topic == "gate_2/status":
            payload = json.loads(text_payload)
            if "closed" in payload and payload["closed"] is not None:
                self._current_status = 1  # move to state 1

        # This will be used to tell gate 1 that gate 2 has detected an intruder that is different
        # from the personnel entered in gate 1 (State 9 to 6)
        if topic == "alert":
            if json.loads(text_payload)["type"] == "diff":
                self._current_status = 6  # move to state 6

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
                self._personnel_id = json.loads(text_payload)["personnel_id"]
                self._logger.info(f"Personnel ID: {self._personnel_id}")
                self._current_status = 8  # move to state 8

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
        self._publisher.connect()
        self._subscriber.connect()
        self._subscriber.subscribe("update/embedding", 2)

        while not self._publisher.connected or not self._subscriber.connected:
            time.sleep(1)

        if self._type == "gate1":
            self._current_status = 1  # originally state 1
            self._subscriber.on_message_callback = self.gate_1_subscribe_callback
            self._subscriber.subscribe("gate_2/status", 2)
            self._subscriber.subscribe("command", 2)
            self._subscriber.subscribe("alert", 2)

        if self._type == "gate2":
            self._current_status = None  # wait for the message from gate 1
            self._subscriber.on_message_callback = self.gate_2_subscribe_callback
            self._subscriber.subscribe("verified", 2)

        while True:
            if self._current_status is not None:
                self._status_action_dict[self._current_status]()
