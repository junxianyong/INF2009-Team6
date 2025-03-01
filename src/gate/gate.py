from gate.gate_types import GateType
from gate.state_manager import StateManager
from gate.states import GateState
from mqtt.publisher import Publisher
from mqtt.subscriber import Subscriber
import logging
import time
import json
from gate.update import UpdateDownloader
from queue import Queue
from gate.example import *


class Gate:
    def __init__(
        self,
        gate_type,
        mqtt_broker,
        mqtt_port,
        mqtt_username,
        mqtt_password,
        update_url,
        update_save_path,
        logging_level,
    ):
        self._last_logged_state = None
        self.gate_type = gate_type
        self._mqtt_broker = mqtt_broker
        self._mqtt_port = mqtt_port
        self._mqtt_username = mqtt_username
        self._mqtt_password = mqtt_password

        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

        # Add console handler if none exists
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.publisher = Publisher(
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
        self.personnel_id = None
        self.is_busy = False
        self._update_thread = None
        self._update_url = update_url
        self._update_save_dir = update_save_path
        self._update_queue = Queue()
        self.state_manager = StateManager(self)

    def _log_state(self, state_number):
        """Log state only if it has changed"""
        if state_number != self._last_logged_state:
            self.logger.info(f"State {state_number}")
            self._last_logged_state = state_number
        # If the state is not None or 1, the system is busy
        self.is_busy = state_number not in [None, 1]

    def _handle_update(self, updates):
        """Handle multiple update processes in parallel"""
        if self.is_busy:
            self.logger.info(f"System is busy, queueing updates")
            self._update_queue.put(updates)
            return True

        if self._update_thread and self._update_thread.is_alive():
            self.logger.warning("Update already in progress")
            return False

        def update_callback(success, results=None):
            if success:
                self.logger.info("All updates completed successfully")
            else:
                self.logger.error("Some updates failed")
                if results:
                    for update_type, result in results.items():
                        self.logger.info(
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
            self.logger.level,
        )
        self._update_thread.start()
        return True

    def process_pending_updates(self):
        """Process any pending updates in the queue"""
        if not self._update_queue.empty() and not self.is_busy:
            updates = self._update_queue.get()
            self.logger.info(f"Processing queued updates")
            self._handle_update(updates)

    def gate_1_subscribe_callback(self, topic, text_payload, raw_payload):
        """
        Callback function for the subscriber to handle messages from Gate 2 and the Dashboard
        """
        self.logger.info(f"Received message: {text_payload} from topic: {topic}")
        if text_payload is None:
            return
        # This will be used to handle command from the Dashboard to open/close the gate
        if topic == "command":
            if json.loads(text_payload)["action"] == "open":
                open_gate(1)
                self.state_manager.current_state = (
                    GateState.MANUAL_OPEN
                )  # move to state 5
            if json.loads(text_payload)["action"] == "close":
                close_gate(1)
                self.state_manager.current_state = (
                    GateState.WAITING_FOR_FACE
                )  # move to state 1

        # This will be used to tell gate 1 that gate 2 has let the personnel in
        # (Stete 11 to 1)
        if topic == "gate_2/status":
            payload = json.loads(text_payload)
            if "closed" in payload and payload["closed"] is not None:
                self.state_manager.current_state = (
                    GateState.WAITING_FOR_FACE
                )  # move to state 1

        # This will be used to tell gate 1 that gate 2 has detected an intruder that is different
        # from the personnel entered in gate 1 (State 9 to 6)
        if topic == "alert":
            if json.loads(text_payload)["type"] == "diff":
                self.state_manager.current_state = (
                    GateState.ALERT_ACTIVE
                )  # move to state 6

        # Handle update command
        if topic == "update/embedding":
            try:
                payload = json.loads(text_payload)
                updates = {k: v for k, v in payload.items() if k in ["face", "voice"]}
                if updates:
                    self._handle_update(updates)
            except json.JSONDecodeError:
                self.logger.error("Invalid update command format")

    def gate_2_subscribe_callback(self, topic, text_payload, raw_payload):
        """
        Callback function for the subscriber to handle messages from Gate 1
        """
        self.logger.info(f"Received message: {text_payload} from topic: {topic}")
        if text_payload is None:
            return

        # This will be used to tell gate 2 that the mantrap has only one personnel and
        # can proceed to face verification (State 4 to 8)
        if topic == "verified":
            if json.loads(text_payload)["personnel_id"]:
                self.personnel_id = json.loads(text_payload)["personnel_id"]
                self.logger.info(f"Personnel ID: {self.personnel_id}")
                self.state_manager.current_state = (
                    GateState.VERIFYING_FACE_G2
                )  # move to state 8

        # Handle update command
        if topic == "update/embedding":
            try:
                payload = json.loads(text_payload)
                updates = {k: v for k, v in payload.items() if k in ["face", "voice"]}
                if updates:
                    self._handle_update(updates)
            except json.JSONDecodeError:
                self.logger.error("Invalid update command format")

    def run(self):
        self.publisher.connect()
        self._subscriber.connect()
        self._subscriber.subscribe("update/embedding", 2)

        while not self.publisher.connected or not self._subscriber.connected:
            time.sleep(1)

        if self.gate_type == GateType.GATE1:
            self.state_manager.current_state = GateState.WAITING_FOR_FACE
            self._subscriber.on_message_callback = self.gate_1_subscribe_callback
            self._subscriber.subscribe("gate_2/status", 2)
            self._subscriber.subscribe("command", 2)
            self._subscriber.subscribe("alert", 2)

        if self.gate_type == GateType.GATE2:
            self.state_manager.current_state = GateState.IDLE
            self._subscriber.on_message_callback = self.gate_2_subscribe_callback
            self._subscriber.subscribe("verified", 2)

        while True:
            self.state_manager.process_current_state()
