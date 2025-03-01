from gate.event_manager import GateCallbackHandler
from gate.enum.gate_types import GateType
from gate.state_manager import StateManager
from gate.enum.states import GateState
from gate.update_manager import UpdateManager
from network.mqtt.publisher import Publisher
from network.mqtt.subscriber import Subscriber
import logging
import time


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
        self.state_manager = StateManager(self)
        self.update_manager = UpdateManager(self, update_url, update_save_path)
        self.callback_handler = GateCallbackHandler(self)

    def _log_state(self, state_number):
        """Log state only if it has changed"""
        if state_number != self._last_logged_state:
            self.logger.info(f"State {state_number}")
            self._last_logged_state = state_number
        # If the state is not None or 1, the system is busy
        self.is_busy = state_number not in [None, 1]

    def process_pending_updates(self):
        """Delegate to update manager"""
        self.update_manager.process_pending_updates()

    def run(self):
        self.publisher.connect()
        self._subscriber.connect()
        self._subscriber.subscribe("update/embedding", 2)

        while not self.publisher.connected or not self._subscriber.connected:
            time.sleep(1)

        if self.gate_type == GateType.GATE1:
            self.state_manager.current_state = GateState.WAITING_FOR_FACE
            self._subscriber.on_message_callback = (
                self.callback_handler.handle_gate1_message
            )
            self._subscriber.subscribe("gate_2/status", 2)
            self._subscriber.subscribe("command", 2)
            self._subscriber.subscribe("alert", 2)

        if self.gate_type == GateType.GATE2:
            self.state_manager.current_state = GateState.IDLE
            self._subscriber.on_message_callback = (
                self.callback_handler.handle_gate2_message
            )
            self._subscriber.subscribe("verified", 2)

        while True:
            self.state_manager.process_current_state()
