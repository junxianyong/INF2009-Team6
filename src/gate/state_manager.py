from gate.enum.states import GateState
from gate.enum.gate_types import GateType
from gate.example import *
import json
from datetime import datetime
from utils.logger_mixin import LoggerMixin


class StateManager(LoggerMixin):
    def __init__(self, gate):
        self.gate = gate
        self._last_logged_state = None
        self._current_state = None
        self.logger = self._setup_logger(__name__, self.gate.logger.level)

        # Map states to handler methods
        self._state_handlers = {
            GateState.WAITING_FOR_FACE: self._handle_waiting_for_face,
            GateState.VERIFYING_FACE: self._handle_verifying_face,
            GateState.WAITING_FOR_PASSAGE_G1: self._handle_waiting_for_passage_g1,
            GateState.CHECKING_MANTRAP: self._handle_checking_mantrap,
            GateState.MANUAL_OPEN: self._handle_manual_open,
            GateState.ALERT_ACTIVE: self._handle_alert_active,
            GateState.CAPTURE_INTRUDER: self._handle_capture_intruder,
            GateState.VERIFYING_FACE_G2: self._handle_verifying_face_g2,
            GateState.CAPTURE_MISMATCH: self._handle_capture_mismatch,
            GateState.VERIFYING_VOICE: self._handle_verifying_voice,
            GateState.WAITING_FOR_PASSAGE_G2: self._handle_waiting_for_passage_g2,
        }

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, state):
        self._current_state = state
        self._log_state_change(state)

    def _log_state_change(self, state):
        """Log state only if it has changed"""
        if state != self._last_logged_state:
            self.logger.info(f"State changed to: {state.name}")
            self._last_logged_state = state

        # If the state is not IDLE or WAITING_FOR_FACE, the system is busy
        self.gate.is_busy = state not in [
            None,
            GateState.IDLE,
            GateState.WAITING_FOR_FACE,
        ]

    def process_current_state(self):
        """Process the current state using the appropriate handler"""
        if (
            self._current_state is not None
            and self._current_state in self._state_handlers
        ):
            self._state_handlers[self._current_state]()

    def _handle_waiting_for_face(self):
        self.gate.process_pending_updates()
        if self.gate.motion_detector.wait_for_motion():
            self.current_state = GateState.VERIFYING_FACE

    def _handle_verifying_face(self):
        personnel_id = self.gate.face_verification.wait_for_face_and_verify()
        if personnel_id:
            self.gate.personnel_id = personnel_id
            open_gate(1)
            self.gate.publisher.publish(
                "gate_1/status",
                json.dumps({"opened": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}),
                2,
            )
            self.current_state = GateState.WAITING_FOR_PASSAGE_G1
        else:
            self.current_state = GateState.WAITING_FOR_FACE

    def _handle_waiting_for_passage_g1(self):
        if personnel_passed():
            close_gate(1)
            self.current_state = GateState.CHECKING_MANTRAP

    def _handle_checking_mantrap(self):
        if self.gate.gate_type == GateType.GATE2:
            return

        self.gate.publisher.publish(
            "gate_1/status",
            json.dumps({"closed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}),
            2,
        )

        if mantrap_scan():
            self.gate.publisher.publish(
                "verified", json.dumps({"personnel_id": self.gate.personnel_id}), 2
            )
            self.current_state = GateState.IDLE  # gate 2 takes over
        else:
            self.current_state = GateState.CAPTURE_INTRUDER

    def _handle_manual_open(self):
        # No transitions - waits for command callback to change state
        pass

    def _handle_alert_active(self):
        alert_buzzer_and_led()
        # No transitions - waits for command callback to change state

    def _handle_capture_intruder(self):
        buffer = capture_intruder()
        self.gate.publisher.publish(
            "alert",
            json.dumps(
                {
                    "type": "multi",
                    "image": buffer,
                }
            ),
            2,
        )
        self.current_state = GateState.ALERT_ACTIVE

    def _handle_verifying_face_g2(self):
        personnel_id = self.gate.face_verification.wait_for_face_and_verify()
        self.logger.debug(
            f"Gate 2 personnel ID: {personnel_id} vs Gate 1 personnel ID {self.gate.personnel_id} is {personnel_id == self.gate.personnel_id}"
        )
        if personnel_id == self.gate.personnel_id:
            self.current_state = GateState.VERIFYING_VOICE
        else:
            self.current_state = GateState.CAPTURE_MISMATCH

    def _handle_capture_mismatch(self):
        buffer = capture_intruder()
        self.gate.publisher.publish(
            "alert",
            json.dumps(
                {
                    "type": "diff",
                    "image": buffer,
                }
            ),
            2,
        )
        self.current_state = GateState.IDLE  # gate 1 takes over

    def _handle_verifying_voice(self):
        if self.gate.voice_auth.authenticate_user(self.gate.personnel_id):
            open_gate(2)
            self.gate.publisher.publish(
                "gate_2/status",
                json.dumps({"opened": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}),
                2,
            )
            self.current_state = GateState.WAITING_FOR_PASSAGE_G2
        else:
            self.current_state = GateState.CAPTURE_MISMATCH

    def _handle_waiting_for_passage_g2(self):
        if personnel_passed():
            close_gate(2)
            self.gate.publisher.publish(
                "gate_2/status",
                json.dumps({"closed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}),
                2,
            )
            self.current_state = GateState.IDLE
            self.gate.process_pending_updates()
