import json
from gate.enum.states import GateState
from gate.example import open_gate, close_gate


class GateCallbackHandler:
    def __init__(self, gate):
        self.gate = gate

    def handle_gate1_message(self, topic, text_payload, raw_payload):
        """Callback for Gate 1 to handle messages from Gate 2 and Dashboard"""
        self.gate.logger.info(f"Received message: {text_payload} from topic: {topic}")
        if text_payload is None:
            return

        if topic == "command":
            self._handle_command(json.loads(text_payload))
        elif topic == "gate_2/status":
            self._handle_gate2_status(json.loads(text_payload))
        elif topic == "alert":
            self._handle_alert(json.loads(text_payload))
        elif topic == "update/embedding":
            self._handle_update_embedding(text_payload)

    def handle_gate2_message(self, topic, text_payload, raw_payload):
        """Callback for Gate 2 to handle messages from Gate 1"""
        self.gate.logger.info(f"Received message: {text_payload} from topic: {topic}")
        if text_payload is None:
            return

        if topic == "verified":
            self._handle_verified(json.loads(text_payload))
        elif topic == "update/embedding":
            self._handle_update_embedding(text_payload)

    def _handle_command(self, payload):
        if payload["action"] == "open":
            open_gate(1)
            self.gate.state_manager.current_state = GateState.MANUAL_OPEN
        elif payload["action"] == "close":
            close_gate(1)
            self.gate.state_manager.current_state = GateState.WAITING_FOR_FACE

    def _handle_gate2_status(self, payload):
        if "closed" in payload and payload["closed"] is not None:
            self.gate.state_manager.current_state = GateState.WAITING_FOR_FACE

    def _handle_alert(self, payload):
        if payload["type"] == "diff":
            self.gate.state_manager.current_state = GateState.ALERT_ACTIVE

    def _handle_verified(self, payload):
        if payload["personnel_id"]:
            self.gate.personnel_id = payload["personnel_id"]
            self.gate.logger.info(f"Personnel ID: {self.gate.personnel_id}")
            self.gate.state_manager.current_state = GateState.VERIFYING_FACE_G2

    def _handle_update_embedding(self, text_payload):
        try:
            payload = json.loads(text_payload)
            updates = {k: v for k, v in payload.items() if k in ["face", "voice"]}
            if updates:
                self.gate.update_manager.handle_update(updates)
        except json.JSONDecodeError:
            self.gate.logger.error("Invalid update command format")
