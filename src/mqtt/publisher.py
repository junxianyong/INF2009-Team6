import logging
import paho.mqtt.client as mqtt


class Publisher:
    """
    A MQTT Publisher client that handles connections and message publishing.

    This class provides functionality to connect to an MQTT broker and publish
    messages to specific topics with configurable QoS levels.

    Attributes:
        broker (str): The MQTT broker address
        port (int): The port number for the MQTT connection
        connected (bool): Connection status flag
        client (mqtt.Client): The MQTT client instance
        _logger: Logger instance for this class
    """

    def __init__(
        self,
        broker,
        port=1883,
        username=None,
        password=None,
        logging_level=logging.INFO,
    ):
        """
        Initialize the MQTT Publisher.

        Args:
            broker (str): MQTT broker address
            port (int, optional): Broker port number. Defaults to 1883.
            username (str, optional): Authentication username. Defaults to None.
            password (str, optional): Authentication password. Defaults to None.
            logging_level: The logging level to use. Defaults to logging.INFO.
        """
        self.broker = broker
        self.port = port
        self.connected = False
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if username and password:
            self.client.username_pw_set(username, password)
        self.client.on_connect = self._on_connect
        self.client.on_publish = self._on_publish

        # Configure logging
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging_level)

        # Check if handler already exists to prevent double logging
        if not self._logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        # Set propagate to False to prevent double logging when this logger is a child of another logger
        self._logger.propagate = False

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """
        Callback for when the client connects to the broker.
        """
        if reason_code == 0:
            self._logger.info("Connected to MQTT broker")
            self.connected = True
        else:
            self._logger.error(f"Failed to connect, return code: {reason_code}")
            if reason_code == 5:  # Authentication failed
                self._logger.error(
                    "Authentication failed - check username and password"
                )

    def _on_publish(self, client, userdata, mid, reason_code, properties):
        """
        Callback for when a message is published.
        """
        self._logger.debug(f"Message published with ID: {mid}")

    def connect(self):
        """
        Connect to the MQTT broker.
        """
        try:
            self._logger.info(f"Connecting to MQTT broker {self.broker}:{self.port}")
            self.client.connect(self.broker, self.port)
            self.client.loop_start()
        except Exception as e:
            self._logger.exception(f"Connection failed: {e}")
            return False
        return True

    def publish(self, topic, message, qos=0):
        """
        Publish a message to a topic on the MQTT broker.
        """
        try:
            self._logger.info(f"Publishing to {topic} with QoS {qos}: {message}")
            result = self.client.publish(topic, message, qos)
            result.wait_for_publish()
            return result.is_published()
        except Exception as e:
            self._logger.exception(f"Publishing failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from the MQTT broker and stop the network loop."""
        self._logger.info("Disconnecting from MQTT broker")
        self.client.loop_stop()
        self.client.disconnect()
