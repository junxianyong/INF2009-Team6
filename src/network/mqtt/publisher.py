import logging
import paho.mqtt.client as mqtt
from utils.logger_mixin import LoggerMixin


class Publisher(LoggerMixin):
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
        self._broker = broker
        self._port = port
        self._connected = False
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if username and password:
            self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_publish = self._on_publish
        self._logger = self._setup_logger(__name__, logging_level)

    @property
    def connected(self):
        return self._connected

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """
        Callback for when the client connects to the broker.
        """
        if reason_code == 0:
            self._logger.info("Connected to MQTT broker")
            self._connected = True
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
            self._logger.info(f"Connecting to MQTT broker {self._broker}:{self._port}")
            self._client.connect(self._broker, self._port)
            self._client.loop_start()
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
            result = self._client.publish(topic, message, qos)
            result.wait_for_publish()
            return result.is_published()
        except Exception as e:
            self._logger.exception(f"Publishing failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from the MQTT broker and stop the network loop."""
        self._logger.info("Disconnecting from MQTT broker")
        self._client.loop_stop()
        self._client.disconnect()
