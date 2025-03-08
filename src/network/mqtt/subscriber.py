from time import sleep
import paho.mqtt.client as mqtt
import logging
from utils.logger_mixin import LoggerMixin


class Subscriber(LoggerMixin):
    """
    A MQTT Subscriber client that handles connections and message subscriptions.

    This class provides functionality to connect to an MQTT broker, subscribe to topics,
    and process received messages either through a default handler or a custom callback.

    Attributes:
        broker (str): The MQTT broker address
        port (int): The port number for the MQTT connection
        connected (bool): Connection status flag
        client (mqtt.Client): The MQTT client instance
        on_message_callback (callable): Optional callback for message handling
        _logger: Logger instance for this class
    """

    def __init__(
        self, mqtt_config, on_message_callback=None, logging_level=logging.INFO
    ):
        """
        Initialize the MQTT Subscriber.

        Args:
            mqtt_config: A dictionary containing the MQTT configuration parameters.
            on_message_callback (callable, optional): Custom message handler. Defaults to None.
            logging_level: The logging level to use. Defaults to logging.INFO.
        """
        self._broker = mqtt_config["broker"]
        self._port = mqtt_config["port"]
        self._connected = False
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if "username" and "password" in mqtt_config:
            self._client.username_pw_set(
                mqtt_config["username"], mqtt_config["password"]
            )
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_subscribe = self._on_subscribe
        self.on_message_callback = on_message_callback
        self._logger = self._setup_logger(__name__, logging_level)

    @property
    def connected(self):
        return self._connected

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """
        Callback for when the client connects to the broker.

        Args:
            client: The client instance
            userdata: User data of any type
            flags: Response flags sent by the broker
            reason_code: Connection result code
            properties: Properties from the connection response
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

    def _on_message(self, client, userdata, message):
        """
        Callback for when a message is received from the broker.

        Args:
            client: The client instance
            userdata: User data of any type
            message: The received message instance
        """
        if self.on_message_callback:
            try:
                decoded_payload = message.payload.decode()
                self._logger.debug(
                    f"Received message on topic {message.topic}: {decoded_payload}"
                )
            except UnicodeDecodeError:
                # If decode fails, pass the raw bytes
                decoded_payload = None
                self._logger.debug(f"Received binary message on topic {message.topic}")
            self.on_message_callback(message.topic, decoded_payload, message.payload)
        else:
            try:
                self._logger.info(
                    f"Received message on topic {message.topic}: {message.payload.decode()}"
                )
            except UnicodeDecodeError:
                self._logger.info(
                    f"Received binary message on topic {message.topic}: {message.payload!r}"
                )

    def _on_subscribe(self, client, userdata, mid, reason_codes, properties):
        """
        Callback for when the client subscribes to a topic.

        Args:
            client: The client instance
            userdata: User data of any type
            mid: Message ID
            reason_codes: List of reason codes for each topic filter
            properties: Properties from the SUBACK packet
        """
        self._logger.info(f"Subscribed with message ID: {mid}")

    def connect(self):
        """
        Connect to the MQTT broker.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            self._logger.info(f"Connecting to MQTT broker {self._broker}:{self._port}")
            self._client.connect(self._broker, self._port)
            self._client.loop_start()
        except Exception as e:
            self._logger.exception(f"Connection failed: {e}")
            return False
        return True

    def subscribe(self, topic, qos=0):
        """
        Subscribe to a topic on the MQTT broker.

        Args:
            topic (str): The topic to subscribe to
            qos (int, optional): Quality of Service level. Defaults to 0.

        Returns:
            bool: True if subscription is successful, False otherwise
        """
        try:
            self._logger.info(f"Subscribing to topic {topic} with QoS {qos}")
            self._client.subscribe(topic, qos)
        except Exception as e:
            self._logger.exception(f"Subscription failed: {e}")
            return False
        return True

    def disconnect(self):
        """Disconnect from the MQTT broker and stop the network loop."""
        self._logger.info("Disconnecting from MQTT broker")
        self._client.loop_stop()
        self._client.disconnect()


if __name__ == "__main__":
    # Set up logging for the main module
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Example usage with custom callback
    def custom_message_handler(topic, text_payload, raw_payload):
        if text_payload is not None:
            logger.debug(
                f"Custom handler received: Topic={topic}, Message={text_payload}"
            )
        else:
            logger.debug(
                f"Custom handler received binary data: Topic={topic}, Raw={raw_payload!r}"
            )

    subscriber = Subscriber(
        broker="localhost",
        port=1883,
        username="mosquitto",
        password="mosquitto",
        on_message_callback=custom_message_handler,
    )
    subscriber.connect()
    while not subscriber.connected:
        pass
    subscriber.subscribe("test", 2)
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        subscriber.disconnect()
