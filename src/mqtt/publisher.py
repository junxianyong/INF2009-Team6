from time import sleep
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
    """

    def __init__(self, client_id, broker, port=1883, username=None, password=None):
        """
        Initialize the MQTT Publisher.

        Args:
            client_id (str): Unique identifier for the client
            broker (str): MQTT broker address
            port (int, optional): Broker port number. Defaults to 1883.
            username (str, optional): Authentication username. Defaults to None.
            password (str, optional): Authentication password. Defaults to None.
        """
        self.broker = broker
        self.port = port
        self.connected = False
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if username and password:
            self.client.username_pw_set(username, password)
        self.client.on_connect = self._on_connect
        self.client.on_publish = self._on_publish

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
            print("Connected to MQTT broker")
            self.connected = True
        else:
            print(f"Failed to connect, return code: {reason_code}")
            if reason_code == 5:  # Authentication failed
                print("Authentication failed - check username and password")

    def _on_publish(self, client, userdata, mid, reason_code, properties):
        """
        Callback for when a message is published.

        Args:
            client: The client instance
            userdata: User data of any type
            mid: Message ID
            reason_code: Publish result code
            properties: Properties from the PUBACK packet
        """
        print(f"Message published")

    def connect(self):
        """
        Connect to the MQTT broker.

        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            self.client.connect(self.broker, self.port)
            self.client.loop_start()
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
        return True

    def publish(self, topic, message, qos=0):
        """
        Publish a message to a topic on the MQTT broker.

        Args:
            topic (str): The topic to publish to
            message (str): The message to publish
            qos (int, optional): Quality of Service level. Defaults to 0.

        Returns:
            bool: True if publishing is successful, False otherwise
        """
        try:
            result = self.client.publish(topic, message, qos)
            result.wait_for_publish()
            return result.is_published()
        except Exception as e:
            print(f"Publishing failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from the MQTT broker and stop the network loop."""
        self.client.loop_stop()
        self.client.disconnect()


if __name__ == "__main__":
    publisher = Publisher(
        client_id="publisher",
        broker="test.mosquitto.org",
        username="rw",
        password="readwrite",
        port=1884,
    )
    publisher.connect()
    while not publisher.connected:
        pass
    for i in range(10):
        sleep(1)
        publisher.publish("test", f"Hello, world! {i}", 3)
    publisher.disconnect()
