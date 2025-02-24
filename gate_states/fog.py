from flask import Flask, jsonify, render_template
import threading
import json
import sys
import time
import paho.mqtt.client as paho

app = Flask(__name__)

gate_status = {
    "gate1_voice": False,
    "gate1_face": False,
    "gate1_ultrasonic": False,
    "gate1_is_door_open": False,
    "gate2_voice": False,
    "gate2_face": False,
    "gate2_ultrasonic": False,
    "gate2_is_door_open": False,
}

class Fog:
    def __init__(self, broker="localhost", port=1883):
        self.broker = broker
        self.port = port
        self.client = paho.Client()
        self.client.on_message = self.on_message

    def connect(self):
        if self.client.connect(self.broker, self.port, 60) != 0:
            print("Cannot connect to MQTT Broker")
            sys.exit(-1)

    def subscribe(self):
        self.client.subscribe([("gate1ToFog/command", 0), ("gate2ToFog/command", 0)])

    def setup(self):
        self.connect()
        self.subscribe()
        self.client.loop_start()  

    def on_message(self, client, userdata, msg):
        global gate_status
        try:
            status = json.loads(msg.payload.decode())
            gate_status = status
            print("Updated gate status:", gate_status)
            
            gate_status_msg = json.dumps(gate_status)
            
            self.publish("fogToGate1/command", gate_status_msg)
            self.publish("fogToGate2/command", gate_status_msg)
        except Exception as e:
            print("Error processing MQTT message:", e)

    def publish(self, topic, payload, qos=0):
        self.client.publish(topic, payload, qos)

def mqtt_loop():
    fog_comp = Fog()
    fog_comp.setup()
    while True:
        time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    return jsonify(gate_status)

if __name__ == '__main__':
    mqtt_thread = threading.Thread(target=mqtt_loop)
    mqtt_thread.daemon = True
    mqtt_thread.start()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
