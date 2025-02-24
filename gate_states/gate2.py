import sys
import paho.mqtt.client as paho
import threading
import json
import time



class gate2:
    gate_status={
        "gate1_voice":False,
        "gate1_face":False,
        "gate1_ultrasonic":False,
        "gate1_is_door_open":False,
        "gate2_voice":False,
        "gate2_face":False,
        "gate2_ultrasonic":False,
        "gate2_is_door_open":False,
    }

    def __init__(self, broker="localhost", port=1883):
        self.message = ""
        self.broker = broker
        self.port = port
        self.client = paho.Client()
        self.client.on_message = self.on_message

    def connect(self):
        if self.client.connect(self.broker, self.port, 60) != 0:
            print("Cannot connect to MQTT Broker")
            sys.exit(-1)

    def subscribe(self, topic):
        self.client.subscribe(topic)

    def publish(self, topic, payload, qos=0):
        self.client.publish(topic, payload, qos)

    def start(self):
        try:
            print("Press Ctrl+C to exit")
            self.client.loop_forever()
        except KeyboardInterrupt:
            print("Disconnecting...")
            self.client.disconnect()

    def setup(self):
        self.connect()
        self.subscribe("fogToGate2/command")
        self.client.loop_start()
        # self.on_message = gate_2.client.on_message
    
    def sending_msg(self):
        while(True):
            print("Gate 2: Enter command")
            user_input = input()

            if user_input == "1":
                self.gate_status["gate2_voice"] = not self.gate_status["gate2_voice"]
                print("Gate 2 voice is: "+str(self.gate_status["gate2_voice"]))
                gate_status_msg = json.dumps(self.gate_status)
                gate_2.publish("gate2ToFog/command", gate_status_msg)
            elif user_input == "2":
                self.gate_status["gate2_face"] = not self.gate_status["gate2_face"]
                print("Gate 2 face is: "+str(self.gate_status["gate2_face"]))
                gate_status_msg = json.dumps(self.gate_status)
                gate_2.publish("gate2ToFog/command", gate_status_msg)
            elif user_input == "3":
                self.gate_status["gate2_ultrasonic"] = not self.gate_status["gate2_ultrasonic"]
                print("Gate 2 ultrasonic is: "+str(self.gate_status["gate2_ultrasonic"]))
                gate_status_msg = json.dumps(self.gate_status)
                gate_2.publish("gate2ToFog/command", gate_status_msg)
            elif user_input == "4":
                self.gate_status["gate2_is_door_open"] = not self.gate_status["gate2_is_door_open"]
                if(self.gate_status["gate2_is_door_open"]==True):
                    print("Gate 2 open")
                else:
                    print("Gate 2 closed")
            elif user_input == "send":
                print(self.gate_status)
                gate_status_msg = json.dumps(self.gate_status)
                gate_2.publish("gate2ToFog/command", gate_status_msg)
            
            if self.gate_status["gate2_voice"] == True and self.gate_status["gate2_face"] == True:
                self.gate_status["gate2_is_door_open"] = True
                print("Gate 2 is opening")
                self.gate_status["gate2_voice"] = False
                self.gate_status["gate2_face"] = False
                gate_status_msg = json.dumps(self.gate_status)
                gate_2.publish("gate2ToFog/command", gate_status_msg)

            if self.gate_status["gate2_ultrasonic"] == True and self.gate_status["gate2_is_door_open"] == True:
                self.gate_status["gate2_is_door_open"] = False

                gate_status_msg = json.dumps(self.gate_status)
                gate_2.publish("gate2ToFog/command", gate_status_msg)
                time.sleep(5)
                print("Gate 2 is closing")
                self.gate_status["gate2_ultrasonic"] = False
                gate_status_msg = json.dumps(self.gate_status)
                gate_2.publish("gate2ToFog/command", gate_status_msg)

    def on_message(self, client, userdata, msg):
        # print(f"Sent from {msg.topic}: {msg.payload.decode()}")
        self.gate_status = json.loads(msg.payload.decode())
        print(self.gate_status)

if __name__ == "__main__":
    gate_2 = gate2()
    gate_2.setup()
    
    sending_msg_thread = threading.Thread(target=gate_2.sending_msg,args=())

    sending_msg_thread.start()
    
    sending_msg_thread.join()
        
    # gate_2.client.disconnect()
