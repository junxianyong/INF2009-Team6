import RPi.GPIO as GPIO
import time

SERVO_PIN = 14  

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)  

def set_angle(angle):
    duty_cycle = 2.0 + (angle / 18.0)  
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)
    
def set_door_open():
    set_angle(180)

def set_door_close():
    set_angle(0)

try:
    while True:
        userInput = input("Door state 'open' or 'close': ")
        if userInput == "open":
            set_door_open()
        elif userInput == "close":
            set_door_close()
            

except KeyboardInterrupt:
    pass
finally:
    pwm.stop()
    GPIO.cleanup()
