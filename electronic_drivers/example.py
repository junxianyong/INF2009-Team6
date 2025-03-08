import RPi.GPIO as GPIO
import I2C_LCD_driver
from time import *
import RPi.GPIO as GPIO
from time import sleep
import threading


SERVO_PIN = 14  

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0) 

#Disable warnings (optional)
GPIO.setwarnings(False)
#Select GPIO mode
GPIO.setmode(GPIO.BCM)
#Set buzzer - pin 23 as output
buzzer=27
GPIO.setup(buzzer,GPIO.OUT)

mylcd = I2C_LCD_driver.lcd()

isDoorOpening = False

def buzzerOn():
    GPIO.output(buzzer,GPIO.HIGH)

def buzzerOff():
    GPIO.output(buzzer,GPIO.LOW)
    
def buzzerControl():
    while True:
        if isDoorOpening == True:
            GPIO.output(buzzer, GPIO.HIGH)
            #print("Beep")
            sleep(0.5)
            GPIO.output(buzzer, GPIO.LOW)
            #print("No Beep")
            sleep(0.5)    
            

def displayText(text):
    mylcd.lcd_display_string(text, 1)

def set_angle(angle):
    duty_cycle = 2.0 + (angle / 18.0)  
    pwm.ChangeDutyCycle(duty_cycle)
    sleep(0.5)
    pwm.ChangeDutyCycle(0)
    
def set_door_open():
    set_angle(180)

def set_door_close():
    set_angle(0)



buzzer_thread = threading.Thread(target=buzzerControl, daemon=True)
buzzer_thread.start()
try:


    while True:
        userInput = input("Door state 'open' or 'close': ")
        if userInput == "open":
            set_door_open()
            displayText("open")
            isDoorOpening = True
            
        elif userInput == "close":
            set_door_close()
            displayText("close")
            isDoorOpening = False
            

except KeyboardInterrupt:
    pass
finally:
    pwm.stop()
    GPIO.cleanup()
