import I2C_LCD_driver
from time import *

mylcd = I2C_LCD_driver.lcd()

def displayText(text):
    mylcd.lcd_display_string(text, 1)

displayText("open")

