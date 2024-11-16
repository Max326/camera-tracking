import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)

# Test the GPIO pin by turning it on and off
GPIO.output(18, GPIO.HIGH)  # Set GPIO 18 high (on)
sleep(1)
GPIO.output(18, GPIO.LOW)   # Set GPIO 18 low (off)
sleep(1)

GPIO.cleanup()
