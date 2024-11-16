from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

factory = PiGPIOFactory()
servo = Servo(17, pin_factory=factory)

servo.min()  # Move to minimum position
sleep(1)
servo.max()  # Move to maximum position
sleep(1)
servo.mid()  # Move to middle position
