from gpiozero import AngularServo
from time import sleep

servo = AngularServo(18, min_pulse_width=0.0005, max_pulse_width=0.0023)

while (True):
    servo.angle = 0
    print("moved to 0")
    sleep(1)
