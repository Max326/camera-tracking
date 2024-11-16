from gpiozero import AngularServo
from time import sleep

servo = AngularServo(18, min_pulse_width=0.0005, max_pulse_width=0.0025)

while True:
    servo.angle = -45
    print("moved to -45")
    sleep(1)
    servo.angle = -40
    print("moved to -40")
    sleep(1)
    servo.angle = -35
    print("moved to -35")
    sleep(1)
    servo.angle = -30
    print("moved to -30")
    sleep(1)
    servo.angle = -25
    print("moved to -25")
    sleep(1)
    servo.angle = -30
    print("moved to -30")
    sleep(1)
    servo.angle = -35
    print("moved to -35")
    sleep(1)
    servo.angle = -40
    print("moved to -40")
    sleep(1)
