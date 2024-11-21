from gpiozero import AngularServo
from time import sleep

servo = AngularServo(18, min_pulse_width=0.0005, max_pulse_width=0.0023) # works well

# servo = AngularServo(18)
# servo = AngularServo(18, min_pulse_width=0.0005, max_pulse_width=0.0025)

servoValue = 0

servo.angle = servoValue
print("moved to 0")
sleep(0.5)

angleChange = 20

while True:
    servoValue += angleChange
    servo.angle = servoValue

    print("moved to ", servoValue)
    sleep(0.5)

    servoValue -= angleChange
    servo.angle = servoValue

    print("moved to ", servoValue)
    sleep(0.5)
