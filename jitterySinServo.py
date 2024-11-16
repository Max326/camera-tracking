from gpiozero import AngularServo
from time import sleep, time

# Initialize the servo with appropriate parameters
servo = AngularServo(17, min_angle=0, max_angle=180, min_pulse_width=0.0005, max_pulse_width=0.0025)

# Define the speed and the timing for the smooth transition
increment = 1  # Increment of angle change (1 degree)
delay_time = 0.2  # Delay in seconds between each angle update
start_time = time()

while True:
    for angle in range(0, 181, increment):  # Move from 0 to 180 degrees
        servo.angle = angle
        sleep(delay_time)  # Pause for a short time to give the servo time to move

    for angle in range(180, -1, -increment):  # Move back from 180 to 0 degrees
        servo.angle = angle
        sleep(delay_time)
