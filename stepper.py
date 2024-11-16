import time
from gpiozero import OutputDevice

# Define the GPIO pins for the ULN2003 driver
IN1 = 17
IN2 = 18
IN3 = 27
IN4 = 22

# Create OutputDevice instances to control the stepper motor
coil1 = OutputDevice(IN1)
coil2 = OutputDevice(IN2)
coil3 = OutputDevice(IN3)
coil4 = OutputDevice(IN4)

# Define the half-step sequence for the 28BYJ-48 motor
step_sequence = [
    [coil1, None, None, None],    # Step 1
    [coil1, coil2, None, None],   # Step 2
    [None, coil2, None, None],    # Step 3
    [None, coil2, coil3, None],   # Step 4
    [None, None, coil3, None],    # Step 5
    [None, None, coil3, coil4],   # Step 6
    [None, None, None, coil4],    # Step 7
    [coil1, None, None, coil4],   # Step 8
]

# Function to rotate the motor
def rotate_motor(steps, delay=0.005):
    for _ in range(abs(steps)):
        for step in step_sequence:
            # Activate coils for this step
            if step[0]: step[0].on()  # Coil 1
            if step[1]: step[1].on()  # Coil 2
            if step[2]: step[2].on()  # Coil 3
            if step[3]: step[3].on()  # Coil 4

            # Deactivate coils after the step
            if step[0]: step[0].off()  # Coil 1
            if step[1]: step[1].off()  # Coil 2
            if step[2]: step[2].off()  # Coil 3
            if step[3]: step[3].off()  # Coil 4

            # Delay between steps
            time.sleep(delay)

# Rotate 360 degrees (2048 steps)
steps_per_revolution = 2048

# Rotate 360 degrees left (clockwise)
print("Rotating 360 degrees left...")
rotate_motor(steps_per_revolution)
time.sleep(1)

# Rotate 360 degrees right (counter-clockwise)
print("Rotating 360 degrees right...")
rotate_motor(-steps_per_revolution)
time.sleep(1)
