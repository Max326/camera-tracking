from gpiozero import PWMOutputDevice
from time import sleep

# Initialize the PWM output on GPIO 18
servo = PWMOutputDevice(18)

# Check if servo is initialized correctly
if servo.is_active:
    print("Servo is active.")
else:
    print("Servo is not active. Check wiring or power supply.")

# Move servo to 50% duty cycle (simulate 90° position)
servo.value = 0.5
print("Moving to 90° (50% duty cycle).")

# Wait for 5 seconds
sleep(5)

# Set servo to 0% duty cycle (simulating 0° position)
servo.value = 0
print("Moving to 0° (0% duty cycle).")

# Wait for 5 seconds
sleep(5)

# Set servo to 100% duty cycle (simulate 180° position)
servo.value = 1
print("Moving to 180° (100% duty cycle).")

# Clean up and close the servo
servo.close()
print("Servo movement complete.")
