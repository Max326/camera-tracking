import libgpiod
from time import sleep

SERVO_PIN = 17  # GPIO pin connected to the servo
pi = libgpiod.pi()

# SG90 pulse width range: 500-2500 microseconds
MIN_PULSE = 500  # Corresponds to -90 degrees
MAX_PULSE = 2500  # Corresponds to +90 degrees

def angle_to_pulse(angle):
    """Convert angle (-90 to 90) to pulse width (500 to 2500)."""
    return int(MIN_PULSE + (MAX_PULSE - MIN_PULSE) * (angle + 90) / 180)

try:
    while True:
        for angle in [90, 85, 80, 75, 70, 75, 80, 85, 90]:
            pulse_width = angle_to_pulse(angle)
            pi.set_servo_pulsewidth(SERVO_PIN, pulse_width)
            sleep(0.5)
except KeyboardInterrupt:
    print("Program stopped by user")
finally:
    pi.set_servo_pulsewidth(SERVO_PIN, 0)  # Turn off the servo
    pi.stop()
