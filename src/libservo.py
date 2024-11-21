import gpiod
import time

# Select GPIO chip and line
chip = gpiod.Chip('gpiochip0')
line = chip.get_line(18)

# Request the GPIO line for output
config = gpiod.LineRequest()
config.consumer = "Servo Control"
config.request_type = gpiod.LineRequest.DIRECTION_OUTPUT
line.request(config)

def set_servo_angle(line, angle):
    # Convert angle to pulse width (in seconds)
    pulse_width = 0.001 + (angle / 180.0) * 0.001  # 1 ms to 2 ms
    period = 0.02  # 20 ms (50 Hz)
    for _ in range(50):  # Run for 1 second
        line.set_value(1)
        time.sleep(pulse_width)
        line.set_value(0)
        time.sleep(period - pulse_width)

# Test the servo
set_servo_angle(line, 0)    # Move to 0 degrees
time.sleep(1)
set_servo_angle(line, 90)   # Move to 90 degrees
time.sleep(1)
set_servo_angle(line, 180)  # Move to 180 degrees
