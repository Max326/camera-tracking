import gpiod
import time

# Open the GPIO chip
chip = gpiod.Chip('gpiochip0')  # Use gpiochip0, or the appropriate chip on your system
line = chip.get_line(18)        # Set GPIO pin 18 for PWM output
line.request(consumer='servo_pwm', type=gpiod.LINE_REQ_DIR_OUT)

# PWM signal parameters
frequency = 50  # 50Hz PWM frequency (for standard servos like MG996R)
period = 1 / frequency  # PWM period in seconds

def generate_pwm(duty_cycle):
    """Generate PWM signal with the given duty cycle."""
    high_time = period * duty_cycle
    low_time = period - high_time

    line.set_value(1)  # High for 'high_time'
    time.sleep(high_time)
    line.set_value(0)  # Low for 'low_time'
    time.sleep(low_time)

try:
    while True:
        print("Generating PWM signal")
        
        # Servo at 0° (5% duty cycle)
        duty_cycle = 0.05
        generate_pwm(duty_cycle)

        time.sleep(2)  # Wait for 2 seconds
        
        # Servo at 90° (10% duty cycle)
        duty_cycle = 0.1
        generate_pwm(duty_cycle)

        time.sleep(2)  # Wait for 2 seconds

except KeyboardInterrupt:
    print("Program interrupted.")
finally:
    line.release()  # Release the GPIO line
    chip.close()  # Close the chip
