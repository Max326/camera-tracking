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

# Test function to energize coils one by one
def simple_stepper_test():
    print("Energizing coil 1...")
    coil1.on()
    time.sleep(1)
    coil1.off()

    print("Energizing coil 2...")
    coil2.on()
    time.sleep(1)
    coil2.off()

    print("Energizing coil 3...")
    coil3.on()
    time.sleep(1)
    coil3.off()

    print("Energizing coil 4...")
    coil4.on()
    time.sleep(1)
    coil4.off()

# Run the simple test
simple_stepper_test()
