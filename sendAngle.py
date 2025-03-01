import serial
from time import sleep

ser = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=1)

def sendAngle(angle):
    if -90 <= angle <= 90:
        msg = f"{angle}\n"
        ser.write(msg.encode())
        ser.flush()
        print(f"sent: {msg.strip()}")
    else:
        print("error: angle out of range")

i = -90

while True:
    sendAngle(i)  # Wysyłaj stały kąt 45 stopni
    i += 1
    if (i == 90):
        i = 0
    sleep(0.1)