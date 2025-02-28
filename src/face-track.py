import cv2
from gpiozero import AngularServo
from time import sleep
import serial

ser = serial.Serial('/dev/serial0', baudrate=115200, timeout=1)

def sendAngle(angle):
    if 100 <= angle <= 140:
        msg = f"{angle}\n"
        ser.write(msg.encode())
        print(f"sent: {msg.strip()}")
    else:
        print("error: angle out of range")

# Initialize the servo
servo = AngularServo(18, min_pulse_width=0.0005, max_pulse_width=0.0023)

# Specify the full path to the Haar cascade file
haar_cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

# Load the Haar Cascade model
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Open the video stream
cap = cv2.VideoCapture(0)

# Initial servo angle
servo.angle = 0
current_angle = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Use the first detected face (largest one)
        (x, y, w, h) = faces[0]

        # Calculate the center of the face
        face_center_y = y + w // 2

        # Get the width of the frame
        frame_width = frame.shape[1]

        # Map the face position to servo range [-45, 45]
        normalized_position = (face_center_y - frame_width / 2) / (frame_width / 2)
        target_angle = -60 * normalized_position  # Map to [-45, 45]

        # Limit the angle within the servo range
        target_angle = max(-60, min(60, target_angle))

        sendAngle(target_angle)
        current_angle = target_angle

        # Move the servo if the angle difference is significant
        # if abs(target_angle - current_angle) > 1:  # Update only if change > 1 degree
        # servo.angle = target_angle
        # current_angle = target_angle
        print(f"Servo moved to: {current_angle:.2f} degrees")

    # Display the video feed
    cv2.imshow('Face Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
