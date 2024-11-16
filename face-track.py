import cv2
from gpiozero import AngularServo
from time import sleep

# Initialize the servo
servo = AngularServo(18, min_pulse_width=0.0005, max_pulse_width=0.0025)

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
        face_center_x = x + w // 2

        # Get the width of the frame
        frame_width = frame.shape[1]

        # Map the face position to servo range [-45, 45]
        normalized_position = (face_center_x - frame_width / 2) / (frame_width / 2)
        target_angle = -45 * normalized_position  # Map to [-45, 45]

        # Limit the angle within the servo range
        target_angle = max(-45, min(45, target_angle))

        # Move the servo if the angle difference is significant
        if abs(target_angle - current_angle) > 1:  # Update only if change > 1 degree
            servo.angle = target_angle
            current_angle = target_angle
            print(f"Servo moved to: {current_angle:.2f} degrees")

    # Display the video feed
    cv2.imshow('Face Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
