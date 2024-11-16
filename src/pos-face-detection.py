import cv2
import os

# Specify the full path to the Haar cascade file
haar_cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

# Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Open the video stream
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (Haar cascade works on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image (scaleFactor and minNeighbors control detection quality)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Print or return the positions of the faces
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            print(f"Face detected at position: x={x}, y={y}, w={w}, h={h}")

            # Draw rectangles around the detected faces
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
