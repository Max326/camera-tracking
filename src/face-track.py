import cv2
import serial
from time import sleep

ser = serial.Serial('/dev/ttyUSB0', baudrate=115200, timeout=1)

# Funkcja do wysyłania kąta przez UART
def sendAngle(angle):
    if 30 <= angle <= 90:
        msg = f"{angle}\n"
        ser.write(msg.encode())
        ser.flush()
        print(f"sent: {msg.strip()}")
    else:
        print("error: angle out of range")

# Ścieżka do klasyfikatora Haar cascade
haar_cascade_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)

# Inicjalizacja kąta serwa
current_angle = 50  # Środek zakresu (100-140)
sendAngle(current_angle)

step_size = 1  # Wielkość kroku przy regulacji kąta
threshold = 10  # Histereza, aby unikać drobnych oscylacji

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_center_y = y + h // 2
        frame_height = frame.shape[0]
        center_threshold = frame_height // 2

        if face_center_y < center_threshold - threshold:
            current_angle = min(90, current_angle + step_size)  # Przesuń serwo w górę
        elif face_center_y > center_threshold + threshold:
            current_angle = max(30, current_angle - step_size)  # Przesuń serwo w dół

        sendAngle(current_angle)
        print(f"Adjusted angle: {current_angle}")

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
