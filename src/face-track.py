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
current_angle = 50  # Początkowy kąt serwa (środek zakresu 30-90)
sendAngle(current_angle)

# Parametry regulatora P
Kp = 0.03  # Wzmocnienie regulatora P (można dostosować)
# setpoint = 240  # Pożądana pozycja twarzy (środek obrazu w osi Y)
dead_zone = 0  # Strefa martwa, aby uniknąć oscylacji

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_center_y = y + h // 2  # Środek twarzy w osi Y
        frame_height = frame.shape[0]

        # Obliczenie błędu (odchylenia od setpoint)
        setpoint = frame_height // 2
        error = setpoint - face_center_y

        # Jeśli błąd jest poza strefą martwą, zastosuj regulator P
        if abs(error) > dead_zone:
            # Regulator P: korekta = Kp * error
            correction = Kp * error

            # Aktualizacja kąta serwa
            current_angle += correction

            # Ograniczenie kąta do zakresu 30-90
            current_angle = max(30, min(90, current_angle))

            # Wyślij nowy kąt do serwa
            sendAngle(int(current_angle))
            print(f"Adjusted angle: {int(current_angle)}, Error: {error}")

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()