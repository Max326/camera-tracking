import cv2
import numpy as np
import time

# Użyj 'tflite_runtime.interpreter' zamiast 'tensorflow.lite.Interpreter' na RPi
# Spróbuje zaimportować pełny TF, a jeśli się nie uda, użyje lekkiej wersji
try:
    import tensorflow.lite as tflite
except ImportError:
    import tflite_runtime.interpreter as tflite

# --- Konfiguracja ---
MODEL_PATH = "runs/detect/train4/weights/best_saved_model/best_int8.tflite" # Ścieżka do Twojego modelu .tflite
VIDEO_PATH = "data/videos/VID_20250813_122520.mp4"
CONF_THRESHOLD = 0.5  # Próg pewności detekcji (dostosuj w razie potrzeby)
NMS_THRESHOLD = 0.4   # Próg dla Non-Maximum Suppression

# --- Ładowanie modelu TFLite ---
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height, input_width = input_details[0]['shape'][1], input_details[0]['shape'][2]

print(f"Model oczekuje wejścia o rozmiarze: {input_width}x{input_height}")

# --- Przetwarzanie wideo ---
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
total_inference_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- Pre-processing obrazu ---
    # Zapamiętaj oryginalne wymiary klatki
    original_height, original_width, _ = frame.shape
    
    # Zmień rozmiar klatki do rozmiaru wejściowego modelu
    img_resized = cv2.resize(frame, (input_width, input_height))
    
    # Dodaj wymiar 'batch' i przekonwertuj do odpowiedniego typu
    input_data = np.expand_dims(img_resized, axis=0)
    input_data = input_data.astype(input_details[0]['dtype'])
    
    # --- Inferencja ---
    inference_start = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    inference_end = time.time()
    
    frame_inference_time = inference_end - inference_start
    total_inference_time += frame_inference_time
    frame_count += 1
    
    # --- Post-processing wyników (kluczowy etap dla YOLO) ---
    # Transpozycja wyników, aby miały kształt (liczba_detekcji, 5) -> [x, y, w, h, conf]
    detections = np.squeeze(output_data).T
    
    boxes = []
    confidences = []

    for detection in detections:
        confidence = detection[4]
        if confidence >= CONF_THRESHOLD:
            # Skalowanie współrzędnych ramki do oryginalnego rozmiaru obrazu
            x, y, w, h = detection[0:4]
            x1 = int((x - w / 2) * original_width / input_width)
            y1 = int((y - h / 2) * original_height / input_height)
            x2 = int((x + w / 2) * original_width / input_width)
            y2 = int((y + h / 2) * original_height / input_height)
            boxes.append([x1, y1, x2-x1, y2-y1]) # cv2.dnn.NMSBoxes wymaga (x,y,w,h)
            confidences.append(float(confidence))

    # Zastosuj Non-Maximum Suppression, aby usunąć nachodzące na siebie ramki
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    
    # Narysuj ostateczne ramki na oryginalnej klatce
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"Pilka: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- Wyświetlanie FPS i obrazu ---
    avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
    cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("YOLOv8 TFLite Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Średnia wydajność (FPS) na podstawie czasu samej inferencji: {avg_fps:.2f}")

