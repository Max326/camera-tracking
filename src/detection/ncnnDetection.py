from ultralytics import YOLO
import cv2
import time

# Załaduj model, podając ścieżkę do FOLDERU NCNN
ncnn_model = YOLO('runs/detect/train4/weights/best_ncnn_model')

# Użyj kamery lub pliku wideo
video_path = 'data/videos/VID_20250813_122520.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_start_time = time.time()

    # Inferencja - reszta działa tak samo!
    results = ncnn_model(frame)

    annotated_frame = results[0].plot()

    frame_end_time = time.time()
    frame_time = frame_end_time - frame_start_time
    fps = 1 / frame_time if frame_time > 0 else 0
    print(f"FPS: {fps:.2f}")

    cv2.imshow("NCNN Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
total_time = end_time - start_time
overall_fps = frame_count / total_time if total_time > 0 else 0
print(f"Overall FPS: {overall_fps:.2f}")

cap.release()
cv2.destroyAllWindows()
