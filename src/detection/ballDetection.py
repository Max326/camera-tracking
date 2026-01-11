from ultralytics import YOLO
import cv2
import time

# Load the trained YOLOv8 model
# model = YOLO('runs/detect/train4/weights/best.pt')
model = YOLO('yolo11n.pt')

# Path to the video file
video_path = 'data/videos/VID_20250813_122520.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Start timing for this frame
    frame_start_time = time.time()

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # End timing for this frame
    frame_end_time = time.time()
    frame_time = frame_end_time - frame_start_time

    # Calculate FPS for this frame
    fps = 1 / frame_time if frame_time > 0 else 0
    print(f"FPS: {fps:.2f}")

    # Display the frame
    cv2.imshow('YOLOv8 Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Increment frame count
    frame_count += 1

# Calculate overall FPS
end_time = time.time()
total_time = end_time - start_time
overall_fps = frame_count / total_time if total_time > 0 else 0
print(f"Overall FPS: {overall_fps:.2f}")

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()