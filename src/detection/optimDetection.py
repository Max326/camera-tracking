import onnxruntime as ort
import cv2
import numpy as np
import time

# Load the ONNX model
onnx_model_path = 'runs/detect/train4/weights/best.onnx'
session = ort.InferenceSession(onnx_model_path)

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

# Define a function to decode YOLOv8 outputs
def decode_outputs(outputs, frame_shape, conf_threshold=0.5):
    boxes, scores, class_ids = [], [], []
    for output in outputs[0]:  # Iterate over predictions
        conf = output[4]  # Confidence score
        if conf >= conf_threshold:
            x_center, y_center, width, height = output[:4]
            x1 = int((x_center - width / 2) * frame_shape[1])
            y1 = int((y_center - height / 2) * frame_shape[0])
            x2 = int((x_center + width / 2) * frame_shape[1])
            y2 = int((y_center + height / 2) * frame_shape[0])
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            class_ids.append(int(output[5]))  # Class ID
    return boxes, scores, class_ids

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for ONNX model
    input_frame = cv2.resize(frame, (640, 640))  # Resize to model input size
    input_frame = input_frame.transpose(2, 0, 1)  # HWC to CHW
    input_frame = input_frame[np.newaxis, :, :, :].astype(np.float32) / 255.0  # Normalize

    # Run inference
    outputs = session.run(None, {session.get_inputs()[0].name: input_frame})

    # Decode outputs
    boxes, scores, class_ids = decode_outputs(outputs, frame.shape)

    # Draw bounding boxes on the frame
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {class_id}: {score:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv8 Detection', frame)

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