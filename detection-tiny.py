import cv2
import numpy as np

# Load the YOLOv4-Tiny model
yolo_net = cv2.dnn.readNet("yolo/yolov4-tiny.weights", "yolo/yolov4-tiny.cfg")

# Define the class labels for YOLO
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Open the video stream (use your camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match the input size for YOLOv4-Tiny
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (64, 64), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)

    # Get the output layer indices
    layer_names = yolo_net.getLayerNames()
    output_layers_indices = yolo_net.getUnconnectedOutLayers()

    # Get the output layer names
    output_layers = [layer_names[i - 1] for i in output_layers_indices]

    # Forward pass through YOLO
    outs = yolo_net.forward(output_layers)

    # Process the outputs (bounding boxes, confidences, and class ids)
    class_ids = []
    confidences = []
    boxes = []
    height, width, channels = frame.shape

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression to eliminate redundant boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the detected boxes and labels
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with the detections
    cv2.imshow("Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
