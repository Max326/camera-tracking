from ultralytics import YOLO

# Create a YOLOv8 model
model = YOLO('yolov11n.pt')  # Use a pre-trained YOLOv8 model

# Train the model
model.train(data='data/dataset/data.yaml', epochs=50, imgsz=640)