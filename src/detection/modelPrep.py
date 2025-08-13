from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/train4/weights/best.pt')

# Export to ONNX format
model.export(format='onnx')