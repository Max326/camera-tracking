from ultralytics import YOLO

# Load the trained model
# model = YOLO('runs/detect/train4/weights/best.pt')
model = YOLO('yolo11n.pt')

# Export to ONNX format
model.export(format='ncnn')

# Eksportuj do TFLite z kwantyzacją INT8
# Potrzebujesz małego zestawu danych kalibracyjnych (np. 100 zdjęć z Twojego datasetu)
# model.export(format='tflite', data='data/dataset/data.yaml', int8=True)
