import cv2
import os

# Input and output directories
video_dir = "data/videos"
output_dir = "data/photos"

os.makedirs(output_dir, exist_ok=True)

# Extract frames from each video
for video_file in os.listdir(video_dir):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Save every nth frame (e.g., every 10th frame)
            if frame_count % 10 == 0:
                frame_name = f"{os.path.splitext(video_file)[0]}_frame{frame_count}.jpg"
                cv2.imwrite(os.path.join(output_dir, frame_name), frame)

            frame_count += 1

        cap.release()

print("Frame extraction complete!")