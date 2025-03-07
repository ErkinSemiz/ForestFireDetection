import cv2
from pathlib import Path
import os

# Define paths
root_path = Path(__file__).resolve().parent.parent
video_ext = '.mp4'
video_name = 'example_gray'
video_relative_path = Path('data/videos') / f"{video_name}{video_ext}"
video_path = root_path / video_relative_path

save_relative_path = Path('data/frames') / video_name
save_dir = root_path / save_relative_path

# Print paths for debugging
print(f"Video path: {video_path}")
print(f"Save path: {save_dir}")

# Create directories if they do not exist
os.makedirs(save_dir, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    print("Video opened successfully.")

count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    cv2.imwrite(str(save_dir / f"frame_{count:04d}.jpg"), frame)
    count += 1

cap.release()
print(f"Extracted {count} frames.")

