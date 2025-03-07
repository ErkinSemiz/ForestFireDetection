import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Define paths
root_path = Path(__file__).resolve().parent.parent

# Video and model names
video_name = 'example_color'
video_ext = '.mp4'
model_name_color = 'best_color'
model_name_gray = 'best_gray'
model_ext = '.pt'

# Relative paths
video_relative_path = Path('data/videos') / f"{video_name}{video_ext}"
model_relative_path_color = Path('model') / f"{model_name_color}{model_ext}"
model_relative_path_gray = Path('model') / f"{model_name_gray}{model_ext}"
output_relative_path = Path('outputs') / f"OUT_{video_name}{video_ext}"

# Full paths
video_path = root_path / video_relative_path
model_path_color = root_path / model_relative_path_color
model_path_gray = root_path / model_relative_path_gray
output_path = root_path / output_relative_path

# 1. Load your two custom-trained models
model_color = YOLO(str(model_path_color))
model_gray = YOLO(str(model_path_gray))

# 2. Open the video file
cap = cv2.VideoCapture(str(video_path))

# 3. Prepare VideoWriter for output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

def is_grayscale(frame, threshold=15, color_percentage=5):
    """
    Determine if a frame is grayscale, allowing for small regions of color.
    Args:
        frame: Input frame (BGR image).
        threshold: Max difference between channels to consider as grayscale.
        color_percentage: Max percentage of the image that can be 'colorful'
                          for it to still be classified as grayscale.
    Returns:
        True if the frame is predominantly grayscale, False otherwise.
    """
    # Split the frame into its B, G, and R channels
    b, g, r = cv2.split(frame)

    # Compute absolute differences between channels
    diff_bg = cv2.absdiff(b, g)
    diff_br = cv2.absdiff(b, r)
    diff_gr = cv2.absdiff(g, r)

    # Threshold for "non-grayscale" pixels (pixels with significant color differences)
    non_gray_pixels = np.where((diff_bg > threshold) | (diff_br > threshold) | (diff_gr > threshold), 1, 0)

    # Calculate the percentage of non-grayscale pixels in the frame
    non_gray_ratio = (np.sum(non_gray_pixels) / frame.size) * 100

    # If the percentage of colorful regions is below the threshold, consider it grayscale
    return non_gray_ratio < color_percentage

# 4. Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Decide which model to use
    if is_grayscale(frame, threshold=15, color_percentage=5):
        # Use the grayscale model
        results = model_gray.predict(frame, conf=0.25)
        print("Grayscale")
    else:
        # Use the color model
        results = model_color.predict(frame, conf=0.25)
        print("Color")
    # Get the annotated frame
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

# 5. Release resources
cap.release()
out.release()
print(f"Processed video saved to {output_path}.")
