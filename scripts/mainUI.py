import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import threading
from pathlib import Path

# Define paths
root_path = Path(__file__).resolve().parent.parent
model_name_color = 'best_color.pt'
model_name_gray = 'best_gray.pt'
model_path_color = root_path / 'model' / model_name_color
model_path_gray = root_path / 'model' / model_name_gray

# Function to process video
def process_video(input_path, output_path):
    try:
        # Load models
        model_color = YOLO(str(model_path_color))
        model_gray = YOLO(str(model_path_gray))

        # Open video
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        def is_grayscale(frame, threshold=15, color_percentage=5):
            b, g, r = cv2.split(frame)
            diff_bg = cv2.absdiff(b, g)
            diff_br = cv2.absdiff(b, r)
            diff_gr = cv2.absdiff(g, r)
            non_gray_pixels = (diff_bg > threshold) | (diff_br > threshold) | (diff_gr > threshold)
            non_gray_ratio = (non_gray_pixels.sum() / frame.size) * 100
            return non_gray_ratio < color_percentage

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Decide which model to use
            if is_grayscale(frame, threshold=15, color_percentage=5):
                results = model_gray.predict(frame, conf=0.05)
            else:
                results = model_color.predict(frame, conf=0.25)

            # Annotate frame
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

            # Display the frame in the UI (resized to fit canvas)
            display_frame(annotated_frame)

            # Update progress
            progress_label.config(text=f"Processing frame {i+1}/{frame_count}...")
            root.update_idletasks()

        cap.release()
        out.release()
        progress_label.config(text="Processing complete!")
        messagebox.showinfo("Success", f"Processed video saved to {output_path}!")

    except Exception as e:
        progress_label.config(text="Error occurred!")
        messagebox.showerror("Error", str(e))

# Function to display a frame in the Tkinter canvas
def display_frame(frame):
    # Resize the frame to fit the canvas dimensions
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    frame = cv2.resize(frame, (canvas_width, canvas_height))

    # Convert the frame (BGR to RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)

    # Display on the canvas
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    canvas.imgtk = imgtk  # Store a reference to avoid garbage collection

# Function to start processing in a new thread
def start_processing():
    input_path = input_path_var.get()
    output_path = output_path_var.get()

    if not input_path or not output_path:
        messagebox.showerror("Error", "Please select input and output paths.")
        return

    # Run the video processing in a separate thread
    threading.Thread(target=process_video, args=(input_path, output_path)).start()

# Function to select input file
def select_input_file():
    initial_dir = root_path / 'data' / 'videos'
    file_path = filedialog.askopenfilename(
        title="Select Input Video",
        initialdir=initial_dir,
        filetypes=(("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*")),
    )
    input_path_var.set(file_path)

    # Automatically set the output path based on the selected input video
    if file_path:
        video_name = Path(file_path).stem
        output_path = root_path / 'outputs' / f"OUT_{video_name}.mp4"
        output_path_var.set(output_path)

# Function to select output file
def select_output_file():
    initial_dir = root_path / 'outputs'
    file_path = filedialog.asksaveasfilename(
        title="Select Output Video",
        initialdir=initial_dir,
        defaultextension=".mp4",
        filetypes=(("MP4 Files", "*.mp4"), ("All Files", "*.*")),
    )
    output_path_var.set(file_path)

# Create the UI
root = tk.Tk()
root.title("Erkin Semiz - Forest Fire Detection")

# Make the window responsive
root.columnconfigure(1, weight=1)
root.rowconfigure(4, weight=1)

# Variables to store input and output paths
input_path_var = tk.StringVar()
output_path_var = tk.StringVar()

# Input video selection
tk.Label(root, text="Input Video:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
tk.Entry(root, textvariable=input_path_var, width=50).grid(row=0, column=1, padx=10, pady=10, sticky="ew")
tk.Button(root, text="Browse", command=select_input_file).grid(row=0, column=2, padx=10, pady=10)

# Output video selection
tk.Label(root, text="Output Video:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
tk.Entry(root, textvariable=output_path_var, width=50).grid(row=1, column=1, padx=10, pady=10, sticky="ew")
tk.Button(root, text="Browse", command=select_output_file).grid(row=1, column=2, padx=10, pady=10)

# Start processing button
tk.Button(root, text="Start Processing", command=start_processing, bg="green", fg="white").grid(
    row=2, column=1, padx=10, pady=20, sticky="ew"
)

# Progress label
progress_label = tk.Label(root, text="", fg="blue")
progress_label.grid(row=3, column=0, columnspan=3, pady=10, sticky="ew")

# Canvas to display frames (set a fixed size for the display)
canvas = tk.Canvas(root, width=800, height=600, bg="black")
canvas.grid(row=4, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

# Run the application
root.mainloop()
