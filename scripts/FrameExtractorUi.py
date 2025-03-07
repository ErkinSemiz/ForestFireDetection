import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import threading
from pathlib import Path

# Function to extract frames from video
def extract_frames(video_path, save_dir):
    try:
        cap = cv2.VideoCapture(video_path)
        count = 0

        # Create the save directory if it doesn't exist
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Save each frame as a JPG file
            frame_path = save_dir_path / f"frame_{count:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            count += 1
            # Update progress
            progress_label.config(text=f"Extracting frame {count}...")
            root.update_idletasks()

        cap.release()
        progress_label.config(text=f"Extraction complete! {count} frames saved.")
        messagebox.showinfo("Success", f"Extracted {count} frames to {save_dir}")

    except Exception as e:
        progress_label.config(text="Error occurred!")
        messagebox.showerror("Error", str(e))


# Function to start extraction in a new thread
def start_extraction():
    video_path = video_path_var.get()
    save_dir = save_dir_var.get()

    if not video_path or not save_dir:
        messagebox.showerror("Error", "Please select an input video and a save directory.")
        return

    # Run extraction in a separate thread to keep UI responsive
    threading.Thread(target=extract_frames, args=(video_path, save_dir)).start()


# Function to select input video
def select_input_video():
    initial_dir = Path(__file__).resolve().parent.parent / 'data/videos'
    file_path = filedialog.askopenfilename(
        title="Select Input Video",
        initialdir=initial_dir,
        filetypes=(("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*")),
    )
    video_path_var.set(file_path)

    # Automatically set the save directory based on the selected video
    if file_path:
        video_name = Path(file_path).stem
        save_dir = Path(__file__).resolve().parent.parent / 'data/frames' / video_name
        save_dir_var.set(save_dir)


# Function to select save directory
def select_save_directory():
    initial_dir = Path(__file__).resolve().parent.parent / 'data/frames'
    dir_path = filedialog.askdirectory(title="Select Save Directory", initialdir=initial_dir)
    save_dir_var.set(dir_path)


# Create the UI
root = tk.Tk()
root.title("Erkin Semiz - Frame Extractor")

# Make the window responsive
root.columnconfigure(1, weight=1)
root.rowconfigure(3, weight=1)

# Variables to store paths
video_path_var = tk.StringVar()
save_dir_var = tk.StringVar()

# Input video selection
tk.Label(root, text="Input Video:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
tk.Entry(root, textvariable=video_path_var, width=50).grid(row=0, column=1, padx=10, pady=10, sticky="ew")
tk.Button(root, text="Browse", command=select_input_video).grid(row=0, column=2, padx=10, pady=10)

# Save directory selection
tk.Label(root, text="Save Directory:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
tk.Entry(root, textvariable=save_dir_var, width=50).grid(row=1, column=1, padx=10, pady=10, sticky="ew")
tk.Button(root, text="Browse", command=select_save_directory).grid(row=1, column=2, padx=10, pady=10)

# Start extraction button
tk.Button(root, text="Start Extraction", command=start_extraction, bg="green", fg="white").grid(
    row=2, column=1, padx=10, pady=20, sticky="ew"
)

# Progress label
progress_label = tk.Label(root, text="", fg="blue")
progress_label.grid(row=3, column=0, columnspan=3, pady=10, sticky="ew")

# Run the application
root.mainloop()
