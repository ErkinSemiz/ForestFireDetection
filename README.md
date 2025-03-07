# Forest Fire Detection from Drone Recorded Grayscale and Coloured Video Tapes

Forest Fire Detection with model training, frame extracting, frame labeling and Bounding Box labeled video processing with monochrome and stereo cam footages.

![image](https://github.com/user-attachments/assets/7261e09d-6e5c-4475-94ad-97ec40599efd)


**Creator:** Erkin Semiz  
**Python Version:** Python 3.10 (Anaconda Environment)  
**Environment File:** `python310MLfinal.yaml`  
**Requirements File:** `requirements.txt`  

---

## **Project Overview**
This project implements a comprehensive pipeline for **forest fire detection** from drone-recorded videos in both grayscale and color formats. The solution includes three main phases: **frame generation**, **model training**, and **fire detection**. It is designed to efficiently process large datasets and detect fire and smoke from aerial footage.

---

## **Project Folder Structure**
PROJECT/ 
├── data/ 
    ├── frames/ # Extracted frames from videos │ 
    ├── labels/ # Labels for training │ 
    ├── videos/ # Input video files (grayscale and color) │ 
        ├── deneme_gray.mp4 │ 
        ├── deneme.mp4 
├── model/ 
    ├── best_color.pt # YOLOv8 model for color video detection │ 
    ├── best_gray.pt # YOLOv8 model for grayscale video detection │ 
    ├── ModelTraining.ipynb # Model training notebook 
├── outputs/ 
    ├── OUT.mp4 # Processed video with annotations 
├── scripts/  
    ├── FrameExtractor.py # Script to extract frames from videos │ 
    ├── FrameExtractorUI.py # UI for frame extraction │ 
    ├── main.py # Main detection script │ 
├── mainUI.py # UI for video processing
├── python310MLfinal.yaml # Anaconda environment file 
├── requirements.txt # Python dependencies 
└── README.md # Project documentation

---

## **Project Workflow**
The project is divided into three major phases:

### **1. Frame Generation**
The first phase involves generating frames from input videos. You can extract frames using the script `FrameExtractor.py` or its corresponding UI version, `FrameExtractorUI.py`.

**Steps**:
1. Place input videos in the `data/videos/` directory.
2. Specify the input video path and save directory in the script or use the UI to select them interactively.
3. Run the script to extract frames into `data/frames/`.

**Example:**
python scripts/FrameExtractor.py

Output:
Frames are saved as frame_0000.jpg, frame_0001.jpg, etc., in the specified directory.

### **2. Model Training
The second phase is training the YOLOv8 models on the annotated datasets for fire and smoke detection.

Datasets:

Grayscale Dataset:
Path: /content/drive/MyDrive/FinalProject/firedetectormono
Dataset Link
Color Dataset:
Path: /content/drive/MyDrive/FinalProject/FireDetectionDataset/
Dataset Link
Steps:

Open ModelTraining.ipynb from the model folder.
Use the training data (70% train, 20% valid, 10% test) to train two models:
Grayscale model (best_gray.pt)
Color model (best_color.pt)
Save the trained weights to the model directory.
Example: Run the notebook in Jupyter or Google Colab to train the models.

Outputs:

best_gray.pt
best_color.pt
### **3. Fire Detection
The final phase is the detection of forest fires and smoke from input videos using the trained models.

Script: main.py

How it works:
    Loads the trained YOLOv8 models (best_gray.pt and best_color.pt).
    Processes the input video from the data/videos/ directory.
    Decides which model to use for each frame (grayscale or color) based on the frame content.
    Annotates detected fire/smoke regions with bounding boxes.
    Saves the processed video to the outputs/ directory.

python scripts/main.py
Example Workflow:

Place the input video (e.g., deneme.mp4) in data/videos/.
Run the script or use mainUI.py to process the video interactively.
The annotated video will be saved as OUT.mp4 in the outputs/ directory.

### **Setup Instructions
1. Prerequisites
    Python 3.10
    Anaconda (recommended)
2. Setting Up the Environment
Using Anaconda:

    conda env create -f python310MLfinal.yaml
    conda activate python310MLfinal

Using requirements.txt:

    pip install -r requirements.txt

### **Scripts Overview
FrameExtractor.py
    Extracts frames from input videos and saves them as image files.
    Hardcoded paths for video and save directory can be replaced with dynamic user input through FrameExtractorUI.py.
ModelTraining.ipynb
    Trains YOLOv8 models on grayscale and color datasets.
    Produces best_gray.pt and best_color.pt.
main.py
    Detects forest fires and smoke in input videos.
    Uses separate models for grayscale and color frames.
mainUI.py
    Interactive UI for running video processing with user-selected input and output paths.

### **Key Features
    Automated Frame Extraction: Extract frames from videos for dataset creation.
    Semi-Automatic Labeling: Use tools like Roboflow to label large datasets efficiently.
    Dual Model Detection: Separate YOLOv8 models for grayscale and color video processing.
    Interactive UIs: Easy-to-use interfaces for frame extraction and fire detection.

### **Acknowledgements
    Ultralytics YOLOv8
    Roboflow for dataset preparation and annotation tools.
