People Counting System (YOLO + OpenCV + Tkinter)
📌 Description

This project is a desktop application that uses YOLOv11 for object detection and OpenCV for tracking people in videos or images.
The main goal is to count the number of people passing through a user-defined region (ROI) such as an entrance or an area.

✨ Features

- Detect and track people (person class).
- Draw a polygon region (ROI) directly on the video.
- Count:
  Current number of people inside the region.
  Total number of people who entered the region.
- Can choose many model (yolo11n, yolo11s, yolo11m).
- Adjustable detection confidence threshold.
- Save output video with bounding boxes and counts.
- Save cropped images of detected people.

Simple GUI built with Tkinter.

🛠️ Technologies
- PythonOpenCV
- Ultralytics YOLO
- cvzone
- Tkinter
▶️ How to Run

Install dependencies:
- Use pip install opencv-python numpy ultralytics cvzone to install all requirement.
- Place the YOLO model file (yolo11n.pt, yolo11s.pt, or yolo11m.pt) in the project folder.
- Run the program: python main.py

🎮 Usage
-Choose Video → select a video file and draw ROI (polygon) by clicking on the video.
-Choose Image → run detection on a single image.
-Controls: Space → Pause/Resume video.
           r → Reset ROI.
           q → Quit program.

📂 Output
- Annotated video with bounding boxes and counts (if saving option chosen).
- Cropped person images saved in output_images/<video_name>/image/.
