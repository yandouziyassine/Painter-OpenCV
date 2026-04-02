# Painter-OpenCV

A real-time, gesture-controlled drawing application built with **OpenCV**, **MediaPipe**, and **FastAPI**. This project allows users to draw on a virtual canvas using hand gestures captured via a webcam, with a web-based interface for monitoring and control.

## 🚀 Features

* **Real-time Hand Tracking:** Uses MediaPipe for high-fidelity finger landmark detection.
* **Gesture Recognition:** * **Index Finger up:** Draw on the canvas.
    * **Two fingers up / Palm:** Erase mode.
* **Virtual Toolbar:** Change colors or clear the canvas by "touching" virtual buttons in the video feed.
* **Web Dashboard:** * Live MJPEG video stream.
    * Real-time status monitoring (FPS, current color, mode).
    * Remote hardware control (Enable/Disable camera).
    * Clean server shutdown button.
* **Responsive UI:** Documentation and video feed integrated into a single, modern dark-mode interface.

## 🏗️ Tech Stack

* **Backend:** Python 3.x, FastAPI, Uvicorn.
* **Computer Vision:** OpenCV, MediaPipe.
* **Frontend:** HTML5, CSS3 (Flexbox), Vanilla JavaScript.
* **Multithreading:** Dedicated thread for the CV pipeline to ensure smooth web performance.

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yandouziyassine/Painter-OpenCV.git](https://github.com/yandouziyassine/Painter-OpenCV.git)
   cd Painter-OpenCV
2.
   python -m venv .venv
source .venv/bin/scripts/activate  # On Windows: .venv\Scripts\activate

3.
   pip install fastapi uvicorn opencv-python mediapipe numpy

4.
python Server.py
