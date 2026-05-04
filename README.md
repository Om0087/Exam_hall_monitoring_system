# 🎓 Exam Hall Monitoring System

An advanced, real-time Computer Vision based application designed to monitor exam halls and detect potential cheating behaviors. Built with Python, Flask, WebSockets, and state-of-the-art machine learning models including YOLOv8, dlib, and Mediapipe.

## ✨ Key Features

The system actively monitors live camera feeds or uploaded video files to detect:

1. **📱 Prohibited Objects Detection:** Uses YOLOv8 to detect mobile phones in the camera frame.
2. **🗣️ Talking & Lip Movement:** Analyzes the Mouth Aspect Ratio (MAR) using dlib facial landmarks to detect if a student is whispering or talking.
3. **👀 Eye Gaze Tracking:** Uses Mediapipe Face Mesh to precisely track iris movements and detect abnormal left/right gazing.
4. **🔄 Head Turning:** Uses 3D head pose estimation (SolvePnP) to track excessive head turning.
5. **👥 Impersonation & Multiple Faces:** Flags the frame if the number of detected faces exceeds the allowed limit.
6. **📏 Proximity Detection:** Measures the distance between students and flags if they get too close to one another.
7. **🤝 Item Passing Detection:** Tracks small objects (like phones, books) and hands to identify suspicious item passing between multiple students.

## 🏗️ Architecture

- **Backend:** Flask, Flask-SocketIO (for real-time streaming)
- **Asynchronous Processing:** Eventlet background threads to prevent UI blocking
- **Computer Vision:** OpenCV
- **Face & Landmark Detection:** dlib (`shape_predictor_68_face_landmarks.dat`)
- **Object Detection:** YOLOv8 Nano (`yolov8n.pt`)
- **Precise Eye Tracking:** Mediapipe Face Mesh

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Om0087/Exam_hall_monitoring_system.git
cd Exam_hall_monitoring_system
```

### 2. Install Dependencies
Make sure you have Python installed. It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Download Model Weights
Due to GitHub size limitations, the machine learning models are not included in the repository. You must download them and place them in the root directory:
- **dlib 68-Point Face Landmark Model:** Download `shape_predictor_68_face_landmarks.dat`
- **YOLOv8 Nano:** Download `yolov8n.pt`

### 4. Run the Application
```bash
python app.py
```
Once the server starts, open your browser and navigate to `http://localhost:5000` (or the port specified in the terminal).

## 📄 Logging & Reports
The system automatically records all suspicious events (including timestamps, student IDs, and the specific behavior detected) into a thread-safe `detection_logs.csv` file. You can download these logs directly from the web interface at the end of the session.

## 🤝 Contributing
Feel free to fork this project, submit issues, and create pull requests to help improve the detection logic or add new features!
