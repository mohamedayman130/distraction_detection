
# 🚗 Real-Time Driver Distraction Detection System

This project is a **real-time distraction detection** web application built using:
- **FastAPI** for backend and live video streaming
- **OpenCV** for video processing
- **MediaPipe** for detecting face mesh and hands
- Deployed via **Uvicorn**

It detects:
- 👁️ Eye closure (drowsiness)
- 💤 Yawning
- ✋ Hand near face (potential phone use or distraction)

---

## 🔧 Features

- **Live webcam streaming** through browser
- Detects 3 types of distractions:
  - **Eye closure** using Eye Aspect Ratio (EAR)
  - **Yawning** via lip distance
  - **Hand near face** by measuring fingertip to nose distance
- Displays `FOCUSED` or `DISTRACTED` status in real time
- Annotated landmarks for both face and hands
- Fast and lightweight – runs in real-time on local machines

---

## 📁 Project Structure

```
.
├── main.py             # FastAPI app with all logic
├── index.html          # HTML frontend for viewing stream
├── README.md           # This documentation
```

---

## 💻 Requirements

- Python 3.8+
- pip

### 🔌 Install Dependencies

```bash
pip install fastapi uvicorn opencv-python mediapipe numpy
```

---

## 🚀 Running the App

### Step 1: Start the Server

```bash
python main.py
```

By default, the app will run at: `http://localhost:8000`

### Step 2: Open the Browser

Navigate to:

```
http://localhost:8000
```

You will see the live video feed with distraction status and FPS.

---

## 🧠 How It Works

### 1. **Eye Closure Detection**
- Calculates Eye Aspect Ratio (EAR)
- Triggers drowsiness alert if EAR remains low for a few frames

### 2. **Yawn Detection**
- Measures vertical lip distance
- Calculates a `yawn metric` based on mouth width
- Triggers yawn alert if value exceeds threshold

### 3. **Hand Near Face Detection**
- Detects hands using MediaPipe
- Measures distance from hand fingertips (index, middle) to nose
- If distance is low for a few frames → triggers hand-near-face warning

---

## 🧪 Configuration Parameters

| Parameter               | Description                              | Default |
|-------------------------|------------------------------------------|---------|
| `EAR_THRESH`            | Eye Aspect Ratio threshold               | 0.2     |
| `EAR_CONSEC_FRAMES`     | Frames below EAR to count as drowsy      | 5       |
| `YAWN_THRESH`           | Yawn metric threshold (%)                | 38      |
| `YAWN_CONSEC_FRAMES`    | Consecutive frames to trigger yawn       | 4       |
| `HAND_DIST_THRESH`      | Pixel distance hand to nose              | 120     |
| `HAND_CONSEC_FRAMES`    | Frames for hand-near-face                | 3       |
| `DISTRACTED_THRESHOLD`  | Counter to show "DISTRACTED"             | 15      |

---

## 🖼️ Screenshot

> Add a screenshot like below to show output

![screenshot](screenshot.png)

---

## 📌 Notes

- Make sure your webcam is accessible.
- Designed to work with `localhost` only — no external camera or IP streaming is enabled here.
- If using Android IP Webcam app, replace `cv2.VideoCapture(0)` with IP stream URL.

---

## 🛠️ Future Enhancements (Optional Ideas)

- Integrate **YOLOv8** to detect mobile phone usage visually
- Log distraction events into CSV or database
- Add sound alerts (beep or voice)
- Host as Docker container for cross-platform deployment

---

## 📃 License

MIT License – Free to use and modify
