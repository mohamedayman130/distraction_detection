# 🚗 Real-Time Lane Detection with FastAPI

This project is a real-time lane detection system using OpenCV for image processing and FastAPI for serving an MJPEG video stream to a browser. It uses Hough Transform to detect lane lines and overlays direction guidance (e.g., "Turn Left", "Turn Right", "Straight") on a live camera feed.

## 📸 Demo

![Demo Screenshot](demo-screenshot.png) <!-- Optional: Replace with your own screenshot or remove -->

## ⚙️ Features

- Real-time lane detection with visual overlays
- Direction suggestion based on lane deviation
- FastAPI server with MJPEG streaming endpoint
- Configurable smoothing, Canny, and Hough parameters
- Supports Android IP Webcam or local webcam as video source

---

## 🧰 Requirements

- Python 3.7+
- OpenCV
- FastAPI
- Uvicorn
- NumPy

### Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

```
fastapi
uvicorn
opencv-python
numpy
```

---

## 🚀 How to Run

### 1. Setup your video source

You can use the IP Webcam Android app:

- Install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) on your Android phone.
- Connect your phone and PC to the same Wi-Fi.
- Open IP Webcam app → Start server → note the IP and port (e.g., `http://192.168.1.3:8080`)
- Update `VIDEO_SOURCE` in `main.py` with the video stream URL:
  ```python
  VIDEO_SOURCE = "http://192.168.1.3:8080/video"
  ```

Or use a USB webcam by setting:

```python
VIDEO_SOURCE = 0
```

### 2. Run the server

```bash
python main.py
```

### 3. Open your browser

Go to [http://localhost:8000](http://localhost:8000) to see the live lane detection stream.

---

## 🧠 How It Works

- Converts each video frame to grayscale
- Applies Gaussian blur and Canny edge detection
- Extracts the region of interest (ROI)
- Uses Hough Transform to detect lines
- Classifies and smooths left/right lane lines
- Draws lane overlays and direction guidance
- Streams processed frames via FastAPI MJPEG stream

---

## 📁 Project Structure

```
.
├── main.py              # Main application with FastAPI and OpenCV
├── index.html           # Frontend HTML for live stream display
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🛠 Configuration

You can tune detection accuracy via these parameters in `main.py`:

```python
CANNY_THRESHOLDS = (150, 200)           # Canny edge detector thresholds
GAUSSIAN_KERNEL = (5, 5)                # Blur kernel size
SMOOTHING_FRAMES = 10                   # Moving average smoothing
HOUGH_PARAMS = dict(
    rho=1,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=100,
    maxLineGap=300
)
```

---

## ❓ Troubleshooting

- **No video showing:** Check `VIDEO_SOURCE` and make sure your webcam or IP stream is reachable.
- **MJPEG not working on mobile:** Ensure browser supports MJPEG streams (Chrome, Firefox work well).
- **Too slow or laggy:** Reduce frame resolution or tune Hough/Canny parameters.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 🙌 Credits

Developed using:

- [OpenCV](https://opencv.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)