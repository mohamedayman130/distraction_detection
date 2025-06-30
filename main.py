# main.py
import cv2
import numpy as np
import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse

app = FastAPI()

# ----- Lane Detection Configuration -----
VIDEO_SOURCE = "https://192.168.1.3:8080/video"
DEBUG = True
SMOOTHING_FRAMES = 10
CANNY_THRESHOLDS = (150, 200)
GAUSSIAN_KERNEL = (5, 5)

HOUGH_PARAMS = dict(
    rho=1,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=100,
    maxLineGap=300
)

prev_left_fits = []
prev_right_fits = []

# ----- Lane Detection Functions -----
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)

def extrapolate_line(fit, y_bottom, y_top):
    if fit is None:
        return None
    m, b = fit
    x_bottom = int(m * y_bottom + b)
    x_top = int(m * y_top + b)
    return [(x_bottom, y_bottom), (x_top, y_top)]

def fit_average(fit_list, new_fit):
    if new_fit is not None:
        fit_list.append(new_fit)
    if len(fit_list) > SMOOTHING_FRAMES:
        fit_list.pop(0)
    return np.mean(fit_list, axis=0) if fit_list else None

def separate_lines(lines, img_shape):
    left_points, right_points = [], []
    height, width = img_shape[:2]
    mid_x = width // 2

    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(slope) < 0.5:
                continue
            if slope < 0 and x1 < mid_x and x2 < mid_x:
                left_points.extend([(x1, y1), (x2, y2)])
            elif slope > 0 and x1 > mid_x and x2 > mid_x:
                right_points.extend([(x1, y1), (x2, y2)])

    def fit_line(points):
        if len(points) < 2:
            return None
        x_coords, y_coords = zip(*points)
        return np.polyfit(y_coords, x_coords, 1)

    return fit_line(left_points), fit_line(right_points)

def process_frame(frame):
    height, width = frame.shape[:2]
    roi_top = height // 2 + 5

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    edges = cv2.Canny(blur, *CANNY_THRESHOLDS)

    roi_vertices = np.array([[
        (0, height),
        (width // 2 - 200, roi_top),
        (width // 2 + 200, roi_top),
        (width, height)
    ]], dtype=np.int32)
    cropped_edges = region_of_interest(edges, roi_vertices)

    lines = cv2.HoughLinesP(cropped_edges, **HOUGH_PARAMS)
    left_fit, right_fit = separate_lines(lines, frame.shape)

    smoothed_left_fit = fit_average(prev_left_fits, left_fit)
    smoothed_right_fit = fit_average(prev_right_fits, right_fit)

    line_img = np.zeros_like(frame)
    if DEBUG and lines is not None:
        for line in lines[:30]:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    for fit, color in zip([smoothed_left_fit, smoothed_right_fit], [(0, 255, 0), (0, 255, 0)]):
        pts = extrapolate_line(fit, height, roi_top)
        if pts:
            cv2.line(line_img, pts[0], pts[1], color, 5)

    result = cv2.addWeighted(frame, 0.8, line_img, 1.0, 0)

    left_pts = extrapolate_line(smoothed_left_fit, height, roi_top)
    right_pts = extrapolate_line(smoothed_right_fit, height, roi_top)

    if left_pts and right_pts:
        polygon = np.array([left_pts[0], left_pts[1], right_pts[1], right_pts[0]], dtype=np.int32)
        overlay = result.copy()
        cv2.fillPoly(overlay, [polygon.reshape((-1, 1, 2))], (60, 200, 60))
        result = cv2.addWeighted(overlay, 0.3, result, 0.7, 0)

        lane_center = (left_pts[1][0] + right_pts[1][0]) // 2
        frame_center = frame.shape[1] // 2
        offset = lane_center - frame_center

        if abs(offset) < 25:
            direction_text = "Straight"
        elif offset < -20:
            direction_text = "Turn Left"
        else:
            direction_text = "Turn Right"

        cv2.putText(result, direction_text, (frame.shape[1] // 2 - 100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    return result

# ----- FastAPI Endpoints -----
@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html") as f:
        return f.read()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# ----- Frame Generator for MJPEG -----
def generate_frames():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    fps_counter = []

    while True:
        start_time = time.time()
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (720, 720))
        lane_frame = process_frame(frame)

        end_time = time.time()
        fps = 1.0 / max((end_time - start_time), 1e-6)
        fps_counter.append(fps)
        avg_fps = np.mean(fps_counter[-30:])

        if DEBUG:
            cv2.putText(lane_frame, f"FPS: {avg_fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        _, jpeg = cv2.imencode('.jpg', lane_frame)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        )

# ----- Run with Uvicorn -----
def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run()
