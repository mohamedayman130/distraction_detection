import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import threading
import time
import uvicorn

app = FastAPI()

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.8)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)

# Constants
LEFT_IRIS = 468
RIGHT_IRIS = 473
UPPER_LIP = 13
LOWER_LIP = 14
NOSE = 1
EAR_THRESH = 0.2
EAR_CONSEC_FRAMES = 5
HAND_DIST_THRESH = 120
HAND_CONSEC_FRAMES = 3
YAWN_THRESH = 38
YAWN_CONSEC_FRAMES = 4
DISTRACTED_THRESHOLD = 15

# State
ear_history = deque(maxlen=10)
yawn_history = deque(maxlen=10)
hand_history = deque(maxlen=10)

distract_counter = 0
eye_closed_frames = 0
hand_near_frames = 0
yawn_frames = 0

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

def generate_frames():
    global ear_history, yawn_history, hand_history
    global distract_counter, eye_closed_frames, hand_near_frames, yawn_frames

    while True:
        start_time = time.time()
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(cv2.resize(frame, (1280, 720)), 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        face_result = face_mesh.process(rgb)
        hand_result = hands.process(rgb)

        distracted_now = False

        if face_result.multi_face_landmarks:
            face = face_result.multi_face_landmarks[0].landmark

            def pt(i): return int(face[i].x * w), int(face[i].y * h)

            left_eye_pts = [pt(i) for i in [33, 160, 158, 133, 153, 144]]
            right_eye_pts = [pt(i) for i in [362, 385, 387, 263, 373, 380]]

            def euclidean(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))

            def ear(eye):
                A = euclidean(eye[1], eye[5])
                B = euclidean(eye[2], eye[4])
                C = euclidean(eye[0], eye[3])
                return (A + B) / (2.0 * C)

            avg_ear = (ear(left_eye_pts) + ear(right_eye_pts)) / 2
            ear_history.append(avg_ear)
            smoothed_ear = np.mean(ear_history)

            if smoothed_ear < EAR_THRESH:
                eye_closed_frames += 1
            else:
                eye_closed_frames = 0

            if eye_closed_frames >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, 'DROWSY (Eyes Closed)', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                distracted_now = True

            upper_lip = pt(UPPER_LIP)
            lower_lip = pt(LOWER_LIP)
            mouth_width = euclidean(pt(78), pt(308))
            lip_dist = euclidean(upper_lip, lower_lip)
            yawn_metric = lip_dist / (mouth_width + 1e-5) * 100

            yawn_history.append(yawn_metric)
            avg_yawn = np.mean(yawn_history)

            if avg_yawn > YAWN_THRESH:
                yawn_frames += 1
            else:
                yawn_frames = 0

            if yawn_frames >= YAWN_CONSEC_FRAMES:
                cv2.putText(frame, 'YAWNING', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                distracted_now = True

            mp_drawing.draw_landmarks(frame, face_result.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS)
        else:
            cv2.putText(frame, "FACE NOT DETECTED", (400, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Hand near face detection
        if hand_result.multi_hand_landmarks and face_result.multi_face_landmarks:
            nose = pt(NOSE)
            hand_close = False

            for hand_landmarks in hand_result.multi_hand_landmarks:
                for idx in [8, 12]:
                    fingertip = hand_landmarks.landmark[idx]
                    fx, fy = int(fingertip.x * w), int(fingertip.y * h)
                    if np.linalg.norm(np.array(nose) - np.array((fx, fy))) < HAND_DIST_THRESH:
                        hand_close = True
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_history.append(1 if hand_close else 0)

            if np.mean(hand_history) > 0.5:
                hand_near_frames += 1
            else:
                hand_near_frames = 0

            if hand_near_frames >= HAND_CONSEC_FRAMES:
                cv2.putText(frame, 'HAND NEAR FACE', (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                distracted_now = True

        distract_counter = min(distract_counter + 1, 30) if distracted_now else max(distract_counter - 1, 0)

        if distract_counter >= DISTRACTED_THRESHOLD:
            cv2.putText(frame, 'DISTRACTED!', (450, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, 'FOCUSED', (450, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)

        # FPS counter
        fps = 1 / (time.time() - start_time + 1e-5)
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Encode & stream
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
