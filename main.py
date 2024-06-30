from flask import Flask, render_template, Response, redirect, url_for
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder="template")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result')
def result():
    global last_status, last_probability
    return render_template('result.html', status=last_status, probability=last_probability)

# Initialize global variables
cap = None
last_status = ""
last_probability = 0

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

def draw_fancy_progress_bar(image, progress, position=(50, 50), size=(200, 20), bar_color=(245, 117, 16), bg_color=(50, 50, 50)):
    x, y = position
    w, h = size
    cv2.rectangle(image, (x, y), (x + w, y + h), bg_color, -1)
    cv2.rectangle(image, (x, y), (x + int(w * progress), y + h), bar_color, -1)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)

def generate_frames():
    global last_status, last_probability, cap
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.face_landmarks:
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                          mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            try:
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

                row = pose_row + face_row

                X = pd.DataFrame([row])
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                prob_percentage = round(body_language_prob[np.argmax(body_language_prob)] * 100, 2)
                progress = body_language_prob[np.argmax(body_language_prob)]

                last_status = body_language_class
                last_probability = prob_percentage

                draw_fancy_progress_bar(image, progress, position=(50, 90), size=(300, 40))

                cv2.putText(image, f'STATUS: {body_language_class.split(" ")[0]}', (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f'PROBABILITY: {prob_percentage}%', (50, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error: {e}")

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_capture')
def stop_capture():
    global cap
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
    return redirect(url_for('result'))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
