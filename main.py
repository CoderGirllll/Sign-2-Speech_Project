import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import time
from collections import deque

# Load model
with open("sign_to_text/gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Text-to-speech engine
engine = pyttsx3.init()
prev_prediction = ""
prediction_buffer = deque(maxlen=15)  # Store last 15 predictions for smoothing

cap = cv2.VideoCapture(0)
print("Starting Sign-to-Speech System. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

            if len(landmarks) == 63:
                landmarks_np = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(landmarks_np)[0]

    # Add prediction to buffer
    if prediction:
        prediction_buffer.append(prediction)

        # When buffer is full, check most common prediction
        if len(prediction_buffer) == 15:
            most_common = max(set(prediction_buffer), key=prediction_buffer.count)
            if most_common != prev_prediction:
                print(f"Detected Sign: {most_common}")
                engine.say(most_common)
                engine.runAndWait()
                prev_prediction = most_common
                time.sleep(1.5)  # Avoid rapid repeats

    # Show prediction on frame
    cv2.putText(frame, f'Prediction: {prediction if prediction else ""}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
