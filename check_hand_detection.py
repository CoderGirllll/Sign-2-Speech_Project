# Install these packages if not already installed
# !pip install mediapipe opencv-python pyttsx3 --quiet

import cv2
import mediapipe as mp
import pyttsx3

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Text-to-Speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Set speech rate

# Placeholder gesture recognition logic
def recognize_gesture(landmarks):
    # Here you could use landmark geometry or ML model for real detection
    return "A"

# Open webcam
cap = cv2.VideoCapture(0)

prev_letter = ''
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    # Convert image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Check if hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            
            # Recognize the gesture
            recognized_letter = recognize_gesture(landmarks)

            # Speak the letter if changed
            if recognized_letter != prev_letter:
                print("Detected:", recognized_letter)
                tts_engine.say(recognized_letter)
                tts_engine.runAndWait()
                prev_letter = recognized_letter

    # Display the frame
    cv2.imshow("Sign to Speech", img)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()