from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
import mediapipe as mp
import pandas as pd
import datetime
import time

model = load_model('gender_detection.h5')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

webcam = cv2.VideoCapture(0)

classes = ['man', 'woman']

man_count = 0
woman_count = 0

previous_faces = []

log_df = pd.DataFrame(columns=['SNo', 'Message Received', 'Date', 'Time'])

last_help_time = 0

def is_new_face(face, previous_faces, threshold=50):
    (startX, startY, endX, endY) = face
    for (p_startX, p_startY, p_endX, p_endY) in previous_faces:
        if abs(startX - p_startX) < threshold and abs(startY - p_startY) < threshold and \
           abs(endX - p_endX) < threshold and abs(endY - p_endY) < threshold:
            return False
    return True

def is_victory_sign(hand_landmarks):
    if len(hand_landmarks) == 0:
        return False
    
    landmarks = hand_landmarks[0].landmark
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]

    if (thumb_tip.y < index_tip.y < middle_tip.y and
        abs(index_tip.x - middle_tip.x) < 0.1):
        return True
    return False

cv2.namedWindow("gender detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("gender detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

serial_number = 1

while webcam.isOpened():
    status, frame = webcam.read()
    faces, confidence = cv.detect_face(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_faces = []
    current_man_count = 0
    current_woman_count = 0
    detected_victory_by_woman = False

    for idx, f in enumerate(faces):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        face_crop = np.copy(frame[startY:endY, startX:endX])

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        conf = model.predict(face_crop)[0] 
        gender_idx = np.argmax(conf)
        label = classes[gender_idx]

        if is_new_face(f, previous_faces):
            if label == 'man':
                man_count += 1
            else:
                woman_count += 1

        if label == 'man':
            current_man_count += 1
        else:
            current_woman_count += 1

        label_text = "{}: {:.2f}%".format(label, conf[gender_idx] * 100)
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, label_text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        current_faces.append(f)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if is_victory_sign([hand_landmarks]) and label == 'woman':
                    detected_victory_by_woman = True

    count_text = f"Men: {current_man_count}, Women: {current_woman_count}"
    cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if current_woman_count == 1 and current_man_count >= 1:
        alert_text = "Alert! Alone woman detected"
        cv2.putText(frame, alert_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if detected_victory_by_woman:
        cv2.putText(frame, 'HELP!', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 139), 2)

        current_time = time.time()
        if current_time - last_help_time >= 60:
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            log_df.loc[len(log_df)] = [serial_number, 'HELP! detected', now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")]
            serial_number += 1
            last_help_time = current_time

    cv2.imshow("gender detection", frame)

    previous_faces = current_faces

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log_df.to_excel('detection_log.xlsx', index=False)

webcam.release()
cv2.destroyAllWindows()
