import cv2
import cvzone
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
with open('res_model.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

while True:
    img, frame = cap.read()
    if not img:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))
        face_normalized = face_resized / 255.0
        face_reshaped = np.reshape(face_normalized, (1, 128, 128, 1))
        gender_pred, age_pred = model.predict(face_reshaped)
        gender = "Male" if gender_pred[0] > 0.5 else "Female"
        age = age_pred[0][0]
        cvzone.putTextRect(frame, f'Gender: {gender}', (x, y-30), scale=1, thickness=1, colorR=(0,255,0), colorB=(0,0,255))
        cvzone.putTextRect(frame, f'Age: {int(age)}', (x, y-10), scale=1, thickness=1, colorR=(0,255,0), colorB=(0,0,255))
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()