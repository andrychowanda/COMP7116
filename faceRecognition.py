import cv2
import sys

#!pip install opencv-python-headless 

haarCascadeFrontal = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
haarCascadeEye = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")


#videoCapture = cv2.VideoCapture(0)
videoCapture = cv2.VideoCapture("contoh.mp4")  

while 1:
    _, frame = videoCapture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    faces = haarCascadeFrontal.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_gray = gray[y:y+h, x:x+w]
        face_ori = frame[y:y+h, x:x+w]
    
        eyes = haarCascadeEye.detectMultiScale(face_gray, 1.3, 6)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_ori, (ex,ey), (ex + ew,ey + eh), (255, 0, 0), 2)

    cv2.imshow("Simple Face Rec", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()


