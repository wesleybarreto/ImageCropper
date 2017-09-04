import numpy as np
import cv2
import os

for filenames in os.walk('OriginalPhotos'):
    for i in filenames[2]:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        img = cv2.imread('OriginalPhotos/' + str(i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x-30,y-30),(x+w+30,y+h+30),(255,255,255),0, 0)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        photo = img[y-30:y+h+30, x-30:x+w+30]
        cv2.imwrite('CroppedPhotos/' + str(i), photo)
