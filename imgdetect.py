import numpy
import cv2

face_cascade = cv2.CascadeClassifier('/Users/hanjaewon/Documents/opencvproj/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('/Users/hanjaewon/Documents/opencvproj/opencv/data/haarcascades/haarcascade_eye.xml')

img = cv2.imread('./img/manypeople.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.05, 10)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
