import numpy
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
def faceDetect():
    eye_detect = True
    face_cascade = cv2.CascadeClassifier('/Users/hanjaewon/Documents/opencvproj/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/Users/hanjaewon/Documents/opencvproj/opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    info = ''

    try:
        cap = cv2.VideoCapture(0)
        print("load camera")
    except:
        print('fail to load camera')
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if eye_detect:
            info = 'Eye Detection On'
        else:
            info = 'Eye Detection Off'
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        cv2.putText(frame, info, (5,15),font,0.5,(255,0,255),1)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame,'Detected Face', (x-5, y-5), font, 0.5, (255, 255,0), 2)
            if eye_detect:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imshow('frame',frame)
        k = cv2.waitKey(30)
        if k == ord('i'): #i키를 입력함으로써 눈을 인식할지 말지 변경 
            eye_detect = not eye_detect
        if k == 27:
            break;

    cap.release()
    cv2.destroyAllWindows()

faceDetect()
print("print something!")
