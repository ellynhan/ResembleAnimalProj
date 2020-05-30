import sys
import os
import dlib
import glob
import numpy as np
import cv2

predictor_path = './shape_predictor_68_face_landmarks.dat'
faces_folder_path = './img/iu/'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
cnt = 0
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    
    img = dlib.load_rgb_image(f)
    rgb_img = img[:,:,::-1]
    
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        
        for b in range(68):
            cv2.circle(img, (shape.part(b).x, shape.part(b).y), 2, (255,0,0), 1)
            cv2.putText(img,str(b), (shape.part(b).x, shape.part(b).y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255,0), 1)
            
    cv2.imshow('result',rgb_img)
    cv2.imwrite('./img/result'+str(cnt)+'.jpg',rgb_img)
    cnt=cnt+1
    cv2.waitKey(0)
    cv2.destroyAllWindows()
