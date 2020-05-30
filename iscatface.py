import sys
import os
import dlib
import glob
import numpy as np
import cv2

predictor_path = './shape_predictor_68_face_landmarks.dat'
faces_folder_path = './img/dog/'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")): #폴더 내에 있는 파일 개수만큼 반복문 돌기
    
    img = dlib.load_rgb_image(f)
    #rgb_img = img[:,:,::-1] #dlib로 인해 얼굴 색이 이상한거 변환
    height,width,channel = img.shape
    
    dets = detector(img, 1)
    #print("Number of faces detected: {}".format(len(dets))) 인식된 얼굴이 두 개 이상일 때만 사용하기
    #for k,d in enumerate(dets):
    
    
    shape = predictor(img, dets[0]) #여러개일때는 그냥 d
    #print("왼쪽 눈 뒷쪽: {}, 왼쪽 눈 앞쪽: {},  오른쪽 눈 뒷쪽: {}, 오른쪽 눈 앞쪽: {}".format(shape.part(36),shape.part(39),shape.part(45),shape.part(42)))
    angleY = shape.part(39).y - shape.part(42).y
    angleX = shape.part(42).x - shape.part(39).x
    angle = (angleY/angleX)
    
    #기울기가 -0.05에서 0.05사이라면 기울기 변환을 하지 않는다
    #기울기가 음수라면 +로 회전 양수라면 -로 회전
    #회전을 하면 기준 값 내에 들 때까지 회전
    cnt = 0
    while  angle>0.05 or angle<-0.05: #기존 범위를 벗어 났을 때
            transY = shape.part(39).y + shape.part(42).y
            transX = shape.part(39).x + shape.part(42).y
            if angle < 0:
                matrix = cv2.getRotationMatrix2D((transX/2, transY/2), 2, 1) #기울기값이 증가함
            else:
                matrix = cv2.getRotationMatrix2D((transX/2, transY/2), -2, 1) #기울기값이 감소함
            img = cv2.warpAffine(img,matrix,(width,height))
            shape = predictor(img, dets[0]) #회전된 이미지를 shape에 넣기.
            angleY = shape.part(39).y - shape.part(42).y
            angleX = shape.part(42).x - shape.part(39).x
            angle = (angleY/angleX)
            cnt = cnt + 1
        
    print("왼쪽 눈이 얼마나 올라가 있나요?:{}".format(shape.part(29).y-shape.part(36).y))
    print("오른쪽 눈은 얼마나 올라가 있나요?:{}".format(shape.part(42).y-shape.part(45).y))
    
    '''print("변환 기울기: {}, 회전횟수: {}".format(angle,cnt))
    cv2.imshow('dst',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    
    
    
        
        
        
        
        
        
    '''for b in range(68):
        cv2.circle(img, (shape.part(b).x, shape.part(b).y), 2, (255,0,0), 1)
        cv2.putText(img,str(b), (shape.part(b).x, shape.part(b).y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255,0), 1)
            
    cv2.imshow('result',rgb_img)
    cv2.imwrite('./img/result'+str(cnt)+'.jpg',rgb_img)
    cnt=cnt+1
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

