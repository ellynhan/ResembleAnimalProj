import cv2 as cv
import dlib
import numpy as np
from os import listdir
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
from urllib.parse import quote_plus

def crop(img, url):
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(img)
    if len(faces)!=0 :
        f = faces[0]
        x = f.left() if f.left()>0 else 0
        y = f.top() if f.top()>0 else 0
        w = f.right() - f.left()
        h = f.bottom() - f.top()
        #print('before:',img.shape)
        cropped = img[y:y+h, x:x+w]
        #print('after:',cropped.shape)
#        cv.imshow('cropped',cropped)
#        cv.waitKey(0)
        cv.imwrite(url,cropped)
    else:
        print("not detected"+url)




baseUrl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='
animals = ['cat','dog','rabbit','turtle']
names = [['안소희', '정수정', '한예슬'],['박보영','방민아','이민정'],['나연','수지','아이유'],['마마무 솔라','예리','하연수']]
#plusUrl = input('검색어 입력: ')
crawl_num =30
#int(input('크롤링할 갯수 입력(최대 50개): '))

for index, animal in enumerate(animals):
    for person in names[index]:
        url = baseUrl + quote_plus(person)
        html = urlopen(url)
        soup = bs(html, "html.parser")
        img = soup.find_all(class_='_img')
        
        n = 1
        for i in img:
            imgUrl = i['data-source']
            with urlopen(imgUrl) as f:
                saveUrl = './image/'+animal+'/'+person + str(n)+'.jpg'
                getImage = f.read()
                arr = np.asarray(bytearray(getImage), dtype=np.uint8)
                img = cv.imdecode(arr, -1) # 'Load it as it is'
                #cv.imshow('img',img)
                #cv.waitKey(0)
                #h.write(getImage)
                crop(img,saveUrl)
                    
            n += 1
            if n > crawl_num:
                break

