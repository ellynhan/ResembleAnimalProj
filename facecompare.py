import face_recognition

iu = face_recognition.load_image_file('./img/iu/iu.jpg')
iu_face_encording = face_recognition.face_encodings(iu)[0]

unknownface = face_recognition.load_image_file('./img/unknown/unnamed1.jpg')
unknown_face_encording = face_recognition.face_encodings(unknownface)[0]

# 얼굴비교
result = face_recognition.compare_faces([iu_face_encording], unknown_face_encording)

if result[0]:
    print('아이유입니다.')
else:
    print('아이유가 아닙니다.')
