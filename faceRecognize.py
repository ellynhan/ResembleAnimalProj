import face_recognition
import cv2
import os
import numpy as np

class FaceRecog():
    def __init__(self):
        self.image = cv2.imread('target.jpg')
        self.known_face_encodings = []
        self.known_face_names = []
        dirname = 'known'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                face_encoding = face_recognition.face_encodings(img)[0]
                self.known_face_encodings.append(face_encoding)

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_image = True
        
    def __del__(self):
        del self.image
        
    def get_image(self):
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_img = self.image[:, :, ::-1]
        # Only process every other frame of video to save time
        if self.process_this_image:
            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(rgb_img)
            self.face_encodings = face_recognition.face_encodings(rgb_img, self.face_locations)
            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)
                print(distances)
                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                name = "Unknown"
                if min_value < 0.6:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]

                self.face_names.append(name)

        self.process_this_image = not self.process_this_image

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Draw a box around the face
            cv2.rectangle(self.image, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(self.image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(self.image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return self.image
        
if __name__ == '__main__':
    face_recog = FaceRecog()
    print(face_recog.known_face_names)
    img = face_recog.get_image()
    cv2.imshow("Image",img)

cv2.waitKey(0)
cv2.destroyAllWindows()
