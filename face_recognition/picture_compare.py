import os

import cv2
import face_recognition

image_path = os.path.join("..", "data", "pictures")


def get_image_encodings(path, name):
    person_name = os.path.splitext(name)[0]
    file = str(os.path.join(path, name))
    img = cv2.imread(file)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return person_name, img, face_recognition.face_encodings(rgb_img)


class SimpleFacerec:
    known_encodings = []

    def __init__(self):
        self.known_encodings = []

    def load_encoded_images(self, path):
        files = os.listdir(path)
        for file_name in files:
            if (file_name.endswith(".jpg") or
                    file_name.endswith(".png")
                    or file_name.endswith("jpeg")):
                person_name, img, enc = get_image_encodings(path, file_name)
                self.known_encodings.append((person_name, img, enc))
        return len(self.known_encodings)

    def detect_faces(self, encoding):
        found = []
        for name, img, enc in self.known_encodings:
            r = face_recognition.compare_faces([enc[0]], encoding[0], tolerance=0.1)
            if r[0]:
                found.append(name)
        return found

    def face_lowest_distances(self, encoding):
        min_element = None
        person_name = None
        for person_name_temp, img, enc in self.known_encodings:
            distance = face_recognition.face_distance([enc[0]], encoding)
            if min_element is None or distance < min_element:
                min_element = distance
                person_name = person_name_temp
        return person_name




# et_image_encodings("Joey Tribbiani.jpg")
# et_image_encodings("Monica Geller.jpeg")
#
# g", img1)
# g", img2)
#
# Windows()
#
# _recognition.compare_faces([enc1[0]], enc2[0])
#
#
