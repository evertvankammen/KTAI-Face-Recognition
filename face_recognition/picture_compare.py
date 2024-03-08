import os

import cv2
import face_recognition

image_path = os.path.join("..", "data", "pictures")


def get_image_encodings(path, name):
    file = str(os.path.join(path, name))
    img = cv2.imread(file)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return name, img, face_recognition.face_encodings(rgb_img)


class SimpleFacerec:
    encodings = []

    def __init__(self):
        self.encodings = []

    def load_encoded_images(self, path):
        files = os.listdir(path)
        for file_name in files:
            if (file_name.endswith(".jpg") or
                    file_name.endswith(".png")
                    or file_name.endswith("jpeg")):
                name, img, enc = get_image_encodings(path, file_name)
                self.encodings.append((name, img, enc))

    def detect_faces(self, encoding):
        found = []
        for name, img, enc in self.encodings:
            r = face_recognition.compare_faces([enc[0]], encoding[0])
            if r[0]:
                found.append(name)
        return found

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
