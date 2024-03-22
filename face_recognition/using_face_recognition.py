import os
from collections import Counter
from pathlib import Path

import cv2
import face_recognition

EMBEDDINGS_FILE = "embeddings.csv"


def get_image_encodings(path, name):
    file = str(os.path.join(path, name))
    img = cv2.imread(file)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_recognition.face_encodings(rgb_img)
    if len(result) == 1:  # make sure training picture have 1 face
        return result[0]
    return None


class UsingFaceRecognition:
    known_encodings = []

    def __init__(self):
        self.known_encodings = []

    def save_encodings_images(self, path):
        if Path(EMBEDDINGS_FILE).is_file():
            raise UserWarning("embeddings file already exists")
        for root, dirs, files in os.walk(path):
            for filename in files:
                person_name = root.split('\\')[-1]
                print(person_name)
                if filename.lower().endswith(('.jpg', 'jpeg', '.png')):
                    print(filename)
                    enc = get_image_encodings(os.path.join(path, person_name), filename)
                    self.write_encoded_images(person_name, enc)

    def write_encoded_images(self, person_name, enc):
        if person_name is None or enc is None:
            return None
        file_object = open(EMBEDDINGS_FILE, "a")
        file_object.write(f"{person_name};")
        for key_point in enc:
            file_object.write(f"{key_point};")
        file_object.write("\n")
        file_object.close()
        self.known_encodings.append((person_name, enc))

    def read_encoded_images(self):
        try:
            file_object = open(EMBEDDINGS_FILE, "r")
        except OSError or FileNotFoundError:
            print("No such file or directory")
            raise UserWarning("No embeddings file found, create this first")
        lines = file_object.readlines()
        if len(lines) == 0:
            raise UserWarning("No embeddings, create this first")
        self.known_encodings = []
        for line in lines:
            person_name = line.split(";")[0]
            encodings = line.split(";")[1:-1]
            encodings = [float(x) for x in encodings]
            self.known_encodings.append((person_name, encodings))

    def face_k_lowest_distances(self, encoding, k):
        arr_temp = []
        for person_name_temp, enc in self.known_encodings:
            distance = face_recognition.face_distance([enc], encoding)
            if distance < 0.55:
                arr_temp.append((distance, person_name_temp))

        if len(arr_temp) == 0:
            return None
        array_names = []
        sorted_list = sorted(arr_temp)[:k]
        for (d, n) in sorted_list:
            array_names.append(n)

        print(sorted_list)
        counts = Counter(array_names)
        print(counts)
        return counts.most_common(1)[0][0]
