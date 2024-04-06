import math
import os
from collections import Counter
from pathlib import Path

import cv2
from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint
from using_mediapipe.video.picture_analyser import PictureAnalyser, get_relative_to_box, XY


def distance_normalized_keypoint(keypoint1: NormalizedKeypoint, keypoint2: NormalizedKeypoint):
    return math.sqrt((keypoint1.x - keypoint2.x) ** 2 + (keypoint1.y - keypoint2.y) ** 2)


def euclidean_distance(a, b):
    distance_temp = 0.0
    for i in range(len(a)):
        distance_temp += distance_normalized_keypoint(a[i], b[i])
    return distance_temp


class SimpleFacerec:
    known_encodings = []
    picture_analyser = None
    embeddings_file_name = None

    def __init__(self, embeddings_file_name, min_detection_confidence, model):
        self.known_encodings = []
        self.embeddings_file_name = embeddings_file_name
        self.picture_analyser = PictureAnalyser(min_detection_confidence, model)

    def get_image_encodings(self, path, name):
        file = str(os.path.join(path, name))
        img = cv2.imread(file)
        embeddings = self.picture_analyser.get_embeddings(img)
        relative_x_ys = get_relative_to_box(embeddings)
        if len(relative_x_ys) == 1:
            return relative_x_ys[0]
        else:
            print("Training images should contain one face" + file)

    def save_encodings_images(self, path):
        if Path(self.embeddings_file_name).is_file():
            raise UserWarning("embeddings file already exists")
        for root, dirs, files in os.walk(path):
            for filename in files:
                person_name = root.split('\\')[-1]
                print(person_name)
                if filename.lower().endswith(('.jpg', 'jpeg', '.png')):
                    print(filename)
                    frame_number = int(filename.split('_')[1])
                    enc = self.get_image_encodings(os.path.join(path, person_name), filename)
                    self.write_encoded_images(person_name, enc, frame_number, filename)

    def write_encoded_images(self, person_name, enc, frame_number, filename):
        if person_name is None or enc is None:
            return None
        file_object = open(self.embeddings_file_name, "a")
        file_object.write(f"{person_name};")
        for key_point in enc:
            file_object.write(f"{key_point.x},{key_point.y};")
        file_object.write(f"{frame_number};")
        file_object.write(f"{filename};")
        file_object.write("\n")
        file_object.close()
        self.known_encodings.append((person_name, enc))

    def read_encoded_images(self):
        try:
            file_object = open(self.embeddings_file_name, "r")
        except OSError or FileNotFoundError:
            print("No such file or directory")
            raise UserWarning("No embeddings file found, create this first")

        lines = file_object.readlines()
        if len(lines) == 0:
            raise UserWarning("No embeddings, create this first")
        self.known_encodings = []
        for line in lines:
            line_parts = line.split(";")
            person_name = line_parts[0]
            key_points = []
            for i in range(1, 7):
                x, y = line_parts[i].split(",")
                key_points.append(XY(x=float(x), y=float(y)))
            frame_number = line_parts[7]
            filename = line_parts[8]
            self.known_encodings.append((person_name, key_points, frame_number, filename))

    def face_k_lowest_distances(self, key_points, k, frame=None):
        arr_temp = []
        for person_name_temp, enc, frame_number, filename in self.known_encodings:
            distance = euclidean_distance(key_points, enc)
            arr_temp.append((float(distance), person_name_temp, frame_number))

        array_names = []
        # print(sorted(arr_temp))
        for (d, n, fr) in sorted(arr_temp)[:k]:
            if d < 25:
                if int(fr) == int(frame):
                    print(fr)
                array_names.append(n)

        counts = Counter(array_names)
        try:
            return counts.most_common(1)[0][0]
        except IndexError:
            return "No faces found"
