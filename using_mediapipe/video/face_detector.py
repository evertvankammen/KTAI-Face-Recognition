import math
import os
import pickle
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint

MODELS_DETECTOR_TFLITE = '../../models/detector.tflite'
EMBEDDINGS_FILE = "embeddings.csv"

base_options = python.BaseOptions(model_asset_path=MODELS_DETECTOR_TFLITE)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)


def get_image_encodings(path, name):
    file = str(os.path.join(path, name))
    img = cv2.imread(file)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detection_result = detector.detect(image)
    if len(detection_result.detections) == 1:
        return detection_result.detections[0].keypoints
    return None


def distance_normalized_keypoint(keypoint1: NormalizedKeypoint, keypoint2: NormalizedKeypoint):
    return (keypoint1.x - keypoint2.x) ** 2 + (keypoint1.y - keypoint2.y) ** 2


def euclidean_distance(a, b):
    distance_temp = 0.0
    for i in range(len(a)):
        distance_temp += distance_normalized_keypoint(a[i], b[i])
    return math.sqrt(distance_temp)


class SimpleFacerec:
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
            file_object.write(f"{key_point.x},{key_point.y};")
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
            key_points = []
            for i in range(len(encodings)):
                x, y = encodings[i].split(",")
                key_points.append(NormalizedKeypoint(x=float(x), y=float(y)))

            self.known_encodings.append((person_name, key_points))

    def face_k_lowest_distances(self, key_points, k):
        arr_temp = []
        for person_name_temp, enc in self.known_encodings:
            distance = euclidean_distance(key_points, enc)
            arr_temp.append((float(distance), person_name_temp))
        array_names = []

        for (d, n) in sorted(arr_temp)[:k]:
            array_names.append(n)

        counts = Counter(array_names)
        print(counts)
        return counts.most_common(1)[0][0]
