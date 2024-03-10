import os

import cv2
import face_recognition
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint

base_options = python.BaseOptions(model_asset_path='../models/detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# 0 left eye, 1 right eye, 2 nose, 3 mouth,  4 left side face, 5 right side face

def get_image_encodings(path, name):
    file = str(os.path.join(path, name))
    person_name = os.path.splitext(name)[0]
    img = cv2.imread(file)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detection_result = detector.detect(image)
    if len(detection_result.detections) == 1:
        return person_name, img, detection_result.detections[0].keypoints
    return person_name, img, None


def distance_normalized_keypoint(keypoint1: NormalizedKeypoint, keypoint2: NormalizedKeypoint):
    return (keypoint1.x - keypoint2.x) ** 2 + (keypoint1.y - keypoint2.y) ** 2


def euclidean_distance(a, b):
    distance_temp = 0.0
    for i in range(len(a)):
        if i < 4:
            distance_temp += 2 * distance_normalized_keypoint(a[i], b[i])
        else:
            distance_temp += distance_normalized_keypoint(a[i], b[i])
        return distance_temp


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

    def face_lowest_distances(self, keypoints):
        min_element = None
        person_name = None
        for person_name_temp, img, enc in self.known_encodings:
            distance = euclidean_distance(keypoints, enc)
            if min_element is None or distance < min_element:
                min_element = distance
                person_name = person_name_temp
        return person_name

