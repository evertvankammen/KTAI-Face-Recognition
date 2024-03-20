import math
import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

base_options = python.BaseOptions(model_asset_path='../models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       running_mode=VisionTaskRunningMode.IMAGE,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


def get_image_encodings(path, name):
    file = str(os.path.join(path, name))
    person_name = os.path.splitext(name)[0]
    img = cv2.imread(file)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detection_result = detector.detect(image)

    for list in detection_result.face_landmarks:
        print(len(list))

    if len(detection_result.face_landmarks) == 1:
        return person_name, img, detection_result.face_landmarks[0]
    return person_name, img, None


def distance_normalized_keypoint(keypoint1: NormalizedLandmark, keypoint2: NormalizedLandmark):
    return (keypoint1.x - keypoint2.x) ** 2 + (keypoint1.y - keypoint2.y) ** 2 + (keypoint1.z - keypoint2.z) ** 2


def euclidean_distance(a, b):
    distance_temp = 0.0
    for i in range(len(a)):
        distance_temp += distance_normalized_keypoint(a[i], b[i])

    return distance_temp


class LandmarkFaceRec:
    known_encodings = []
    file_name = "embeddings_lm.csv"

    def __init__(self):
        self.known_encodings = []

    def save_encodings_images(self, path):
        files = os.listdir(path)
        for file_name in files:
            if (file_name.endswith(".jpg") or
                    file_name.endswith(".png")
                    or file_name.endswith("jpeg")):
                person_name, img, enc = get_image_encodings(path, file_name)
                if enc is not None:
                    self.write_encoded_images(person_name, enc)

    def write_encoded_images(self, person_name, enc):
        file_object = open(self.file_name, "a")
        file_object.write(f"{person_name};")
        for key_point in enc:
            file_object.write(f"{key_point.x},{key_point.y},{key_point.z};")
        file_object.write("\n")
        file_object.close()
        self.known_encodings.append((person_name, enc))

    def read_encoded_images(self):
        file_object = open(self.file_name, "r")
        lines = file_object.readlines()
        self.known_encodings = []
        for line in lines:
            person_name = line.split(";")[0]
            encodings = line.split(";")[1:-1]
            key_points = []
            for i in range(len(encodings)):
                x, y, z = encodings[i].split(",")
                key_points.append(NormalizedLandmark(x=float(x), y=float(y), z=float(z)))

            self.known_encodings.append((person_name, key_points))

    def face_lowest_distances(self, keypoints):
        min_element = None
        person_name = None
        for person_name_temp, enc in self.known_encodings:
            distance = euclidean_distance(keypoints, enc)
            print(person_name_temp, distance)
            if min_element is None or distance < min_element:
                min_element = distance
                person_name = person_name_temp
        return min_element, person_name
