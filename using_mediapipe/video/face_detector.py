import math
import os
from collections import Counter
from pathlib import Path

import cv2
from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint

from using_mediapipe.video.picture_analyser import PictureAnalyser, get_relative_to_box, XY


def distance_normalized_keypoint(keypoint1: NormalizedKeypoint, keypoint2: NormalizedKeypoint):
    """

    Calculates the Euclidean distance between two normalized keypoints.

    Parameters:
        keypoint1 (NormalizedKeypoint): The first normalized keypoint.
        keypoint2 (NormalizedKeypoint): The second normalized keypoint.

    Returns:
        float: The Euclidean distance between the two keypoints.

    """
    return math.sqrt((keypoint1.x - keypoint2.x) ** 2 + (keypoint1.y - keypoint2.y) ** 2)


def euclidean_distance(a, b):
    """
    Calculates the Euclidean distance between two vectors.

    :param a: First vector.
    :type a: list or tuple

    :param b: Second vector.
    :type b: list or tuple

    :return: The Euclidean distance between vectors a and b.
    :rtype: float
    """
    distance_temp = 0.0
    for i in range(len(a)):
        distance_temp += distance_normalized_keypoint(a[i], b[i])
    return distance_temp


class SimpleFacerec:
    """
    ## SimpleFacerec

    This class represents a simple face recognition system.

    ### Properties

    - **known_encodings**: list of tuples - A list of known face encodings.
    - **picture_analyser**: PictureAnalyser - An object of the PictureAnalyser class responsible for analyzing input images.
    - **embeddings_file_name**: str - The name of the embeddings file.

    ### Methods

    - **__init__(embeddings_file_name, min_detection_confidence, model)**: Initializes a new instance of the SimpleFacerec class.
      - Parameters:
        - **embeddings_file_name** (str) - The name of the embeddings file.
        - **min_detection_confidence** (float) - The minimum confidence value for face detection.
        - **model** - The face detection model to be used.

    - **get_image_encodings(path, name)**: Returns the relative xy coordinates of a detected face in an input image.
      - Parameters:
        - **path** (str) - The path to the image file.
        - **name** (str) - The name of the image file.
      - Returns:
        - The relative xy coordinates of a detected face.

    - **save_encodings_images(path)**: Saves the face encodings and associated images to the embeddings file.
      - Parameters:
        - **path** (str) - The path of the directory containing the images.
      - Returns:
        - None
      - Raises:
        - UserWarning: If the embeddings file already exists.

    - **write_encoded_images(person_name, enc, frame_number, filename)**: Writes the face encodings to the embeddings file.
      - Parameters:
        - **person_name** (str) - The name of the person in the image.
        - **enc** - The face encodings.
        - **frame_number** (str) - The frame number of the image.
        - **filename** (str) - The name of the image file.
      - Returns:
        - None

    - **read_encoded_images()**: Reads the face encodings from the embeddings file.
      - Returns:
        - None
      - Raises:
        - UserWarning: If the embeddings file is not found or is empty.

    - **face_k_lowest_distances(key_points, k, frame=None)**: Performs face recognition and returns the k most similar faces based on the Euclidean distance.
      - Parameters:
        - **key_points** - The face encodings to be compared with the known encodings.
        - **k** (int) - The number of closest matches to be returned.
        - **frame** - The frame number for filtering results (optional).
      - Returns:
        - The name of the most similar face, or "No faces found" if there are no matches.

    """
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
