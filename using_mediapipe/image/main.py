import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from using_mediapipe.shared.common import visualize

MODELS_DETECTOR_TFLITE = '../../models/detector.tflite'
IMAGE_PATH = os.path.join("..", "..", "data", "pictures")
MIN_DETECTION_CONFIDENCE = 0.75

def initialize_detector():
    base_options = python.BaseOptions(model_asset_path=MODELS_DETECTOR_TFLITE)
    options = vision.FaceDetectorOptions(base_options=base_options)
    return vision.FaceDetector.create_from_options(options)


def run_face_detector(file_name):
    img = cv2.imread(os.path.join(IMAGE_PATH, file_name))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detector = initialize_detector()
    detection_result = detector.detect(image)

    for x in detection_result.detections:
        print(x.bounding_box)
        for y in x.keypoints:
            print(y)
        print(x.categories)

    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result, MIN_DETECTION_CONFIDENCE)
    cv2.imshow("annotated", annotated_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    run_face_detector('Joey Tribbiani.jpg')
