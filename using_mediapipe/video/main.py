import os
import random

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python

from face_detector import SimpleFacerec
from using_mediapipe.shared.common import visualize

MODELS_DETECTOR_TFLITE = '../../models/detector.tflite'
IMAGE_PATH = os.path.join("..", "..", "data", "pictures")
IMAGE_PATH_EMBEDDINGS = os.path.join("..", "..", "data", "embeddings")
VIDEO_PATH = os.path.join("..", "..", "data", "movies")
IMAGE_SAVE_PATH = os.path.join("..", "..", "data", "embeddings")
MIN_DETECTION_CONFIDENCE = 0.75
SAVE_EMBEDDING_RATE = 5


def initialize_detector():
    base_options = mp.tasks.BaseOptions
    face_detector = mp.tasks.vision.FaceDetector
    options = mp.tasks.vision.FaceDetectorOptions
    vision_running_mode = mp.tasks.vision.RunningMode
    options = options(
        base_options=base_options(model_asset_path=MODELS_DETECTOR_TFLITE),
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        running_mode=vision_running_mode.VIDEO)
    return face_detector.create_from_options(options)


def save_encodings_to_file():
    sfc = SimpleFacerec()
    sfc.save_encodings_images(IMAGE_PATH_EMBEDDINGS)


def save_random_image(face_detector_result, image_copy, frame_duration_counter, save_with_found_name: bool = False):
    for nr in range(len(face_detector_result.detections)):
        detection = face_detector_result.detections[nr]
        if detection.categories[0].score < MIN_DETECTION_CONFIDENCE or random.randint(1, 100) > SAVE_EMBEDDING_RATE:
            continue
        bbox = detection.bounding_box
        name = detection.categories[0].display_name
        x, y = bbox.origin_x - 25, bbox.origin_y - 25
        w, h = bbox.width + 50, bbox.height + 50
        try:
            box_image = image_copy[y: y + h, x: x + w]
            if save_with_found_name:
                cv2.imwrite(os.path.join(IMAGE_SAVE_PATH, f"image_{round(frame_duration_counter)}_{name}_{nr}.jpg"),
                            box_image)
            else:
                cv2.imwrite(os.path.join(IMAGE_SAVE_PATH, f"image_{round(frame_duration_counter)}_{nr}.jpg"), box_image)
        except cv2.error as e1:
            print(e1)
        except Exception as e2:
            print(e2)


def run_face_detector(file_name, save_random_embeddings: bool = False, save_with_found_name: bool = False):
    sfc = SimpleFacerec()
    sfc.read_encoded_images()
    detector = initialize_detector()

    video_file = os.path.join(VIDEO_PATH, file_name)
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1000 / fps
    frame_counter = 0
    frame_duration_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            mp_image_temp = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_detector_result = detector.detect_for_video(mp_image_temp, round(frame_duration_counter))

            for nr in range(len(face_detector_result.detections)):
                result = face_detector_result.detections[nr]
                if result.categories[0].score >= MIN_DETECTION_CONFIDENCE:
                    # found_name = sfc.face_lowest_distances(result.keypoints)
                    found_name = sfc.face_k_lowest_distances(result.keypoints, 3)
                    result.categories[0].category_name = str(nr) + ' ' + found_name
                    result.categories[0].display_name = found_name

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            image_copy = np.copy(mp_image.numpy_view())
            annotated_image = visualize(image_copy, face_detector_result, MIN_DETECTION_CONFIDENCE)
            cv2.imshow("Annotated", annotated_image)
            if save_random_embeddings:
                save_random_image(face_detector_result, image_copy, frame_duration_counter, save_with_found_name)
        else:
            break

        frame_counter = frame_counter + 1
        frame_duration_counter = frame_duration_counter + frame_duration


if __name__ == '__main__':
    #save_encodings_to_file()  # enable if you want to save the encodings to a file
    run_face_detector('Friends.mp4', False, False)
