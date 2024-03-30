import os
import random

import cv2

from face_detector import SimpleFacerec
from using_mediapipe.video.picture_analyser import PictureAnalyser

IMAGE_PATH = os.path.join("..", "..", "data", "pictures")
#IMAGE_PATH_EMBEDDINGS = os.path.join("..", "..", "data", "embeddings")
IMAGE_PATH_EMBEDDINGS = os.path.join("..", "..", "encoding_with_pickle", "dataset")
VIDEO_PATH = os.path.join("..", "..", "data", "movies")
IMAGE_SAVE_PATH = os.path.join("..", "..", "data", "embeddings")
MIN_DETECTION_CONFIDENCE = 0.75
SAVE_EMBEDDING_RATE = 5


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


def run_face_detector(file_name):
    sfc = SimpleFacerec()
    sfc.read_encoded_images()
    picture_analyser = PictureAnalyser(model=('short_range_model', 0))

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
            try:
                frame_copy = picture_analyser.analyse_frame(frame.copy(), sfc)
                cv2.imshow("Annotated", frame_copy)
            except Exception as e2:
                print(e2)
                cv2.imshow("Annotated", frame)
        else:
            break

        frame_counter = frame_counter + 1
        frame_duration_counter = frame_duration_counter + frame_duration
