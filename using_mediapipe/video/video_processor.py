import math
import os
import random

import cv2
from typing_extensions import deprecated

from using_mediapipe.video.face_detector import SimpleFacerec
from using_mediapipe.video.picture_analyser import PictureAnalyser, get_box

IMAGE_PATH = os.path.join("..", "..", "data", "pictures")

IMAGE_PATH_EMBEDDINGS = os.path.join("..", "..", "encoding_with_pickle", "dataset")

MIN_DETECTION_CONFIDENCE = 0.75
SAVE_EMBEDDING_RATE = 10


def save_encodings_to_file(embeddings_file_name, file_location, min_detection_confidence, model):
    sfc = SimpleFacerec(embeddings_file_name, min_detection_confidence, model)
    sfc.save_encodings_images(file_location)


def save_image(frame, embeddings, frame_duration_counter, image_save_path):
    for nr in range(len(embeddings)):
        emb = embeddings[nr]
        image_rows, image_cols, _ = emb.shape
        (s_x, s_y), (e_x, e_y) = get_box(emb)
        width = (e_x - s_x)
        height = (e_y - s_y)

        margin = math.floor(width * .50)
        x, y = s_x - margin, s_y - margin
        w, h = width + 2 * margin, height + 2 * margin
        try:
            box_image = frame[y: y + h, x: x + w]
            success = cv2.imwrite(os.path.join(image_save_path, f"image_{round(frame_duration_counter)}_{nr}.jpg"),
                                  box_image)
            print(success)
        except cv2.error as e1:
            print(e1)
        except Exception as e2:
            print(e2)


def take_pictures(video_file_name, min_detection_confidence, model, sample_chance: int, image_save_path):
    picture_analyser = PictureAnalyser(min_detection_confidence=min_detection_confidence, model=model)
    video_file = video_file_name
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1000 / fps
    frame_counter = 0
    frame_duration_counter = 0
    frames_sampled = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if random.randint(1, 100) <= sample_chance:
                try:
                    frames_sampled = frames_sampled + 1
                    embeddings = picture_analyser.get_embeddings(frame)
                    save_image(frame, embeddings, frame_counter,image_save_path=image_save_path)
                except Exception as e2:
                    print(e2)
            frame_counter = frame_counter + 1
            frame_duration_counter = frame_duration_counter + frame_duration
        else:
            break

    return frame_counter, frames_sampled, fps


def run_face_detector(video_file_name, embeddings_file_name, min_detection_confidence, model):
    """
    Runs the face detector on a video file and returns the number of frames processed and the frames per second (fps) of the video.

    Parameters:
    - video_file_name (str): The name of the video file.
    - embeddings_file_name (str): The name of the file containing encoded images for face recognition.
    - min_detection_confidence (float): The minimum confidence level for face detection.
    - model: The face detection model.

    Returns:
    - frame_counter (int): The number of frames processed in the video.
    - fps (float): The frames per second (fps) of the video.

    """
    sfc = SimpleFacerec(embeddings_file_name, min_detection_confidence, model)
    sfc.read_encoded_images()
    picture_analyser = PictureAnalyser(min_detection_confidence=min_detection_confidence, model=model)
    video_file = os.path.join(VIDEO_PATH, video_file_name)
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
                frame_copy = picture_analyser.analyse_frame(frame.copy(), sfc, round(frame_duration_counter))
                cv2.imshow("Annotated", frame_copy)
            except Exception as e2:
                print(e2)
                cv2.imshow("Annotated", frame)
        else:
            break

        frame_counter = frame_counter + 1
        frame_duration_counter = frame_duration_counter + frame_duration
    return frame_counter, fps
