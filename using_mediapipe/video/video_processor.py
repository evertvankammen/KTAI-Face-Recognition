import math
import os
import random

import cv2

from face_detector import SimpleFacerec
from using_mediapipe.video.picture_analyser import PictureAnalyser, normalized_to_pixel_coordinates, get_box

IMAGE_PATH = os.path.join("..", "..", "data", "pictures")
#IMAGE_PATH_EMBEDDINGS = os.path.join("..", "..", "data", "embeddings")
IMAGE_PATH_EMBEDDINGS = os.path.join("..", "..", "encoding_with_pickle", "dataset")
VIDEO_PATH = os.path.join("..", "..", "data", "movies")
IMAGE_SAVE_PATH = os.path.join("..", "..", "data",  "ground_truth")
MIN_DETECTION_CONFIDENCE = 0.75
SAVE_EMBEDDING_RATE = 10


def save_encodings_to_file(embeddings_file_name, file_location,min_detection_confidence, model):
    sfc = SimpleFacerec(embeddings_file_name, min_detection_confidence, model)
    sfc.save_encodings_images(file_location)


def save_random_image(frame, embeddings, frame_duration_counter):
    for nr in range(len(embeddings)):
        emb = embeddings[nr]
        image_rows, image_cols, _ = emb.shape
        (s_x, s_y), (e_x, e_y) = get_box(emb)
        width = (e_x - s_x)
        height = (e_y - s_y)

        margin = math.floor(width * .50)
        #if width < 150:
        #    continue
        # name = detection.categories[0].display_name
        x, y = s_x - margin, s_y - margin
        w, h = width + 2*margin, height + 2*margin
        try:
            box_image = frame[y: y + h, x: x + w]
            success = cv2.imwrite(os.path.join(IMAGE_SAVE_PATH, f"image_{round(frame_duration_counter)}_{nr}.jpg"), box_image)
            print(success)
        except cv2.error as e1:
            print(e1)
        except Exception as e2:
            print(e2)


def take_pictures(video_file_name,min_detection_confidence, model, sample_chance: int):
    picture_analyser = PictureAnalyser(min_detection_confidence=min_detection_confidence,model=model)
    video_file = os.path.join(VIDEO_PATH, video_file_name)
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
                    save_random_image(frame, embeddings, frame_counter)
                except Exception as e2:
                    print(e2)
            frame_counter = frame_counter + 1
            frame_duration_counter = frame_duration_counter + frame_duration
        else:
            break

    return frame_counter, frames_sampled, fps

def run_face_detector(video_file_name, embeddings_file_name,min_detection_confidence, model):
    sfc = SimpleFacerec(embeddings_file_name, min_detection_confidence, model)
    sfc.read_encoded_images()
    picture_analyser = PictureAnalyser(min_detection_confidence=min_detection_confidence,model=model)
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
