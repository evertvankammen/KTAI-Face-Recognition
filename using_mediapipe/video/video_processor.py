import math
import os
import random

import cv2

from using_mediapipe.video.picture_analyser import PictureAnalyser, get_box


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
                    save_image(frame, embeddings, frame_counter, image_save_path=image_save_path)
                except Exception as e2:
                    print(e2)
            frame_counter = frame_counter + 1
            frame_duration_counter = frame_duration_counter + frame_duration
        else:
            break

    return frame_counter, frames_sampled, fps
