import math
import os
import random

import cv2
from using_mediapipe.video.picture_analyser import PictureAnalyser, get_box


def save_image(frame, embeddings, frame_duration_counter, image_save_path):
    """
    Save Image.

    Save an image with bounding box from a given frame and its corresponding embeddings.

    Parameters:
    - frame (numpy.ndarray): The frame from which the image will be saved.
    - embeddings (List[numpy.ndarray]): The list of embeddings for each detected face in the frame.
    - frame_duration_counter (int): The duration of the frame.
    - image_save_path (str): The path where the saved image will be stored.

    Returns:
    None.

    """
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
    """

    Parameters:
    - video_file_name (str): The name of the video file to be processed.
    - min_detection_confidence (float): The minimum detection confidence required for an object to be considered detected.
    - model (Model): The object detection model to be used for object detection.
    - sample_chance (int): The chance (in percentage) of sampling each frame for analysis.
    - image_save_path (str): The path where the analyzed images will be saved.

    Returns:
    - frame_counter (int): The total number of frames in the video file.
    - frames_sampled (int): The number of frames that were sampled for analysis.
    - fps (float): The frames per second of the video file.
    """
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

