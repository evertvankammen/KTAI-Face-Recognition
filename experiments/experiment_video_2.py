import math
import os
from typing import Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(
        image,
        detection_result
) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                           width, height)
            color, thickness, radius = (0, 255, 0), 2, 2
            cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image


BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the video mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='../models/detector.tflite'),
    running_mode=VisionRunningMode.VIDEO)

VIDEO_FILE = os.path.join("..", "data", "pictures", "Friends.mp4")

with FaceDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(VIDEO_FILE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = int(1000/25)
    print(fps)
    print(frame_duration)
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_counter = 0
    frame_duration_counter = 0
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            cv2.imshow('Frame', frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            face_detector_result = detector.detect_for_video(mp_image, frame_duration_counter)

            image_copy = np.copy(mp_image.numpy_view())
            annotated_image = visualize(image_copy, face_detector_result)
            cv2.imshow("Annotated", annotated_image)

            # cv2.waitKey(0)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

        frame_counter = frame_counter + 1
        frame_duration_counter = frame_duration_counter + frame_duration
