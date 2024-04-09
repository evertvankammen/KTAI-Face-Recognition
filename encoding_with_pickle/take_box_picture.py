import math
import os
from typing import Tuple, Union

import cv2


def save_partial_image(frame, box, name, experiment_directory, frame_count):
    (top, right, bottom, left) = box
    image_rows, image_cols, _ = frame.shape
    (s_x, s_y, e_x, e_y) = (left, top, right, bottom)
    width = (e_x - s_x)
    height = (e_y - s_y)
    margin = math.floor(width * .50)

    x, y = s_x - margin, s_y - margin
    w, h = width + 2 * margin, height + 2 * margin
    try:
        box_image = frame[y: y + h, x: x + w]
        file_name = f"{name}_{frame_count}.jpg"
        cv2.imwrite(os.path.join(experiment_directory, file_name), box_image)
        print(f"saving image to {file_name}")
    except cv2.error as e1:
        print(e1)
    except Exception as e2:
        print(e2)
