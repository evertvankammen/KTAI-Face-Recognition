import math
import os

import cv2


def save_partial_image(frame, box, name, experiment_directory, frame_count):
    """
        Saves a portion of an image as a separate file.

        Parameters:
            frame (numpy.ndarray): The full frame from which the portion is cropped.
            box (tuple): A tuple with four elements (top, right, bottom, left) defining the bounding box of the portion.
            name (str): The base name for the saved file.
            experiment_directory (str): The path to the directory where the file should be saved.
            frame_count (int): The frame number used as part of the file name.

        Returns:
            None

        Raises:
            cv2.error: If there is an error in saving the image file with OpenCV.
            Exception: If an unexpected error occurs in saving the image file.

        Notes:
            - The portion of the frame is determined by the bounding box and saved as a separate image file.
            - If there is an error in saving the image file, it is printed.
        """
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
