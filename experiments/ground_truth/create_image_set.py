import os

from using_mediapipe.video.video_processor import take_pictures


def make_set():
    """

    The `make_set` method is responsible for creating a set of ground truth images from a given video file.

    Parameters:
    ----------
    None

    Returns:
    -------
    None

    Example Usage:
    --------------
    make_set()
    """
    video_file = os.path.join("..", "..", "data", "pictures", "Friends.mp4")
    min_detection_confidence = 0.10
    model = 1
    sample_chance = 25
    image_save_path = os.path.join("..", "..", "data", "ground_truth")
    take_pictures(video_file, min_detection_confidence, model, sample_chance, image_save_path)


if __name__ == '__main__':
    make_set()
