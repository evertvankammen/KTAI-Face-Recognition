import os

from using_mediapipe.video.video_processor import take_pictures


def take_screenshots():
    image_save_path = os.path.join("..", "..", "data", "manual_set")
    video_path = os.path.join("..", "..", "data", "movies",'Friends.mp4')
    a, b, c = take_pictures(video_path, min_detection_confidence=0.1, model=1,
                            sample_chance=1, image_save_path=image_save_path)
    print(a, b, c)


take_screenshots()
