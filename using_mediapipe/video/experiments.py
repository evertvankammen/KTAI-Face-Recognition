import os

from using_mediapipe.video.video_processor import run_face_detector, take_pictures, save_encodings_to_file


def take_screenshots():
    a, b, c = take_pictures('Friends.mp4', min_detection_confidence=0.1, model=1, sample_chance=25)
    print(a, b, c)


def experiment_1():
    embeddings_file = "using_internal_images.csv"
    image_path_embeddings = os.path.join("..", "..", "data", "embeddings")
    # save_encodings_to_file(embeddings_file, image_path_embeddings)
    run_face_detector('Friends.mp4', embeddings_file, 1)


def create_trainings_set_model_1():
    embeddings_file = "verify.csv"
    image_path_embeddings = os.path.join("..", "..", "data", "embeddings")
    save_encodings_to_file(embeddings_file, image_path_embeddings, 1)


def create_trainings_set_model_2():
    embeddings_file = "verify_2.csv"
    image_path_embeddings = os.path.join("..", "..", "data", "embeddings")
    save_encodings_to_file(embeddings_file, image_path_embeddings, 0)


def analyse_trainings_set_model_2():
    embeddings_file = "verify_2.csv"
    run_face_detector('Friends.mp4', embeddings_file, 0.01, 1)


take_screenshots()
