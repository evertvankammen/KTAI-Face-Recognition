import os

from using_mediapipe.video.video_processor import save_encodings_to_file, run_face_detector, take_pictures


def take_screenshots():
    take_pictures('Friends.mp4')




def experiment_1():
    embeddings_file = "using_external_images.csv"
    image_path_embeddings = os.path.join("..", "..", "encoding_with_pickle", "dataset")
    #save_encodings_to_file(embeddings_file, image_path_embeddings)
    run_face_detector('Friends.mp4', embeddings_file)




