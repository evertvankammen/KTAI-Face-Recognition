import os

from using_face_recognition.face_recognizer import FaceRecognizer


def analyse_film(movie_name):
    video_file = os.path.join("..", "..", "data", "movies", movie_name)
    encodings_file = os.path.join("embeddings_set.pickle")
    face_recognizer = FaceRecognizer(video_file, encodings_file, output_path=None,
                                     show_display=False, process_nr=1, total_processes=1)

    face_recognizer.process_video(desired_tolerance=0.6,
                                  desired_model="hog",
                                  upsample_times=1,
                                  sample_probability=0.02,
                                  save_images=True)


def experiment():
    analyse_film("Friends.mp4")
    analyse_film("The Ones With Chandler's Sarcasm _ Friends.mp4")


if __name__ == '__main__':
    # create_bars()
    # create_encodings()
    experiment()
    # experiment(nr_of_processes=32, desired_tolerance=0.60, model='hog')
    # experiment(nr_of_processes=32, desired_tolerance=0.70, model='hog')
