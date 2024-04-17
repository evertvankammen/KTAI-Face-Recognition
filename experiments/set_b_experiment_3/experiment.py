import os

from encoding_with_pickle.face_recognizer import FaceRecognizer


def analyse_film():
    video_file = os.path.join("..", "..", "data", "movies", "Friends.mp4")
    encodings_file = os.path.join("embeddings_set.pickle")
    face_recognizer = FaceRecognizer(video_file, encodings_file, output_path=None,
                                     show_display=False, process_nr=1, total_processes=1)

    face_recognizer.process_video(desired_tolerance=0.6,
                                  desired_model="hog",
                                  upsample_times=1,
                                  sample_probability=0.02,
                                  save_images=True)


def experiment():
    analyse_film()


if __name__ == '__main__':
    # create_bars()
    # create_encodings()
    experiment()
    # experiment(nr_of_processes=32, desired_tolerance=0.60, model='hog')
    # experiment(nr_of_processes=32, desired_tolerance=0.70, model='hog')
