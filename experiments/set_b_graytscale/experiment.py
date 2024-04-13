import numpy as np
from matplotlib import pyplot as plt

from encoding_with_pickle.face_recognizer import FaceRecognizer
import os

from encoding_with_pickle.image_encoder import ImageEncoder


def create_encodings():
    image_path = os.path.join("..", "..", "data", "set_from_friends")
    encode_model = 'cnn'
    number_of_images = 2

    pickle_output_path = os.path.join("encodings_set.pickle".format(encode_model, number_of_images))
    image_encoder = ImageEncoder(image_path)
    image_encoder.encode_images(encode_model=encode_model, max_images=number_of_images)
    image_encoder.save_encodings(pickle_output_path)


def analyse_film():
    video_file = os.path.join("..", "..", "data", "pictures", "Friends.mp4")
    encodings_file = os.path.join("encodings_set.pickle")
    output_path = os.path.join("output_result_videos", "test3.avi")
    face_recognizer = FaceRecognizer(video_file, encodings_file, output_path=output_path, process_every_nth_frame=20,
                                     show_display=True)
    frames, sampled, found_names_list_with_frame_number, counted = face_recognizer.process_video(desired_tolerance=0.53,
                                                                                                 desired_width=450,
                                                                                                 desired_model='hog',
                                                                                                 upsample_times=2,
                                                                                                 sample_probability=0.25,
                                                                                                 save_images=False)

    print(frames, sampled, counted)
    with open('exp_set_from_movie_results.txt', 'w') as fp:
        fp.write('sampled: ' + str(sampled) + '\n')
        fp.write(str(counted) + '\n')
        fp.write('\n'.join('%s %s' % x for x in found_names_list_with_frame_number))


create_encodings()
# analyse_film()