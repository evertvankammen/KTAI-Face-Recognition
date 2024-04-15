import numpy as np
from matplotlib import pyplot as plt

from encoding_with_pickle.face_recognizer import FaceRecognizer
import os

from encoding_with_pickle.image_encoder import ImageEncoder


def create_encodings():
    image_path = os.path.join("..", "..", "data", "set_from_friends_2")
    encode_model = 'cnn'
    number_of_images = 2

    pickle_output_path = os.path.join("encodings_cnn_set_from_movie_8_per_char.pickle".format(encode_model, number_of_images))
    image_encoder = ImageEncoder(image_path)
    image_encoder.encode_images(encode_model=encode_model, max_images=number_of_images)
    image_encoder.save_encodings(pickle_output_path)


def analyse_film():
    video_file = os.path.join("..", "..", "data", "pictures", "Friends.mp4")
    encodings_file = os.path.join("encodings_cnn_set_from_movie_8_per_char.pickle")
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
    with open('exp_set_from_movie_results_450.txt', 'w') as fp:
        fp.write('sampled: ' + str(sampled) + '\n')
        fp.write(str(counted) + '\n')
        fp.write('\n'.join('%s %s' % x for x in found_names_list_with_frame_number))

# create_encodings()
# analyse_film()
def make_graph():
    auteurs = ['a','b']
    comparatios = [1,2]

    # x-axis: unieke auteurs
    # y-axis: percentage multicore

    xpoints = np.array(auteurs)
    ypoints = np.array(comparatios)

    plt.title("Distibution of multicore programmers by Project")
    plt.xlabel("Authors in Project")
    plt.ylabel("Percentage Multicore Authors in Project")
    plt.xscale('log')

    # scatter: toon datapunten
    #plt.scatter(xpoints,ypoints)
    # bereken trendline

    # toon trendline
    plt.plot(xpoints, ypoints, "r--")

    # toon figuur
    plt.show()

make_graph()