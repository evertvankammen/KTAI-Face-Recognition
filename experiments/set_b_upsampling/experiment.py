import multiprocessing as mp
import os
import time
from collections import Counter

from matplotlib import pyplot as plt

from encoding_with_pickle.face_recognizer import FaceRecognizer
from encoding_with_pickle.image_encoder import ImageEncoder
from experiments.shared.shared import experiment


def calculate_mse(c1: Counter, c1frames: int, c2: Counter, c2frames: int):
    mse = sum([(100 * (c1.get(x) / c1frames - c2.get(x) / c2frames)) ** 2 for x in c1.keys()]) / len(c1)
    return mse


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], round(y[i]))


def plot_actor_frequencies(file_path, up_sampling):
    """Plot the frequency of recognized actors over video frames."""
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        sampled = int(lines[1].split(':')[1].strip())
        counted = eval(lines[3].strip())
        found_names_list_with_frame_number = [(line.split()[0], int(line.split()[1])) for line in lines[4:]]

    name_counter = Counter(name for name, _ in found_names_list_with_frame_number)

    actors = ['Chandler', 'Joey', 'Monica', 'Phoebe', 'Rachel', 'Ross', 'Unknown']
    frequencies = [name_counter[actor] for actor in actors]
    # Aangepaste kleuren voor elke acteur
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'gray']
    plt.subplots(layout='constrained')
    plt.bar(actors, frequencies, color=colors)
    addlabels(actors, frequencies)
    plt.xlabel('Actors')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Recognized Actors Over Video Frames\ntaken from Friends up sampling: {up_sampling}')
    plt.xticks(rotation=45)
    plt.ylim([0, 5500])  # Set y-axis limit slightly above the maximum frequency
    store_experiment = f"up_sampling_{up_sampling}.png"
    plt.savefig(store_experiment)  # save the figure to file

    plt.show()


if __name__ == '__main__':
    # experiment(nr_of_processes=32, up_sampling_factor=0, model='hog')
    # experiment(nr_of_processes=32, up_sampling_factor=1, model='hog')
    # experiment(nr_of_processes=16, up_sampling_factor=2, model='hog')

    # experiment(nr_of_processes=32, up_sampling_factor=0, model='cnn')
    # experiment(nr_of_processes=32, up_sampling_factor=1, model='cnn')
    # experiment(nr_of_processes=32, up_sampling_factor=2, model='cnn')

    # plot_actor_frequencies("exp_results_t_0.6_m_hog_u_0.txt", 0)
    # plot_actor_frequencies("exp_results_t_0.6_m_hog_u_1.txt", 1)
    plot_actor_frequencies("exp_results_t_0.6_m_hog_u_2.txt", 2)

