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


def addlabels(x, y, offset=0):
    for i in range(len(x)):
        plt.text(i + offset, y[i], round(y[i]))


def extract_counter_from_file(file_path):
    """Extract the Counter from a text file."""
    # Initialiseer een lege Counter
    actors = []
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines[4:]:
            actor, frame = line.split()
            actors.append(actor)

    return Counter(actors)


def compare_counters(file_path, ground_truth_path, extra_text):
    """Compare two Counters and plot the comparison."""
    # Combineer de acteurs uit beide Counters
    experiment = extract_counter_from_file(file_path)
    ground_truth = extract_counter_from_file(ground_truth_path)

    actors = sorted(set(experiment.keys()) | set(ground_truth.keys()))

    # Haal de aantallen frames op voor elke acteur
    frames_counter1 = [experiment.get(actor, 0) for actor in actors]
    frames_counter2 = [4 * ground_truth.get(actor, 0) for actor in actors]

    # Plot de vergelijking
    plt.figure(figsize=(10, 6))
    x = range(len(actors))
    plt.bar(x, frames_counter1, width=0.4, label='Extracted Counter', color='blue',zorder=3)
    addlabels(x, frames_counter1, -0.15)
    x2 = [i + 0.4 for i in x]
    plt.bar(x2, frames_counter2, width=0.4, label='Ground Truth Counter', color='orange',zorder=3)
    addlabels(x2, frames_counter2, +0.25)
    plt.xlabel('Actors')
    plt.ylabel('Number of Frames')
    plt.ylim([0, 5500])
    plt.title(f'Comparison of Extracted Counter {extra_text} vs Ground Truth Counter')
    plt.xticks([i + 0.2 for i in x], actors)
    plt.legend()
    plt.grid(axis='y',zorder=0)
    plt.tight_layout()

    store_experiment = f"_upsample_{extra_text}.png"
    plt.savefig(store_experiment)  # save the figure to file
    plt.show()



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
    #plot_actor_frequencies("exp_results_t_0.6_m_hog_u_2.txt", 2)

    compare_counters("exp_results_t_0.6_m_hog_u_0.txt", "exp_results_manual.txt", "0.6 tolerance no-up-sample")
    compare_counters("exp_results_t_0.6_m_hog_u_1.txt", "exp_results_manual.txt", "0.6 tolerance up-sample = 1")
    compare_counters("exp_results_t_0.6_m_hog_u_2.txt", "exp_results_manual.txt", "0.6 tolerance up-sample = 2")


