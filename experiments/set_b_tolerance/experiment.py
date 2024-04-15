from collections import Counter

from matplotlib import pyplot as plt

from experiments.shared.shared import experiment


def calculate_mse(c1: Counter, c1frames: int, c2: Counter, c2frames: int):
    mse = sum([(100 * (c1.get(x) / c1frames - c2.get(x) / c2frames)) ** 2 for x in c1.keys()]) / len(c1)
    return mse


def addlabels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], round(y[i]))


def plot_actor_frequencies(file_path, tolerance, multiplier=1.0):
    """Plot the frequency of recognized actors over video frames."""
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        sampled = int(lines[1].split(':')[1].strip())
        counted = eval(lines[3].strip())
        found_names_list_with_frame_number = [(line.split()[0], int(line.split()[1])) for line in lines[4:]]

    name_counter = Counter(name for name, _ in found_names_list_with_frame_number)

    actors = ['Chandler', 'Joey', 'Monica', 'Phoebe', 'Rachel', 'Ross', 'Unknown']
    frequencies = [multiplier * name_counter[actor] for actor in actors]
    # Aangepaste kleuren voor elke acteur
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'gray']
    plt.subplots(layout='constrained')
    plt.bar(actors, frequencies, color=colors)
    addlabels(actors, frequencies)
    plt.xlabel('Actors')
    plt.ylabel('Frequency %')
    plt.title(f'Frequency of Recognized Actors Over Video Frames\ntaken from Friends tolerance: {tolerance}')
    plt.xticks(rotation=45)
    plt.ylim([0, 5500])  # Set y-axis limit slightly above the maximum frequency
    store_experiment = f"frequency_tolerance_{tolerance}.png"
    plt.savefig(store_experiment)  # save the figure to file

    plt.show()


if __name__ == '__main__':
    # create_bars()
    # create_encodings()
    experiment(nr_of_processes=32, desired_tolerance=0.50, model='hog')
    # experiment(nr_of_processes=32, desired_tolerance=0.60, model='hog')
    # experiment(nr_of_processes=32, desired_tolerance=0.70, model='hog')

    # experiment(nr_of_processes=32, desired_tolerance=0.50, model='cnn')
    # experiment(nr_of_processes=32, desired_tolerance=0.60, model='cnn')
    # experiment(nr_of_processes=32, desired_tolerance=0.70, model='cnn')

    # create_mse()
    plot_actor_frequencies("exp_results_t_0.5_m_hog_u_1.txt", 0.5)
    plot_actor_frequencies("exp_results_t_0.6_m_hog_u_1.txt", 0.6)
    plot_actor_frequencies("exp_results_t_0.7_m_hog_u_1.txt", 0.7)

    #plot_actor_frequencies("exp_results_t_0.5_m_cnn_u_1.txt",0.6)
    #plot_actor_frequencies("exp_results_t_0.6_m_cnn_u_1.txt",0.7)

    plot_actor_frequencies("exp_results_manual.txt", "gt", multiplier=4.059)
