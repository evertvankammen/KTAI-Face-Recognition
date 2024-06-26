import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
import os


def plot_actor_frequencies(file_path, tolerance, upsample):
    """Plot the frequency of recognized actors over video frames."""
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        sampled = int(lines[0].split(':')[1].strip())
        counted = eval(lines[1].strip())
        found_names_list_with_frame_number = [(line.split()[0], int(line.split()[1])) for line in lines[2:]]

    name_counter = Counter(name for name, _ in found_names_list_with_frame_number)

    actors = ['Chandler', 'Joey', 'Monica', 'Phoebe', 'Rachel', 'Ross', 'Unknown']
    frequencies = [name_counter[actor] for actor in actors]
    # Aangepaste kleuren voor elke acteur
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'gray']

    plt.bar(actors, frequencies, color=colors)
    plt.xlabel('Actors')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Recognized Actors Over Video Frames\nInternet picture with {tolerance} tolerance')
    plt.xticks(rotation=45)
    plt.ylim([0, max(frequencies) * 1.1])  # Set y-axis limit slightly above the maximum frequency

    store_experiment = os.path.join("..", "..", "experiments", "ralph",
                                    f"Experiment_2_set_A_frequency_tolerance_{tolerance}"
                                    f"_upsample_{upsample}.png")
    plt.savefig(store_experiment)  # save the figure to file

    plt.show()


def get_needed_frames():
    frames = []
    file_path_temp = os.path.join("..", "..", "experiments", "set_a_tolerance", "exp_results_manual.txt")
    with open(file_path_temp, 'r') as fp:
        lines = fp.readlines()
        for line in lines[4:]:
            actor, frame = line.split()
            frames.append(int(frame))
    return set(frames)


def plot_video_frames(file_path, tolerance, upsample):
    """Plot frames where actors are recognized from a text file."""
    actor_frames = defaultdict(list)

    frms = get_needed_frames()

    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines[2:]:
            actor, frame = line.split()
            if int(frame) in frms:
                actor_frames[actor].append(int(frame))

    s = sorted(actor_frames.items())

    # Voeg "Unknown" toe als sleutel in het geval het niet in de experiment_a_internet_pictures voorkomt
    if "Unknown" not in actor_frames:
        actor_frames["Unknown"] = []

    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'gray']

    for i, (actor, frames) in enumerate(s):
        # Plot elk punt met een specifieke kleur
        plt.plot([actor] * len(frames), frames, marker='o', linestyle='None', label=actor,
                 color=colors[i % len(colors)])

    plt.xlabel('Actors')
    plt.ylabel('Frame Number')
    plt.title(f'Frames where Actors are Recognized\nTolerance: {tolerance}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    store_experiment = os.path.join("..", "..", "experiments", "ralph",
                                    f"Experiment_1_set_A_actor_frames_tolerance_{tolerance}"
                                    f"_upsample_{upsample}.png")
    plt.savefig(store_experiment)  # save the figure to file
    plt.show()


def addlabels(x, y, offset=0.3):
    for i in range(len(x)):
        plt.text(i + offset, y[i], round(y[i]))


def compare_counters(file_path, ground_truth, tolerance, upsample, extra_text):
    """Compare two Counters and plot the comparison."""
    # Combineer de acteurs uit beide Counters
    experiment = extract_counter_from_file(file_path)
    actors = sorted(set(experiment.keys()) | set(ground_truth.keys()))

    # Haal de aantallen frames op voor elke acteur
    frames_counter1 = [experiment.get(actor, 0) for actor in actors]
    frames_counter2 = [ground_truth.get(actor, 0) for actor in actors]

    # Plot de vergelijking
    plt.figure(figsize=(10, 6))
    x = range(len(actors))
    plt.bar(x, frames_counter1, width=0.4, label='Extracted Counter', color='blue', zorder=3)
    addlabels(x, frames_counter1, -0.15)
    x2 = [i + 0.4 for i in x]
    plt.bar(x2, frames_counter2, width=0.4, label='Ground Truth Counter', color='orange', zorder=3)
    addlabels(x2, frames_counter2, +0.25)
    plt.xlabel('Actors')
    plt.ylabel('Number of Frames')
    plt.title(f'Comparison of Extracted Counter {extra_text} vs Ground Truth Counter')
    plt.xticks([i + 0.2 for i in x], actors)
    plt.legend()
    plt.grid(axis='y', zorder=0)
    plt.tight_layout()
    plt.ylim([0, 5500])

    store_experiment = os.path.join("..", "..", "experiments", "ralph",
                                    f"Experiment_2_set_A_compare_with_ground_truth_tolerance_{tolerance}"
                                    f"_upsample_{upsample}.png")
    plt.savefig(store_experiment)  # save the figure to file
    plt.show()


def extract_counter_from_file(file_path):
    """Extract the Counter from a text file."""
    # Initialiseer een lege Counter
    actor_counter = Counter()

    # Lees het tekstbestand en verwerk elke regel na de tweede
    with open(file_path, 'r') as fp:
        next(fp)  # Skip de eerste regel
        next(fp)  # Skip de tweede regel
        for line in fp:
            actor, _ = line.split()
            actor_counter[actor] += 1

    return actor_counter


ground_truth = Counter(
    {'Chandler': 1324, 'Joey': 1168, 'Monica': 2556, 'Phoebe': 524, 'Rachel': 1156, 'Ross': 1824, 'Unknown': 2748})
# Inlezen van de resultaten uit het tekstbestand
tolerance = '70'
upsample = '1'

file_path_exp1 = os.path.join("..", "..", "experiments", "set_a_tolerance",
                         f"exp_set_from_movie_results_tolerance_{tolerance}_upsample_0_internetpictures_desired_width_750.txt")

file_path_exp2 = os.path.join("..", "..", "experiments", "set_a_upsampling",
                         f"exp_set_from_movie_results_tolerance_{tolerance}_upsample_1_internetpictures_desired_width_750.txt")

compare_counters(file_path_exp2, ground_truth, tolerance, upsample, f"0.{tolerance} tolerance")



# plot_video_frames(file_path_exp2, "0.7 same as ground truth", upsample)
# plot_actor_frequencies(file_path, tolerance, upsample)
