import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
import os

def plot_actor_frequencies(file_path, tolerance):
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
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'gray']

    plt.bar(actors, frequencies, color=colors)
    plt.xlabel('Actors')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Recognized Actors Over Video Frames\nInternet picture with {tolerance} tolerance')
    plt.xticks(rotation=45)
    plt.ylim([0, max(frequencies) * 1.1])  # Set y-axis limit slightly above the maximum frequency

    store_experiment = os.path.join("..", "..", "experiments", "ralph", f"frequency_actors_recognized_tolerance_{tolerance}_sample_probability_0.25.png")
    plt.savefig(store_experiment)  # save the figure to file

    plt.show()

def plot_video_frames(file_path, tolerance):
    """Plot frames where actors are recognized from a text file."""
    actor_frames = defaultdict(list)

    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines[2:]:
            actor, frame = line.split()
            actor_frames[actor].append(int(frame))

    # Voeg "Unknown" toe als sleutel in het geval het niet in de dataset voorkomt
    if "Unknown" not in actor_frames:
        actor_frames["Unknown"] = []

    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'gray']

    for i, (actor, frames) in enumerate(actor_frames.items()):
        # Plot elk punt met een specifieke kleur
        plt.plot([actor] * len(frames), frames, marker='o', linestyle='None', label=actor,
                 color=colors[i % len(colors)])


    plt.xlabel('Actors')
    plt.ylabel('Frame Number')
    plt.title('Frames where Actors are Recognized\nTolerance: ' + tolerance)
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    store_experiment = os.path.join("..", "..", "experiments", "ralph", f"actor_frames_tolerance_{tolerance}_sample_probability_0.25.png")
    plt.savefig(store_experiment)  # save the figure to file
    plt.show()

def compare_counters(file_path, ground_truth):
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
    plt.bar(x, frames_counter1, width=0.4, label='Extracted Counter', color='blue')
    plt.bar([i + 0.4 for i in x], frames_counter2, width=0.4, label='Ground Truth Counter', color='orange')
    plt.xlabel('Actors')
    plt.ylabel('Number of Frames')
    plt.title('Comparison of Extracted Counter vs Ground Truth Counter')
    plt.xticks([i + 0.2 for i in x], actors)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()

    store_experiment = os.path.join("..", "..", "experiments", "ralph",
                                    f"compare_with_ground_truth_tolerance_{tolerance}_sample_probability_0.25.png")
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

ground_truth = Counter({'Chandler': 179, 'Joey': 292, 'Monica': 639, 'Phoebe': 131, 'Rachel': 289, 'Ross': 456, 'Unknown': 689})
# Inlezen van de resultaten uit het tekstbestand
tolerance = '50'
file_path = f"../exp_set_from_movie_results_tolerance_{tolerance}_internetpictures_sample_probability_0.25.txt"

compare_counters(file_path, ground_truth)
plot_video_frames(file_path, tolerance)
plot_actor_frequencies(file_path, tolerance)