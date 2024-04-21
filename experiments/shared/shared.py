import os
import time
from collections import Counter, defaultdict
import multiprocessing as mp

from matplotlib import pyplot as plt

from encoding_with_pickle.face_recognizer import FaceRecognizer
from encoding_with_pickle.image_encoder import ImageEncoder


def create_encodings(path, pickle_output_path):
    """
    Executes the process of encoding images using the specified encode_model and number_of_images.

    Parameters:
    None

    Returns:
    None
    """

    encode_model = 'cnn'
    number_of_images = 2
    image_encoder = ImageEncoder(path)
    image_encoder.encode_images(encode_model=encode_model, max_images=number_of_images)
    image_encoder.save_encodings(pickle_output_path)


def analyse_film(process_nr, desired_tolerance, up_sampling_factor, nr_of_processes, output_queue, model,
                 video_file, encodings_file):

    face_recognizer = FaceRecognizer(video_file, encodings_file, output_path=None,
                                     show_display=False, process_nr=process_nr, total_processes=nr_of_processes)

    rr = face_recognizer.process_video(desired_tolerance=desired_tolerance,
                                       desired_model=model,
                                       upsample_times=up_sampling_factor,
                                       sample_probability=1,
                                       save_images=False)
    if output_queue is not None:
        output_queue.put(rr)
    print("Process {} finished".format(process_nr))


def add_labels(x, y, offset=0.3):
    """
    Add labels to a plot at specified positions.

    Parameters:
    x (list): The x-coordinates of the points where the labels should be placed.
    y (list): The y-coordinates of the points where the labels should be placed.
    offset (float, optional): The offset by which the x-coordinate of each label should be adjusted. Default is 0.3.

    Returns:
    None

    Example:
    add_labels([1, 2, 3], [10, 20, 30])
    """
    for i in range(len(x)):
        plt.text(i + offset, y[i], round(y[i]))


def extract_counter_from_file(file_path, start_point_file=4):
    """
    Extracts the counter of actors from a file.

    Args:
        file_path (str): The path of the file to extract the counter from.
        start_point_file (int, optional): The line number to start extracting the counter from. Default is 4.

    Returns:
        collections.Counter: The counter of actors extracted from the file.
    """
    actors = []
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines[start_point_file:]:
            actor, frame = line.split()
            actors.append(actor)

    return Counter(actors)


def plot_actor_frequencies(file_path, tolerance, multiplier=1.0):
    """
    Plot Actor Frequencies

    This method takes in a file path, tolerance value, and an optional multiplier. It reads the contents of the file, extracts the relevant data, and plots the frequencies of recognized actors.

    Parameters:
    - file_path (str): The path to the input file.
    - tolerance (float): The tolerance value for recognizing actors.
    - multiplier (float, optional): A multiplier to adjust the frequencies (default: 1.0).

    Returns:
    None

    Example Usage:
    plot_actor_frequencies('data.txt', 0.5, multiplier=2.0)
    """
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
    add_labels(actors, frequencies)
    plt.xlabel('Actors')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Recognized Actors Over Video Frames\ntaken from Friends tolerance: {tolerance}')
    plt.xticks(rotation=45)
    plt.ylim([0, 5500])  # Set y-axis limit slightly above the maximum frequency
    store_experiment = f"frequency_tolerance_{tolerance}.png"
    plt.savefig(store_experiment)  # save the figure to file

    plt.show()


def compare_counters(file_path, ground_truth_path, text, experiment_nr=1, start_point_file=4):
    """
    Compare the counters from two files and plot the results.

    Parameters:
        file_path (str): File path of the experiment counter.
        ground_truth_path (str): File path of the ground truth counter.
        text (str): Text to include in the title of the plot.
        experiment_nr (int, optional): Number of the experiment. Default is 1.
        start_point_file (int, optional): Starting point in the file. Default is 4.

    Returns:
        None
    """
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
    plt.bar(x, frames_counter1, width=0.4, label='Extracted Counter', color='blue', zorder=3)
    add_labels(x, frames_counter1, -0.15)
    x2 = [i + 0.4 for i in x]
    plt.bar(x2, frames_counter2, width=0.4, label='Ground Truth Counter', color='orange', zorder=3)
    add_labels(x2, frames_counter2, +0.25)
    plt.xlabel('Actors')
    plt.ylabel('Number of Frames')
    plt.ylim([0, 5500])
    plt.title(f'Comparison of Extracted Counter {text} vs Ground Truth Counter')
    plt.xticks([i + 0.2 for i in x], actors)
    plt.legend()
    plt.grid(axis='y', zorder=0)
    plt.tight_layout()

    store_experiment = f"Experiment_set_A_{experiment_nr}_set_B_frequency_tolerance_{text}.png"
    plt.savefig(store_experiment)  # save the figure to file
    plt.show()


def get_needed_frames(start_point_file=4):
    """
    Get a set of frames needed for processing.

    Parameters:
    - start_point_file (int): The line number to start reading from in the file. Default is 4.

    Returns:
    - set: A set of frame numbers needed for processing.

    Example:
    >>> get_needed_frames(4)
    {10, 11, 15, 20}
    """
    frames = []
    with open("exp_results_manual.txt", 'r') as fp:
        lines = fp.readlines()
        for line in lines[start_point_file:]:
            actor, frame = line.split()
            frames.append(int(frame))
    return set(frames)


def experiment(nr_of_processes=1, desired_tolerance=0.60, up_sampling_factor=1, model='hog', video_file="",
               encodings_file=""):

    if mp.get_start_method() != 'spawn':
        mp.set_start_method('spawn')
    start = time.time()
    number_of_processes = nr_of_processes
    output_queue = mp.Queue()
    jobs = []
    frames = 0
    sampled = 0
    found_names_list_with_frame_number = []
    counted = Counter()

    for i in range(1, nr_of_processes + 1):
        p = mp.Process(target=analyse_film,
                       args=(i, desired_tolerance, up_sampling_factor, nr_of_processes, output_queue, model,
                             video_file, encodings_file))
        jobs.append(p)
        p.start()

    while number_of_processes > 0:
        r = (fr, sa, nms_fr_nr, cn_td) = output_queue.get()  # this waits for an input
        frames = max(fr, frames)
        sampled += sa
        found_names_list_with_frame_number.extend(nms_fr_nr)
        counted += cn_td
        number_of_processes -= 1

    for p in jobs:
        p.join()
    end = time.time()
    duration = end - start
    with open(f'exp_results_A_t_{desired_tolerance}_m_{model}_u_{up_sampling_factor}.txt', 'w') as fp:
        fp.write('duration: ' + str(duration) + ' seconds\n')
        fp.write('sampled: ' + str(sampled) + '\n')
        fp.write('frames: ' + str(frames) + '\n')
        fp.write(str(counted) + '\n')
        fp.write('\n'.join('%s %s' % x for x in found_names_list_with_frame_number))


def plot_video_frames(file_path, text, experiment_nr=1):
    """
    Plot video frames

    Plots the frames where actors are recognized in a video.

    Parameters:
    - file_path (str): The path to the file containing actor and frame data.
    - text (str): The tolerance value for actor recognition.
    - experiment_nr (int, optional): The experiment number. Default is 1.

    Returns:
    None

    Example Usage:
    plot_video_frames("path/to/file.txt", "0.5")
    """
    actor_frames = defaultdict(list)

    frms = get_needed_frames()

    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines[4:]:
            actor, frame = line.split()
            if int(frame) in frms:
                actor_frames[actor].append(int(frame))

    s = sorted(actor_frames.items())

    # Voeg "Unknown" toe als sleutel in het geval het niet in de dataset voorkomt
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
    plt.title(f'Frames where Actors are Recognized\nTolerance: {text}')
    plt.xticks(rotation=45)
    plt.grid(True)

    store_experiment = f"Experiment_setA_{experiment_nr}_set_B_actor_frames_tolerance_{text}.png"
    plt.savefig(store_experiment)  # save the figure to file
    plt.show()
