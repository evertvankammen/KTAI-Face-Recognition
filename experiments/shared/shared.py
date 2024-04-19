import os
import time
from collections import Counter, defaultdict
import multiprocessing as mp

from matplotlib import pyplot as plt

from encoding_with_pickle.face_recognizer import FaceRecognizer
from encoding_with_pickle.image_encoder import ImageEncoder


def create_encodings():
    image_path = os.path.join("..", "..", "data", "b_set_from_friends")
    encode_model = 'cnn'
    number_of_images = 2

    pickle_output_path = os.path.join("embeddings_set.pickle".format(encode_model, number_of_images))
    image_encoder = ImageEncoder(image_path)
    image_encoder.encode_images(encode_model=encode_model, max_images=number_of_images)
    image_encoder.save_encodings(pickle_output_path)


def analyse_film(process_nr, desired_tolerance, up_sampling_factor, nr_of_processes, output_queue, model):
    video_file = os.path.join("..", "..", "data", "pictures", "Friends.mp4")
    encodings_file = os.path.join("embeddings_set.pickle")
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


def addlabels(x, y, offset=0.3):
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
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Recognized Actors Over Video Frames\ntaken from Friends tolerance: {tolerance}')
    plt.xticks(rotation=45)
    plt.ylim([0, 5500])  # Set y-axis limit slightly above the maximum frequency
    store_experiment = f"frequency_tolerance_{tolerance}.png"
    plt.savefig(store_experiment)  # save the figure to file

    plt.show()


def compare_counters(file_path, ground_truth_path, text, experiment_nr=1):
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
    plt.bar(x, frames_counter1, width=0.4, label='Extracted Counter', color='blue', zorder=3)
    addlabels(x, frames_counter1, -0.15)
    x2 = [i + 0.4 for i in x]
    plt.bar(x2, frames_counter2, width=0.4, label='Ground Truth Counter', color='orange', zorder=3)
    addlabels(x2, frames_counter2, +0.25)
    plt.xlabel('Actors')
    plt.ylabel('Number of Frames')
    plt.ylim([0, 5500])
    plt.title(f'Comparison of Extracted Counter {text} vs Ground Truth Counter')
    plt.xticks([i + 0.2 for i in x], actors)
    plt.legend()
    plt.grid(axis='y', zorder=0)
    plt.tight_layout()

    store_experiment = f"Experiment_{experiment_nr}_set_B_frequency_tolerance_{text}.png"
    plt.savefig(store_experiment)  # save the figure to file
    plt.show()


def get_needed_frames():
    frames = []
    with open("exp_results_manual.txt", 'r') as fp:
        lines = fp.readlines()
        for line in lines[4:]:
            actor, frame = line.split()
            frames.append(int(frame))
    return set(frames)


def experiment(nr_of_processes=1, desired_tolerance=0.60, up_sampling_factor=1, model='hog'):
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
                       args=(i, desired_tolerance, up_sampling_factor, nr_of_processes, output_queue, model))
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
    with open(f'exp_results_t_{desired_tolerance}_m_{model}_u_{up_sampling_factor}.txt', 'w') as fp:
        fp.write('duration: ' + str(duration) + ' seconds\n')
        fp.write('sampled: ' + str(sampled) + '\n')
        fp.write('frames: ' + str(frames) + '\n')
        fp.write(str(counted) + '\n')
        fp.write('\n'.join('%s %s' % x for x in found_names_list_with_frame_number))


def get_needed_frames():
    frames = []
    with open("exp_results_manual.txt", 'r') as fp:
        lines = fp.readlines()
        for line in lines[4:]:
            actor, frame = line.split()
            frames.append(int(frame))
    return set(frames)


def plot_video_frames(file_path, text, experiment_nr=1):
    """Plot frames where actors are recognized from a text file."""
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

    store_experiment = f"Experiment_{experiment_nr}_set_B_actor_frames_tolerance_{text}.png"
    plt.savefig(store_experiment)  # save the figure to file
    plt.show()
