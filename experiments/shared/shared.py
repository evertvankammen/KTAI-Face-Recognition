import multiprocessing as mp
import time
from collections import Counter, defaultdict

from matplotlib import pyplot as plt

from using_face_recognition.face_recognizer import FaceRecognizer
from using_face_recognition.image_encoder import ImageEncoder


def create_encodings(path, pickle_output_path):
    """
    Create encodings for given images and save them into a pickle file.

    Parameters:
    - path (str): The path to the directory containing the images.
    - pickle_output_path (str): The path to save the encodings as a pickle file.

    Example usage:
    >>> create_encodings('/path/to/images', '/path/to/output.pickle')
    """
    image_encoder = ImageEncoder(path)
    image_encoder.encode_images(encode_model='cnn', max_images=2)
    image_encoder.save_encodings(pickle_output_path)


def analyse_film(process_nr, desired_tolerance, up_sampling_factor, nr_of_processes, output_queue, model,
                 video_file, encodings_file):
    """
    Analyse Film Method

    This method is used to analyse a film by applying face recognition on each frame of the film.

    Parameters:
    - process_nr (int): The process number of the current process. Used for tracking progress in multiprocess scenarios.
    - desired_tolerance (float): The desired face recognition tolerance. A higher value means more lenient face recognition.
    - up_sampling_factor (int): The up-sampling factor for face detection. A higher value means higher quality detection but slower processing.
    - nr_of_processes (int): The total number of processes running in parallel. Used for tracking progress in multiprocess scenarios.
    - output_queue (Queue): The output queue for storing the result of the analysis. If None, the result will not be stored.
    - model (str): The face recognition model to use.
    - video_file (str): The path to the video file to analyse.
    - encodings_file (str): The path to the face encodings file.

    Returns:
    - None

    Example Usage:
    ```python
    output_queue = Queue()
    model = "cnn"
    video_file = "path/to/video.mp4"
    encodings_file = "path/to/encodings.pickle"

    analyse_film(process_nr=1, desired_tolerance=0.6, up_sampling_factor=2, nr_of_processes=4,
                 output_queue=output_queue, model=model,
                 video_file=video_file, encodings_file=encodings_file)
    ```
    """
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
        x (list): The x-coordinates of the positions where labels will be added.
        y (list): The y-coordinates of the positions where labels will be added.
        offset (float, optional): The offset value to position the labels. Default is 0.3.

    Returns:
        None

    Example:
        x = [0, 1, 2, 3, 4]
        y = [10, 20, 30, 40, 50]
        add_labels(x, y, offset=0.3)
    """
    for i in range(len(x)):
        plt.text(i + offset, y[i], round(y[i]))


def extract_counter_from_file(file_path, start_point_file=4):
    """
    Extracts a counter from a given file.

    Parameters:
    - file_path (str): The path to the file to extract the counter from.
    - start_point_file (int, optional): The line index to start extracting from. Default value is 4.

    Returns:
    - Counter: A counter object representing the frequency count of actors found in the file.

    Example Usage:
        file_path = 'path/to/file.txt'
        start_point = 5
        counter = extract_counter_from_file(file_path, start_point)
    """
    actors = []
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines[start_point_file:]:
            actor, frame = line.split()
            actors.append(actor)

    return Counter(actors)


def get_frames_set(file_path, start_point_file=4):
    frames = []
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines[start_point_file:]:
            actor, frame = line.split()
            frames.append(int(frame))
    return set(frames)


def plot_actor_frequencies(file_path, tolerance, multiplier=1.0):
    """
    Plot the frequencies of recognized actors over video frames.

    :param file_path: The path to the file containing the data.
    :param tolerance: The tolerance value used for recognizing actors.
    :param multiplier: The multiplier to adjust the frequencies (default is 1.0), this is used enable comparing different sample sizes.

    :returns: None
    """
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
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


def compare_counters(file_path, ground_truth_path, text, experiment_nr=1):
    """
    Compare the counters for extracted frames and ground truth frames.

    Parameters:
    - file_path (str): The path to the file containing the extracted frames counter.
    - ground_truth_path (str): The path to the file containing the ground truth frames counter.
    - text (str): Additional text to be included in the plot title.
    - experiment_nr (int): The experiment number. Default is 1.

    Returns:
    None
    """
    # Combineer de acteurs uit beide Counters
    experiment_counter = extract_counter_from_file(file_path)
    ground_truth = extract_counter_from_file(ground_truth_path)

    actors = sorted(set(experiment_counter.keys()) | set(ground_truth.keys()))

    # Haal de aantallen frames op voor elke acteur
    frames_counter1 = [experiment_counter.get(actor, 0) for actor in actors]
    frames_counter2 = [4 * ground_truth.get(actor, 0) for actor in actors]

    # Plot de vergelijking
    plt.figure(figsize=(10, 6))
    x = list(range(len(actors)))
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


def experiment(nr_of_processes=1, desired_tolerance=0.60, up_sampling_factor=1, model='hog', video_file="",
               encodings_file=""):
    """
    This method performs an experiment by running multiple processes to analyze a video file for object detection using facial recognition. The method takes several parameters:

    - nr_of_processes: Optional. Specifies the number of processes to be used for analysis. Default value is 1.
    - desired_tolerance: Optional. Specifies the desired tolerance level for facial recognition. Default value is 0.60.
    - up_sampling_factor: Optional. Specifies the up-sampling factor for the analysis. Default value is 1.
    - model: Optional. Specifies the facial recognition model to be used. Default value is 'hog'.
    - video_file: Optional. Specifies the path of the video file to be analyzed. Default value is an empty string.
    - encodings_file: Optional. Specifies the path of the file containing pre-computed facial encodings. Default value is an empty string.

    The method sets the multiprocessing start method to 'spawn' if it is not already set to 'spawn'.
    It then initializes variables to track the number of processes, an output queue for communication between processes, and lists for storing analysis results.

    Next, the method starts the specified number of processes, each running the 'analyse_film' function with the given parameters. The processes are stored in a list for later joining.

    The method waits for inputs from the output queue and updates the result variables accordingly. Once all processes have finished, the method joins the processes and calculates the duration of the experiment.

    Finally, the method writes the experiment results to a text file, including the duration, number of frames analyzed, number of samples taken, and the count of detected objects. It also writes the found object names with their respective frame numbers.

    Parameters:
    - nr_of_processes: int
    - desired_tolerance: float
    - up_sampling_factor: int
    - model: str
    - video_file: str
    - encodings_file: str

    Returns:
    None

    Example usage:
    experiment(nr_of_processes=2, desired_tolerance=0.65, up_sampling_factor=2, model='cnn', video_file="video.mp4")

    """
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


def plot_video_frames(file_path, ground_truth_file, text, experiment_nr=1):
    """
    Plot video frames with recognized actors.

    Parameters:
    - file_path: str, path to the file containing the actors and frames
    - ground_truth_file: str, path to the ground truth file
    - text: str, tolerance value for the plot title
    - experiment_nr: int, optional, experiment number (default=1)

    Returns:
    None
    """
    actor_frames = defaultdict(list)

    frames_set = get_frames_set(ground_truth_file)

    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines[4:]:
            actor, frame = line.split()
            if int(frame) in frames_set:
                actor_frames[actor].append(int(frame))

    s = sorted(actor_frames.items())

    if "Unknown" not in actor_frames:
        actor_frames["Unknown"] = []

    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'gray']

    for i, (actor, frames) in enumerate(s):
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
