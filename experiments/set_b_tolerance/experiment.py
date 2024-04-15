import os
import time
from collections import Counter
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


def analyse_film(process_nr, desired_tolerance, total_processes, output_queue):
    video_file = os.path.join("..", "..", "data", "pictures", "Friends.mp4")
    encodings_file = os.path.join("embeddings_set.pickle")
    face_recognizer = FaceRecognizer(video_file, encodings_file, output_path=None,
                                     show_display=False, process_nr=process_nr, total_processes=total_processes)

    rr = face_recognizer.process_video(desired_tolerance=desired_tolerance,
                                       desired_width=450,
                                       desired_model='hog',
                                       upsample_times=2,
                                       sample_probability=1,
                                       save_images=False)
    output_queue.put(rr)
    print("Process {} finished".format(process_nr))


def experiment(nr_of_processes, desired_tolerance):
    if mp.get_start_method() != 'spawn':
        mp.set_start_method('spawn')
    start = time.time()
    number_of_processes = nr_of_processes
    output = mp.Queue()
    jobs = []
    frames = 0
    sampled = 0
    found_names_list_with_frame_number = []
    counted = Counter()

    for i in range(1, nr_of_processes + 1):
        p = mp.Process(target=analyse_film, args=(i, desired_tolerance, nr_of_processes, output))
        jobs.append(p)
        p.start()

    while number_of_processes > 0:
        r = (fr, sa, nms_fr_nr, cn_td) = output.get()  # this waits for an input
        frames = max(fr, frames)
        sampled += sa
        found_names_list_with_frame_number.extend(nms_fr_nr)
        counted += cn_td
        number_of_processes -= 1

    for p in jobs:
        p.join()
    end = time.time()
    duration = end - start
    with open(f'exp_results_{desired_tolerance}.txt', 'w') as fp:
        fp.write('duration: ' + str(duration) + ' seconds\n')
        fp.write('sampled: ' + str(sampled) + '\n')
        fp.write('frames: ' + str(frames) + '\n')
        fp.write(str(counted) + '\n')
        fp.write('\n'.join('%s %s' % x for x in found_names_list_with_frame_number))


def bars(counter, total_frames, desired_tolerance):
    xs = sorted(counter.keys())
    ys = [100 * counter.get(x) / total_frames for x in xs]
    fig, ax1 = plt.subplots(layout='constrained')
    ax1.bar(xs, ys, width=0.95)
    ax1.set_ylim([0, 60])
    ax1.set_xticks(xs)
    ax1.set_xticklabels(xs)
    ax1.set_xlabel(f'actors tolerance {desired_tolerance}')
    ax1.set_ylabel('estimated percentage present in Friends clip')

    plt.show()


def calculate_mse(c1: Counter, c1frames: int, c2: Counter, c2frames: int):
    mse = sum([(100 * (c1.get(x) / c1frames - c2.get(x) / c2frames)) ** 2 for x in c1.keys()]) / len(c1)
    return mse


def create_bars():
    bars(
        Counter(
            {'Rachel': 289, 'Chandler': 331, 'Monica': 639, 'Joey': 292, 'Phoebe': 131, 'Ross': 456, 'Unknown': 689})
        , 2293, 'human intelligence')

    bars(
        Counter(
            {'Unknown': 5220, 'Ross': 1229, 'Monica': 796, 'Chandler': 795, 'Rachel': 720, 'Joey': 619, 'Phoebe': 384})
        , 9308, '0.5')

    bars(
        Counter({'Unknown': 2881, 'Monica': 1592, 'Ross': 1436, 'Rachel': 1242, 'Chandler': 1115, 'Joey': 962,
                 'Phoebe': 535})
        , 9308, '0.58')

    bars(
        Counter({'Unknown': 2376, 'Monica': 1673, 'Rachel': 1484, 'Ross': 1454, 'Chandler': 1174, 'Joey': 1019,
                 'Phoebe': 583})
        , 9308, '0.6')

    bars(
        Counter({'Unknown': 1856, 'Rachel': 1795, 'Monica': 1721, 'Ross': 1430, 'Chandler': 1245, 'Joey': 1039,
                 'Phoebe': 677})
        , 9308, '0.62')

    bars(
        Counter(
            {'Rachel': 3100, 'Chandler': 2135, 'Monica': 2081, 'Joey': 1293, 'Phoebe': 688, 'Ross': 399, 'Unknown': 67})
        , 9308, '0.7')


def create_mse():
    mse_p6 = calculate_mse(Counter(
        {'Rachel': 289, 'Chandler': 331, 'Monica': 639, 'Joey': 292, 'Phoebe': 131, 'Ross': 456, 'Unknown': 689}), 2293,

        Counter({'Unknown': 2376, 'Monica': 1673, 'Rachel': 1484, 'Ross': 1454, 'Chandler': 1174, 'Joey': 1019,
                 'Phoebe': 583})
        , 2339)

    mse_p58 = calculate_mse(Counter(
        {'Rachel': 289, 'Chandler': 331, 'Monica': 639, 'Joey': 292, 'Phoebe': 131, 'Ross': 456, 'Unknown': 689}), 2293,

        Counter({'Unknown': 2881, 'Monica': 1592, 'Ross': 1436, 'Rachel': 1242, 'Chandler': 1115, 'Joey': 962,
                 'Phoebe': 535})
        , 2339)

    print(mse_p6, mse_p58)


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
    plt.subplots(layout='constrained')
    plt.bar(actors, frequencies, color=colors)
    plt.xlabel('Actors')
    plt.ylabel('Frequency %')
    plt.title(f'Frequency of Recognized Actors Over Video Frames\ntaken from Friends tolerance: {tolerance}')
    plt.xticks(rotation=45)
    plt.ylim([0, 5500])  # Set y-axis limit slightly above the maximum frequency
    store_experiment = f"frequency_tolerance_{tolerance}.png"
    plt.savefig(store_experiment)  # save the figure to file

    plt.show()


if __name__ == '__main__':
    #create_bars()
    # create_encodings()
    # experiment(32, 0.50)
    # experiment(32, 0.60)
    # experiment(32, 0.70)
    # experiment(32, 0.62)
    # experiment(32, 0.58)
    # create_mse()
    plot_actor_frequencies("exp_results_0.6.txt",0.6)
    plot_actor_frequencies("exp_results_0.5.txt",0.5)
    plot_actor_frequencies("exp_results_0.7.txt",0.7)
    # plot_actor_frequencies("ground_truth.txt","gt")
