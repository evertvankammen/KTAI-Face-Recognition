import os
import time
from collections import Counter
import multiprocessing as mp

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
