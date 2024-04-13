import os
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


def analyse_film(process_nr, desired_tolerance, total_processes, output_queue):
    video_file = os.path.join("..", "..", "data", "pictures", "Friends.mp4")
    encodings_file = os.path.join("embeddings_set.pickle")
    face_recognizer = FaceRecognizer(video_file, encodings_file, output_path=None,
                                     show_display=False, process_nr=process_nr, total_processes=total_processes)

    rr = face_recognizer.process_video(desired_tolerance=desired_tolerance,
                                       desired_width=450,
                                       desired_model='hog',
                                       upsample_times=2,
                                       sample_probability=0.25,
                                       save_images=False)
    output_queue.put(rr)
    print("Process {} finished".format(process_nr))


def experiment(nr_of_processes, desired_tolerance):
    mp.set_start_method('spawn')
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

    with open(f'exp_results_{desired_tolerance}.txt', 'w') as fp:
        fp.write('sampled: ' + str(sampled) + '\n')
        fp.write('frames: ' + str(frames) + '\n')
        fp.write(str(counted) + '\n')
        fp.write('\n'.join('%s %s' % x for x in found_names_list_with_frame_number))


if __name__ == '__main__':
    # create_encodings()
    # experiment(16, 0.50)
    # experiment(32, 0.60)
    # experiment(32, 0.70)
