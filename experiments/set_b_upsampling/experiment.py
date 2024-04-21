import os

from experiments.shared.shared import experiment, plot_video_frames, compare_counters, create_encodings


def make_encodings():
    """
    Creates encodings for a given dataset.

    :raises: None

    :return: None
    """
    create_encodings(os.path.join("..", "..", "data", "b_set_from_friends"),
                     pickle_output_path=os.path.join("embeddings_set.pickle"))


def create_experiment_results():
    experiment(nr_of_processes=32, up_sampling_factor=0, desired_tolerance=0.60, model='hog',
               video_file=os.path.join("..", "..", "data", "pictures", "Friends.mp4"),
               encodings_file="embeddings_set.pickle")
    experiment(nr_of_processes=32, up_sampling_factor=1, desired_tolerance=0.60, model='hog',
               video_file=os.path.join("..", "..", "data", "pictures", "Friends.mp4"),
               encodings_file="embeddings_set.pickle")
    experiment(nr_of_processes=32, up_sampling_factor=2, desired_tolerance=0.60, model='hog',
               video_file=os.path.join("..", "..", "data", "pictures", "Friends.mp4"),
               encodings_file="embeddings_set.pickle")


def video_frames_graphs():
    plot_video_frames("exp_results_t_0.6_m_hog_u_0.txt","exp_results_manual.txt",
                      "0.6 upsample=0 same frames as ground truth", experiment_nr=2)
    plot_video_frames("exp_results_t_0.6_m_hog_u_1.txt","exp_results_manual.txt",
                      "0.6 upsample=1 same frames as ground truth", experiment_nr=2)
    plot_video_frames("exp_results_t_0.6_m_hog_u_2.txt","exp_results_manual.txt",
                      "0.6 upsample=2 same frames as ground truth", experiment_nr=2)


def compare_counters_graphs():
    compare_counters("exp_results_t_0.6_m_hog_u_0.txt", "exp_results_manual.txt", "0.6 tolerance upsample=0",
                     experiment_nr=2)
    compare_counters("exp_results_t_0.6_m_hog_u_1.txt", "exp_results_manual.txt", "0.6 tolerance upsample=1",
                     experiment_nr=2)
    compare_counters("exp_results_t_0.6_m_hog_u_2.txt", "exp_results_manual.txt", "0.6 tolerance upsample=2",
                     experiment_nr=2)


if __name__ == '__main__':
    compare_counters_graphs()
