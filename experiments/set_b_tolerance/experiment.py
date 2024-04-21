import os

from experiments.shared.shared import plot_video_frames, compare_counters, create_encodings, experiment


def make_encodings():
    create_encodings(os.path.join("..", "..", "data", "b_set_from_friends"),
                     pickle_output_path=os.path.join("embeddings_set.pickle"))


def create_experiment_results():
    experiment(nr_of_processes=32, up_sampling_factor=0, desired_tolerance=0.50, model='hog',
               video_file=os.path.join("..", "..", "data", "pictures", "Friends.mp4"),
               encodings_file="embeddings_set.pickle")
    experiment(nr_of_processes=32, up_sampling_factor=0, desired_tolerance=0.60, model='hog',
               video_file=os.path.join("..", "..", "data", "pictures", "Friends.mp4"),
               encodings_file="embeddings_set.pickle")
    experiment(nr_of_processes=32, up_sampling_factor=0, desired_tolerance=0.70, model='hog',
               video_file=os.path.join("..", "..", "data", "pictures", "Friends.mp4"),
               encodings_file="embeddings_set.pickle")


def video_frames_graphs():
    """

    This method, video_frames_graphs, is used to generate graphs for video frames based on certain parameters.

    Parameters:
    - None

    Returns:
    - None

    Example Usage:
    video_frames_graphs()

    """
    plot_video_frames("exp_results_t_0.5_m_hog_u_0.txt", "exp_results_manual.txt",
                      "0.5 upsample=0 same frames as ground truth", experiment_nr=1)
    plot_video_frames("exp_results_t_0.6_m_hog_u_0.txt", "exp_results_manual.txt",
                      "0.6 upsample=0 same frames as ground truth", experiment_nr=1)
    plot_video_frames("exp_results_t_0.7_m_hog_u_0.txt", "exp_results_manual.txt",
                      "0.7 upsample=0 same frames as ground truth", experiment_nr=1)


def compare_counters_graphs():
    """

    Compare Counters Graphs

    This method compares different counters graphs for different experiments.

    Parameters:
    N/A

    Returns:
    N/A

    Example Usage:
    compare_counters_graphs()

    """
    compare_counters("exp_results_t_0.5_m_hog_u_0.txt", "exp_results_manual.txt", "0.5 tolerance upsample=0",
                     experiment_nr=1)
    compare_counters("exp_results_t_0.6_m_hog_u_0.txt", "exp_results_manual.txt", "0.6 tolerance upsample=0",
                     experiment_nr=1)
    compare_counters("exp_results_t_0.7_m_hog_u_0.txt", "exp_results_manual.txt", "0.7 tolerance upsample=0",
                     experiment_nr=1)


if __name__ == '__main__':
    compare_counters_graphs()
