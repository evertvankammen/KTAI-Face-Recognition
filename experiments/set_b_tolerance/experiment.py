from experiments.shared.shared import plot_video_frames, compare_counters


def video_frames_graphs():
    plot_video_frames("exp_results_t_0.5_m_hog_u_0.txt", "0.5 upsample=0 same frames as ground truth", experiment_nr=1)
    plot_video_frames("exp_results_t_0.6_m_hog_u_0.txt", "0.6 upsample=0 same frames as ground truth", experiment_nr=1)
    plot_video_frames("exp_results_t_0.7_m_hog_u_0.txt", "0.7 upsample=0 same frames as ground truth", experiment_nr=1)

    plot_video_frames("exp_results_t_0.5_m_hog_u_1.txt", "0.5 upsample=1 same frames as ground truth", experiment_nr=1)
    plot_video_frames("exp_results_t_0.6_m_hog_u_1.txt", "0.6 upsample=1 same frames as ground truth", experiment_nr=1)
    plot_video_frames("exp_results_t_0.7_m_hog_u_1.txt", "0.7 upsample=1 same frames as ground truth", experiment_nr=1)


def compare_counters_graphs():
    compare_counters("exp_results_t_0.5_m_hog_u_0.txt", "exp_results_manual.txt", "0.5 tolerance upsample=0", experiment_nr=1)
    compare_counters("exp_results_t_0.6_m_hog_u_0.txt", "exp_results_manual.txt", "0.6 tolerance upsample=0", experiment_nr=1)
    compare_counters("exp_results_t_0.7_m_hog_u_0.txt", "exp_results_manual.txt", "0.7 tolerance upsample=0", experiment_nr=1)

    compare_counters("exp_results_t_0.5_m_hog_u_1.txt", "exp_results_manual.txt", "0.5 tolerance upsample=1", experiment_nr=1)
    compare_counters("exp_results_t_0.6_m_hog_u_1.txt", "exp_results_manual.txt", "0.6 tolerance upsample=1", experiment_nr=1)
    compare_counters("exp_results_t_0.7_m_hog_u_1.txt", "exp_results_manual.txt", "0.7 tolerance upsample=1", experiment_nr=1)


if __name__ == '__main__':
    video_frames_graphs()
    compare_counters_graphs()
    # create_encodings()
    # experiment(nr_of_processes=20, up_sampling_factor=0, desired_tolerance=0.50, model='hog')
    # experiment(nr_of_processes=20, up_sampling_factor=0, desired_tolerance=0.60, model='hog')
    # experiment(nr_of_processes=20, up_sampling_factor=0, desired_tolerance=0.70, model='hog')
    # create_mse()
    # plot_actor_frequencies("exp_results_manual.txt", "estimated ground truth", multiplier=4)
    # plot_video_frames("exp_results_manual.txt", "ground truth 25% sample")

