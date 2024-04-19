from experiments.shared.shared import experiment, plot_video_frames, compare_counters


def video_frames_graphs():
    plot_video_frames("exp_results_t_0.6_m_hog_u_0.txt", "0.6 upsample=0 same frames as ground truth", experiment_nr=2)
    plot_video_frames("exp_results_t_0.6_m_hog_u_1.txt", "0.6 upsample=1 same frames as ground truth", experiment_nr=2)
    plot_video_frames("exp_results_t_0.6_m_hog_u_2.txt", "0.6 upsample=2 same frames as ground truth", experiment_nr=2)


def compare_counters_graphs():
    compare_counters("exp_results_t_0.6_m_hog_u_0.txt", "exp_results_manual.txt", "0.6 tolerance upsample=0",
                     experiment_nr=2)
    compare_counters("exp_results_t_0.6_m_hog_u_1.txt", "exp_results_manual.txt", "0.6 tolerance upsample=1",
                     experiment_nr=2)
    compare_counters("exp_results_t_0.6_m_hog_u_2.txt", "exp_results_manual.txt", "0.6 tolerance upsample=2",
                     experiment_nr=3)


if __name__ == '__main__':
    video_frames_graphs()
    compare_counters_graphs()
    # experiment(nr_of_processes=32, up_sampling_factor=0, model='hog')
    # experiment(nr_of_processes=32, up_sampling_factor=1, model='hog')
    # experiment(nr_of_processes=16, up_sampling_factor=2, model='hog')
    #experiment(nr_of_processes=10, up_sampling_factor=5, model='hog')
