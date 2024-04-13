import os
from matplotlib import pyplot as plt


def print_hist_recognized_faces(
    fc_DataFrame,
    total,
    experiment):
    """Converts normalized value pair to pixel coordinates."""

    graph_path = os.path.join("..", "data", "outcomes")

    width = 16
    height = 10
    plt.figure(figsize=(width, height))
    #fig.set_size_inches(width, height)

    # Setting the fontsize of the axis label to 14
    plt.xlabel('Actors', fontsize=14)
    plt.ylabel('#Video_frames', fontsize=14)

    # displaying the title
    plt.title("Number of faces recognized", loc = "left", fontsize=18, color="Red")
    plt.title('Experiment: '+experiment, loc = "right")

    plt.bar(fc_DataFrame.columns, total, width=0.7)
    store_experiment = os.path.join("..", "data", "outcomes", "total frames experimentnr "+experiment)
    plt.savefig(store_experiment)  # save the figure to file
    plt.show()


def  print_frames_of_recognized_faces(
    fc_DataFrame,
    experiment):
    """Converts normalized value pair to pixel coordinates."""

    graph_path = os.path.join("..", "data", "outcomes")

    x = fc_DataFrame.columns
    y = fc_DataFrame.transpose()

    width = 16
    height = 16
    plt.figure(figsize=(width, height))
    # fig.set_size_inches(width, height)

    # Setting the fontsize of the axis label to 14
    plt.xlabel('Actors', fontsize=14)
    plt.ylabel('#Video_frames', fontsize=14)

    # displaying the title
    plt.title("Video frame numbers in which faces are recognized", loc = "left", fontsize=18, color="Red")
    plt.title('Experiment: '+experiment, loc = "right")

    plt.plot(x, y, color='black', marker='o', linestyle='None', markersize=3)

    store_experiment = os.path.join("..", "data", "outcomes", "nr of frames experimentnr "+experiment)
    plt.savefig(store_experiment)  # save the figure to file
    plt.show()