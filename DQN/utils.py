import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
import datetime

results_dir = "DQN Results/"
rundate = str(datetime.datetime.now().strftime("%Y-%m-%d %H'%M''"))

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = results_plotter.ts2xy(results_plotter.load_results(log_folder), 'timesteps')
    y = moving_average(y, window=5)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + "  - 5 Episode Moving Average")
    plt.savefig(results_dir + f"{title} - 5 ep MA {rundate} .png")
    #plt.show()
