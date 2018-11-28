import os
import sys
import matplotlib.pyplot as plt

PLOT_FOLDER = "./plots/"

def save_plotted_progress(optimizer, data=None, name=None, x_lim=None, y_lim=None):
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    data_to_plot = data
    if data_to_plot is  None:
        data_to_plot = optimizer.eval_fn_per_timestep

    _name = name
    if _name is None:
        _name = optimizer.name

    axes = plt.gca()
    if y_lim is not None:
        axes.set_ylim([y_lim[0], y_lim[1]])

    plt.plot(data_to_plot)
    plt.savefig(os.path.join(PLOT_FOLDER, _name))
    plt.clf()
    plt.cla()
    plt.close()
