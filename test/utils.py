import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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


def eval_fn_with_2_params(fn, x, y):
    return fn({'x': x, 'y':y})

def gen_target_fn_samples(target_fn, param_ranges, n_samples_per_axis=100):
    x_range = np.linspace(param_ranges[0][0], param_ranges[0][1], num=n_samples_per_axis)
    y_range = np.linspace(param_ranges[1][0], param_ranges[1][1], num=n_samples_per_axis)
    sample_points = np.array([[x, y] for x in x_range for y in y_range])
    evals = np.array([[eval_fn_with_2_params(target_fn, x, y) for x in x_range] for y in y_range])
    return sample_points, evals, x_range, y_range


def gen_example_2d_plot(sample_points, target_fn, param_ranges, name=None):
    _name = name
    if name is None:
        _name = "example_2d_plot"
    #param_ranges = [[np.min(sample_points[:, 0]), np.max(sample_points[:, 0])]
    #    , [np.min(sample_points[:, 1]), np.max(sample_points[:, 1])]]
    _, evals, x_range, y_range = gen_target_fn_samples(target_fn, param_ranges)

    x_marginal = np.average(evals, axis=0)
    y_marginal = np.average(evals, axis=1)

    p = sns.JointGrid(
        x=sample_points[:, 0],
        y=sample_points[:, 1],
        xlim=[param_ranges[0][0] - 0.1, param_ranges[0][1] + 0.1],
        ylim=[param_ranges[1][0] - 0.1, param_ranges[1][1] + 0.1]
    )

    p = p.plot_joint(
        plt.scatter
    )


    p.ax_marg_x.fill_between(
        x_range,
        x_marginal,
        alpha=0.5
    )

    p.ax_marg_y.fill_betweenx(
        y_range,
        y_marginal,
        alpha=0.5
    )

    sample_points_x_sorted_idx = np.argsort(sample_points[:, 0])
    sample_points_y_sorted_idx = np.argsort(sample_points[:, 1])

    sample_points_x_sorted = sample_points[:, 0][sample_points_x_sorted_idx]
    sample_points_y_sorted = sample_points[:, 1][sample_points_y_sorted_idx]

    evaluations_x_sorted = [eval_fn_with_2_params(target_fn, x, y) for (x, y) in
                            sample_points[sample_points_x_sorted_idx]]  # evaluations[sample_points_x_sorted_idx]
    evaluations_y_sorted = [eval_fn_with_2_params(target_fn, x, y) for (x, y) in
                            sample_points[sample_points_y_sorted_idx]]  # evaluations[sample_points_y_sorted_idx]

    p.ax_marg_x.scatter(
        sample_points_x_sorted,
        evaluations_x_sorted,
        alpha=0.5,
        color='k'
    )

    print(evaluations_y_sorted)

    p.ax_marg_y.scatter(
        evaluations_y_sorted,
        sample_points_y_sorted,
        # orientation = 'horizontal',
        alpha=0.5,
        color='k')

    plt.savefig(os.path.join(PLOT_FOLDER, _name))
    plt.clf()
    plt.cla()
    plt.close()
