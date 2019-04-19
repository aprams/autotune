import os
import sys
import config
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import collections

PLOT_FOLDER = config.PLOT_FOLDER


def get_display_names_for_optimizers(opt_names):
    return [get_display_name_for_optimizer(name) for name in opt_names]


def get_display_name_for_optimizer(opt_name):
    corrected_opt_name = None
    for k in list(config._OPTIMIZER_DISPLAY_NAMES.keys()):
        if k.lower() == opt_name.lower():
            corrected_opt_name = config._OPTIMIZER_DISPLAY_NAMES[k]
            break
    if corrected_opt_name is None:
        raise Exception("Unknown Optimizer name")

    return corrected_opt_name


def get_color_for_optimizer(opt_name):
    corrected_opt_name = None
    for k in list(config._OPTIMIZER_TO_COLOR_DICT.keys()):
        if k.lower() == opt_name.lower():
            corrected_opt_name = k
            break
    if corrected_opt_name is None:
        raise Exception("Unknown Optimizer name")

    return config._OPTIMIZER_TO_COLOR_DICT[corrected_opt_name]



def prune_invalid_params_for_classifier(params, classifier):
    pruned_params = {}
    valid_string_starts = [classifier, 'preprocessing', 'pca']
    for k in params:
        is_valid_key_for_classifier = any([str.startswith(k, x) for x in valid_string_starts])
        if is_valid_key_for_classifier:
            pruned_params[k] = params[k]

    if pruned_params['preprocessing'] == 0 and 'pca:keep_variance' in pruned_params:
        del pruned_params['pca:keep_variance']
    if 'classifier' in pruned_params:
        del pruned_params['classifier']
    return pruned_params


def flatten(structure, key="", flattened=None):
    # https://stackoverflow.com/questions/8477550/
    # flattening-a-list-of-dicts-of-lists-of-dicts-etc-of-unknown-depth-in-python-n
    if flattened is None:
        flattened = {}
    if type(structure) not in(dict, list):
        flattened[key] = structure
    elif isinstance(structure, list):
        for i, item in enumerate(structure):
            flatten(item, "%d" % i, flattened)
    else:
        for new_key, value in structure.items():
            flatten(value, new_key, flattened)
    return flattened


def flatten_list(li):
    """Flatten lists or tuples into their individual items. If those items are
    again lists or tuples, flatten those."""
    if isinstance(li, (list, tuple)):
        for item in li:
            yield from flatten_list(item)
    else:
        yield li


def avg_rank_plot_per_timestep(eval_fns_per_timestep, save_path, legend_loc='best'):
    avg_rank_optimizer_dict = {}
    stacked_eval_fns_per_timstep = np.stack(list(eval_fns_per_timestep.values()), axis=0)
    avg_rank = np.mean(np.argsort(stacked_eval_fns_per_timstep, axis=0), axis=1) + 1
    for i in range(len(eval_fns_per_timestep.keys())):
        k = list(eval_fns_per_timestep.keys())[i]
        avg_rank_optimizer_dict[k] = avg_rank[i]
        color = get_color_for_optimizer(k)
        plt.plot(avg_rank_optimizer_dict[k], label=get_display_name_for_optimizer(k), color=color)
    plt.legend(loc=legend_loc)
    plt.ylabel('Average rank per timestep')
    plt.xlabel('n_iterations')
    plt.savefig(save_path)

def cpu_time_plot_per_optimizer(results, save_path, y_scale='log'):
    # CPU time plotting:
    cpu_times = results_to_numpy(results, 2)
    cpu_times_avg = {x: np.mean(np.fabs(cpu_times[x])) for x in cpu_times}
    plt.figure(figsize=(10, 5))
    colors = [get_color_for_optimizer(optimizer) for optimizer in list(cpu_times_avg.keys())]
    plt.bar(range(len(cpu_times_avg)), cpu_times_avg.values(),
            tick_label=get_display_names_for_optimizers(list(cpu_times_avg.keys())), color=colors)
    plt.yscale(y_scale)
    _y_label = 'CPU time in seconds'
    if y_scale == 'log':
        _y_label += " (log)"
    plt.ylabel(_y_label)
    plt.xlabel('Optimizer')
    plt.savefig(save_path)


def plot_results_multiple(np_results, dataset_idx=0, avg_datasets=False, t_0=0, plot_ranges=True,
                          save_file_name_prefix=None):
    for x_log in [True, False]:
        for y_log in [True, False]:
            file_name = save_file_name_prefix
            if x_log or y_log:
                file_name += "_log"
                if x_log:
                    file_name += "_x"
                if y_log:
                    file_name += "_y"
            plot_results(np_results, dataset_idx=dataset_idx, avg_datasets=avg_datasets, t_0=t_0,
                         plot_ranges=plot_ranges, save_file_name=file_name, use_log_scale_x=x_log,
                         use_log_scale_y=y_log)


def plot_results(np_results, X=None, dataset_idx=0, avg_datasets=False, t_0=0, plot_ranges=True, use_log_scale_x=False,
                 use_log_scale_y=False, save_file_name=None, x_label="Optimization steps", y_label="Loss"):
    """
    Plot results for given dataset or averaged from given start t_0
    :param np_results: data to plot
    :param dataset_idx: index of dataset to plot, not needed if avg_datasets=True
    :param avg_datasets: Bool indicating whether to average over datasets or not
    :param t_0: first time step to plot from
    """
    print("Plots, averaged={0} ".format(avg_datasets) +
          (", dataset_idx={0}".format(dataset_idx) if avg_datasets is False else ""))
    plt.figure()
    for optimizer in np_results.keys():
        color = get_color_for_optimizer(optimizer)
        tmp_data = np_results[optimizer] if (avg_datasets or dataset_idx==None) else \
            np_results[optimizer][dataset_idx]

        avg_min_losses, std_min_losses, lower_min_losses, upper_min_losses = \
            get_mean_std_min_losses_per_timestep(tmp_data, t_0=t_0)
        _x = X
        if _x is None:
            _x = range(t_0, t_0 + len(lower_min_losses))
        else:

            _x = X[optimizer]
            avg_min_x, _, _, _ = \
                get_mean_std_min_losses_per_timestep(_x, t_0=t_0)
            _x = avg_min_x
            _x = np.cumsum(_x)
        plt.plot(_x, avg_min_losses, label=get_display_name_for_optimizer(optimizer), color=color)
        if plot_ranges:
            plt.fill_between(x=_x, y1=lower_min_losses,
                             y2=upper_min_losses, alpha=0.3, color=color)

    _x_label = x_label
    _y_label = x_label

    if use_log_scale_x:
        plt.xscale('log')
        _x_label += " (log scale)"
    if use_log_scale_y:
        plt.yscale('log')
        _y_label += " (log scale)"

    plt.xlabel(_x_label)
    plt.ylabel(_y_label)
    plt.legend(loc='best')
    if save_file_name is not None:
        plt.savefig(os.path.join(config.PLOT_FOLDER, save_file_name))
    else:
        plt.show()

    plt.clf()


def get_mean_std_min_losses_per_timestep(data, axes=None, t_0=0):
    """
    Calculate mean and standard deviation of the given experiment
    :param data: shape [dataset_idx, iteration_idx, timesteps] if avg_datasets = True, else
    shape [iteration_idx, timesteps]
    :param avg_datasets: Bool indicating whether to average over datasets or not
    :param t_0: start time step
    :return: mean, std, 25 percentile, 75 percentile per timestep over experiment runs
    """
    min_losses = np.minimum.accumulate(data, axis=-1)
    min_losses = min_losses[..., t_0:]
    if axes == None:
        axes = tuple(range(len(data.shape) - 1))
    avg_min_losses = np.percentile(min_losses, 50, axis=axes)  # np.nanmean(min_losses, axis=axis)
    std_min_losses = np.nanstd(min_losses, axis=axes)
    lower_min_losses = np.percentile(min_losses, 25, axis=axes)
    upper_min_losses = np.percentile(min_losses, 75, axis=axes)
    # print("avg/std min_losses shape: ", avg_min_losses.shape, std_min_losses.shape)
    return avg_min_losses, std_min_losses, lower_min_losses, upper_min_losses


def results_to_numpy(optimizer_results, result_idx=1, negative=True):
    """
    Convert passed experiment results to numpy array
    :param optimizer_results: dict of [optimizer][classifier]
    :param result_idx: result idx:
    0 = tmp_opt.hyperparameter_set_per_timestep,
    1 = tmp_opt.eval_fn_per_timestep,
    2 = tmp_opt.cpu_time_per_opt_timestep,
    3 = tmp_opt.wall_time_per_opt_timestep
    :return: numpy array of specified results
    """
    np_results = {}
    for optimizer in optimizer_results:
        if type(optimizer_results[optimizer]) == dict:
            tmp_results = {}

            opt_results = optimizer_results[optimizer]
            for classifier in opt_results:
                results = np.array(opt_results[classifier])
                results = np.array(results[..., result_idx], dtype=np.float32)
                tmp_results[classifier] = -results if negative else results
            np_results[optimizer] = tmp_results
        else:

            results = np.array(optimizer_results[optimizer])
            results = np.array(results[..., result_idx], dtype=np.float32)
            np_results[optimizer] = -results if negative else results
    return np_results


def save_plotted_progress(optimizer, data=None, name=None, x_lim=None, y_lim=None):
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    data_to_plot = data
    if data_to_plot is None:
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

    p.ax_marg_x.xlim = (0, np.max(x_marginal))
    p.ax_marg_x.ylim = (0, np.max(x_marginal))
    p.ax_marg_x.fill_between(
        x_range,
        x_marginal * x_marginal,
        alpha=0.5,
        clim=(0, np.max(x_marginal))
    )

    p.ax_marg_y.fill_betweenx(
        y_range,
        y_marginal * 0.1,
        alpha=0.5,
        clim=(0, 1e10)
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

    #p.ax_marg_y.scatter(
    #    evaluations_y_sorted,
    #    sample_points_y_sorted,
    #    # orientation = 'horizontal',
    #    alpha=0.5,
    #    color='k')

    plt.savefig(os.path.join(PLOT_FOLDER, _name))
    plt.clf()
    plt.cla()
    plt.close()

def branin(a=1, b=5.1 / (4 * np.pi**2), c=5. / np.pi,
           r=6, s=10, t=1. / (8 * np.pi)):
    # Taken from: https://fluentopt.readthedocs.io/en/latest/auto_examples/plot_dict_format.html
    """Branin-Hoo function is defined on the square x1 ∈ [-5, 10], x2 ∈ [0, 15].
    It has three minima with f(x*) = 0.397887 at x* = (-pi, 12.275),
    (+pi, 2.275), and (9.42478, 2.475).
    More details: <http://www.sfu.ca/~ssurjano/branin.html>

    This code is adapted from : https://github.com/scikit-optimize/scikit-optimize
    """
    def f(d):
        x, y = d['x'], d['y']
        return (a * (y - b * x ** 2 + c * x - r) ** 2 +
                s * (1 - t) * np.cos(x) + s)
    return f

def get_loss_ranges_per_classifier_dataset(losses, max_n_datasets=None):
    """
    Extracts minimum and maximum losses per dataset and classifier
    :param losses: dict of [classifier][parameters:frozenset][dataset_idx] mapping to a float loss
    :param max_n_datasets: max number of datasets to look at
    :return: dict [classifier] mapping to numpy array of shape [N_DATASETS, 2] with the last dimension being (min_val,
    max_val)
    """
    loss_ranges = {}
    for c in losses:
        loss_ranges[c] = []
        for ds_idx in range(len(list(losses[c].values())[0])):
            if max_n_datasets is not None and ds_idx >= max_n_datasets:
                break
            min_val = math.inf
            max_val = -math.inf
            for params in list(losses[c].keys()):
                tmp_val = losses[c][params][ds_idx]
                if tmp_val < min_val:
                    min_val = tmp_val
                if tmp_val > max_val:
                    max_val = tmp_val
            #print("Loss range for classifier {0} and dataset index {1} is {2}".format(c, ds_idx, (min_val, max_val)))
            if min_val == max_val:
                print("SAME MIN AND MAX FOR ", c, ds_idx)
            loss_ranges[c] += [(min_val, max_val)]
        loss_ranges[c] = np.array(loss_ranges[c])
    return loss_ranges

