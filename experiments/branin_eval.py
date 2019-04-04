import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import multiprocessing as mp
import utils
import random
from autotune import param_space
import config
from autotune.optimizers import grid_search, ga_search, gp_search, tpe_search
from autotune.optimizers import random_search
import matplotlib.pyplot as plt
import numpy as np

from utils import gen_example_2d_plot, branin, save_plotted_progress


def sample_callback_fn(params):
    pass

branin_x = param_space.Real([-5, 10], name='x', n_points_to_sample=200)
branin_y = param_space.Real([0, 15], name='y', n_points_to_sample=200)

branin_param_space = [branin_x, branin_y]
branin_eval_fn = lambda params: -branin()(params)

branin_samples = []

for x in branin_x.create_generator()():
    for y in branin_y.create_generator()():
        branin_samples += [(x, y, -branin_eval_fn({'x': x, 'y': y}))]
branin_samples = np.array(branin_samples)
do_recreate_fun = False

if do_recreate_fun:
    # Make the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(branin_samples[:, 0], branin_samples[:, 1], branin_samples[:, 2], cmap=plt.cm.jet, linewidth=0.2)

    ax.view_init(30, -135)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(x)')

    ax.set_xticks([-5, 0, 5, 10])
    ax.set_yticks([0, 5, 10, 15])

    plt.savefig(os.path.join(config.PLOT_FOLDER, './branin_fun'))


def branin_grid_search(random_seed=None):
    print("-" * 53)
    print("Testing GS")

    optimizer = grid_search.GridSearchOptimizer(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn)
    _ = optimizer.maximize()

    return optimizer


def branin_ga_search(n_iterations=2000, random_seed=None):
    print("-" * 53)
    print("2D Example GA fn")
    np.random.seed(0)
    random.seed(0)

    optimizer = ga_search.GeneticAlgorithmSearch(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn,
                                                 n_pops=8, n_iterations=n_iterations, elite_pops_fraction=0.2,
                                                 random_seed=random_seed)
    _ = optimizer.maximize()

    return optimizer


def branin_gp_search(n_iterations=20, gp_n_warmup=100000, gp_n_iter=25, n_restarts_optimizer=5, name='gp', random_seed=None):
    print("-" * 53)
    print("Testing GP")

    np.random.seed(0)
    random.seed(0)

    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": n_restarts_optimizer}
    optimizer = gp_search.GaussianProcessOptimizer(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn,
                                                   n_iterations=n_iterations, gp_n_warmup=gp_n_warmup,
                                                   gp_n_iter=gp_n_iter, name=name, **gp_params, random_seed=random_seed)
    _ = optimizer.maximize()

    return optimizer


def branin_random_search(n_iterations=2000, random_seed=None):
    print("-" * 53)
    print("Testing RS")

    random.seed(0)
    np.random.seed(0)
    optimizer = random_search.RandomSearchOptimizer(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn,
                                                    n_iterations=n_iterations, random_seed=random_seed)
    _ = optimizer.maximize()

    return optimizer


def branin_tpe_search(n_iterations=2000, n_EI_candidates=24, name='TPE', random_seed=None):
    print("-" * 53)
    print("Testing TPE")

    np.random.seed(0)
    random.seed(0)

    optimizer = tpe_search.TPEOptimizer(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn,
                                        n_iterations=n_iterations, n_EI_candidates=n_EI_candidates,
                                        name=name, random_seed=random_seed)
    _ = optimizer.maximize()

    return optimizer


N_BRANIN_ITERS = 15
N_ITERS_PER_OPT = 5
def worker(i):
    #gs_optimizer = branin_grid_search()
    rs_optimizer = branin_random_search(n_iterations=N_BRANIN_ITERS, random_seed=i)
    #ga_optimizer = branin_ga_search(n_iterations=N_BRANIN_ITERS, random_seed=i)
    #gp_optimizer_short = branin_gp_search(n_iterations=N_BRANIN_ITERS, random_seed=i, gp_n_iter=25, gp_n_warmup=100000, name='gp_short')
    #gp_optimizer_medium = branin_gp_search(n_iterations=N_BRANIN_ITERS, random_seed=i, gp_n_iter=100, gp_n_warmup=100000, name='gp_medium')
    #gp_optimizer = branin_gp_search(n_iterations=N_BRANIN_ITERS, random_seed=i, gp_n_iter=250, gp_n_warmup=100000)
    #tpe_optimizer = branin_tpe_search(n_iterations=N_BRANIN_ITERS, random_seed=i)
    tpe_optimizer_short = branin_tpe_search(n_iterations=N_BRANIN_ITERS, random_seed=i, n_EI_candidates=5, name='TPE_short')
    #tpe_optimizer_long = branin_tpe_search(n_iterations=N_BRANIN_ITERS, random_seed=i, n_EI_candidates=50, name='TPE_long')

    optimizers = [#gs_optimizer,
                  rs_optimizer,
                  #ga_optimizer,
                  #gp_optimizer_short,
                  #gp_optimizer_medium,
                  #gp_optimizer,
                  #tpe_optimizer,
                  tpe_optimizer_short,
                  #tpe_optimizer_long,
                  ]

    results = {}
    for o in optimizers:
        results[o.name] = list(zip(o.hyperparameter_set_per_timestep, o.eval_fn_per_timestep,
                                   o.cpu_time_per_opt_timestep, o.wall_time_per_opt_timestep))
    return results

pool = mp.Pool(6)
results = pool.map(worker, range(N_ITERS_PER_OPT))

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


transposed_results = {}
for opt_name in results[0]:
    transposed_results[opt_name] = [[] for _ in range(N_ITERS_PER_OPT)]
    for i in range(N_ITERS_PER_OPT):
        transposed_results[opt_name][i] = results[i][opt_name]
results = transposed_results
eval_fns_per_timestep = utils.results_to_numpy(results)

# results shape : {opt_name}[N_ITERS_PER_OPT][4][..]

for opt_name in results.keys():
    r = results[opt_name]
    print("branin_plot_", opt_name)
    sample_points = np.array([(x[1], y[1]) if x[0] == 'x' else (y[1], x[1]) for (x, y) in np.array(r)[...,0][0]])
    #print(sample_points)
    print("# Sample points: ", len(sample_points))
    param_ranges = [[branin_param_space[0].space[0], branin_param_space[0].space[1]],
                    [branin_param_space[1].space[0], branin_param_space[1].space[1]]]
    print("Param ranges: ", param_ranges)
    gen_example_2d_plot(sample_points, branin_eval_fn, param_ranges=param_ranges, name="branin_plot_" + opt_name)

all_cum_max_data = []
for opt_name in results.keys():
    print("Eval fns: ", eval_fns_per_timestep[opt_name])
    avg_min_losses, std_min_losses, lower_min_losses, upper_min_losses = get_mean_std_min_losses_per_timestep(eval_fns_per_timestep[opt_name])
    #np.minimum.accumulate(eval_fns_per_timestep[opt_name].T)
    all_cum_max_data += [avg_min_losses]


def plot_results(np_results, dataset_idx=0, avg_datasets=False, t_0=0, plot_ranges=True, use_log_scale=False):
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
    ax = plt.gca()
    for optimizer in np_results.keys():
        color = next(ax._get_lines.prop_cycler)['color']
        tmp_data = np_results[optimizer] if (avg_datasets or dataset_idx==None) else \
            np_results[optimizer][dataset_idx]

        avg_min_losses, std_min_losses, lower_min_losses, upper_min_losses = \
            get_mean_std_min_losses_per_timestep(tmp_data, t_0=t_0)
        plt.plot(range(t_0, t_0 + len(lower_min_losses)), avg_min_losses, label=optimizer + "_cum", color=color)
        if plot_ranges:
            plt.fill_between(x=range(t_0, t_0 + len(lower_min_losses)), y1=lower_min_losses,
                             y2=upper_min_losses, alpha=0.3, color=color)

    plt.xlabel("Optimization steps")
    plt.ylabel("Loss")
    if use_log_scale:
        plt.yscale('log')
    plt.legend(loc='auto')
    plt.show()

plot_results(eval_fns_per_timestep, dataset_idx=None)

print("branin_cum_max_all_log_x_y")
for x in all_cum_max_data:
    print(x)
    plt.loglog(x)
plt.legend([opt_name for opt_name in results], loc='upper right')
plt.savefig(os.path.join(config.PLOT_FOLDER, './branin_cum_max_all_log_x_y'))

plt.clf()

print("branin_cum_max_all_log_x")
for x in all_cum_max_data:
    plt.semilogx(x)
plt.legend([opt_name for opt_name in results], loc='upper right')
plt.savefig(os.path.join(config.PLOT_FOLDER, './branin_cum_max_all_log_x'))

plt.clf()

for x in all_cum_max_data:
    plt.plot(x)
plt.legend([opt_name for opt_name in results], loc='lower right')
plt.savefig(os.path.join(config.PLOT_FOLDER, './branin_cum_max_all'))
