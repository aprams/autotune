import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))


import random
import param_space
import config
from optimizers import grid_search, random_search, ga_search, gp_search, tpe_search
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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


def branin_grid_search():
    print("-" * 53)
    print("Testing GS")

    optimizer = grid_search.GridSearchOptimizer(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn)
    _ = optimizer.maximize()

    return optimizer


def branin_ga_search(n_iterations=2000):
    print("-" * 53)
    print("2D Example GA fn")
    np.random.seed(0)
    random.seed(0)

    optimizer = ga_search.GeneticAlgorithmSearch(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn,
                                                 n_pops=8, n_iterations=n_iterations, elite_pops_fraction=0.2)
    _ = optimizer.maximize()

    return optimizer


def branin_gp_search(n_iterations=20, gp_n_warmup=100000, gp_n_iter=25, n_restarts_optimizer=5, name='gp'):
    print("-" * 53)
    print("Testing GP")

    np.random.seed(0)
    random.seed(0)

    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": n_restarts_optimizer}
    optimizer = gp_search.GaussianProcessOptimizer(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn,
                                                   n_iterations=n_iterations, gp_n_warmup=gp_n_warmup,
                                                   gp_n_iter=gp_n_iter, name=name, **gp_params)
    _ = optimizer.maximize()

    return optimizer


def branin_random_search(n_iterations=2000):
    print("-" * 53)
    print("Testing RS")

    random.seed(0)
    np.random.seed(0)
    optimizer = random_search.RandomSearchOptimizer(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn,
                                                    n_iterations=n_iterations)
    _ = optimizer.maximize()

    return optimizer


def branin_tpe_search(n_iterations=2000, n_EI_candidates=24, name='TPE'):
    print("-" * 53)
    print("Testing TPE")

    np.random.seed(0)
    random.seed(0)

    optimizer = tpe_search.TPEOptimizer(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn,
                                                   n_iterations=n_iterations, n_EI_candidates=n_EI_candidates,
                                        name=name)
    _ = optimizer.maximize()

    return optimizer


n_iters = 250
gs_optimizer = branin_grid_search()
rs_optimizer = branin_random_search(n_iterations=n_iters)
ga_optimizer = branin_ga_search(n_iterations=n_iters)
gp_optimizer_short = branin_gp_search(n_iterations=n_iters, gp_n_iter=25, gp_n_warmup=100000, name='gp_short')
gp_optimizer_medium = branin_gp_search(n_iterations=n_iters, gp_n_iter=100, gp_n_warmup=100000, name='gp_medium')
gp_optimizer = branin_gp_search(n_iterations=n_iters, gp_n_iter=250, gp_n_warmup=100000)
tpe_optimizer = branin_tpe_search(n_iterations=n_iters)
tpe_optimizer_short = branin_tpe_search(n_iterations=n_iters, n_EI_candidates=5, name='TPE_short')
tpe_optimizer_long = branin_tpe_search(n_iterations=n_iters, n_EI_candidates=50, name='TPE_long')

optimizers = [gs_optimizer,
              rs_optimizer,
              ga_optimizer,
              gp_optimizer_short,
              gp_optimizer_medium,
              gp_optimizer,
              tpe_optimizer,
              tpe_optimizer_short,
              tpe_optimizer_long,
              ]

for i in range(len(optimizers)):
    o = optimizers[i]
    print("branin_plot_", o.name)
    #print(o.hyperparameter_set_per_timestep)
    sample_points = np.array([(x[1], y[1]) if x[0] == 'x' else (y[1], x[1]) for (x, y) in o.hyperparameter_set_per_timestep])
    #print(sample_points)
    print("# Sample points: ", len(sample_points))
    param_ranges = [[branin_param_space[0].space[0], branin_param_space[0].space[1]],
                    [branin_param_space[1].space[0], branin_param_space[1].space[1]]]
    print("Param ranges: ", param_ranges)
    gen_example_2d_plot(sample_points, branin_eval_fn, param_ranges=param_ranges, name="branin_plot_" + o.name)

all_cum_max_data = []
for i in range(len(optimizers)):
    o = optimizers[i]
    print("branin_cum_max_", o.name)
    save_plotted_progress(o)
    eval_fn_per_timestep = np.array(o.eval_fn_per_timestep)
    cumulative_max_data = np.minimum.accumulate(-eval_fn_per_timestep)
    #cumulative_max_data = [min([-x for x in o.eval_fn_per_timestep[0:i+1]]) for i in range(len(o.eval_fn_per_timestep))]
    all_cum_max_data += [cumulative_max_data]
    save_plotted_progress(o, data=cumulative_max_data, name="branin_cum_max_" + o.name)


print("branin_cum_max_all_log_x_y")
for x in all_cum_max_data:
    plt.loglog(x)
plt.legend([o.name for o in optimizers], loc='upper right')
plt.savefig(os.path.join(config.PLOT_FOLDER, './branin_cum_max_all_log_x_y'))

plt.clf()

print("branin_cum_max_all_log_x")
for x in all_cum_max_data:
    plt.semilogx(x)
plt.legend([o.name for o in optimizers], loc='upper right')
plt.savefig(os.path.join(config.PLOT_FOLDER, './branin_cum_max_all_log_x'))

plt.clf()

for x in all_cum_max_data:
    plt.plot(x)
plt.legend([o.name for o in optimizers], loc='lower right')
plt.savefig(os.path.join(config.PLOT_FOLDER, './branin_cum_max_all'))
