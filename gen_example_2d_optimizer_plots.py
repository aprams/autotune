import random
import numpy as np
from optimizers import grid_search, random_search, ga_search, gp_search, tpe_search

from test.test_data import sample_callback_fn, example_2d_params, example_2d_eval_fn
from test.utils import gen_example_2d_plot


def example_2d_grid_search():
    print("-" * 53)
    print("Testing GS")

    optimizer = grid_search.GridSearchOptimizer(example_2d_params, example_2d_eval_fn, callback_fn=sample_callback_fn)
    _ = optimizer.maximize()

    return optimizer


def example_2d_ga_search(n_iterations=2000):
    print("-" * 53)
    print("2D Example GA fn")
    np.random.seed(0)
    random.seed(0)

    optimizer = ga_search.GeneticAlgorithmSearch(example_2d_params, example_2d_eval_fn, callback_fn=sample_callback_fn,
                                                 n_pops=8, n_iterations=n_iterations, elite_pops_fraction=0.2)
    _ = optimizer.maximize()

    return optimizer


def example_2d_gp_search(n_iterations=20, gp_n_warmup=100000, gp_n_iter=25, n_restarts_optimizer=5):
    print("-" * 53)
    print("Testing GP")

    np.random.seed(0)
    random.seed(0)

    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": n_restarts_optimizer}
    optimizer = gp_search.GaussianProcessOptimizer(example_2d_params, example_2d_eval_fn, callback_fn=sample_callback_fn,
                                                   n_iterations=n_iterations, gp_n_warmup=gp_n_warmup,
                                                   gp_n_iter=gp_n_iter, **gp_params)
    _ = optimizer.maximize()

    return optimizer


def example_2d_random_search(n_iterations=2000):
    print("-" * 53)
    print("Testing RS")

    random.seed(0)
    np.random.seed(0)
    optimizer = random_search.RandomSearchOptimizer(example_2d_params, example_2d_eval_fn, callback_fn=sample_callback_fn,
                                                    n_iterations=n_iterations)
    _ = optimizer.maximize()

    return optimizer


def example_2d_tpe_search(n_iterations=2000):
    print("-" * 53)
    print("Testing TPE")

    np.random.seed(0)
    random.seed(0)

    optimizer = tpe_search.TPEOptimizer(example_2d_params, example_2d_eval_fn, callback_fn=sample_callback_fn,
                                                   n_iterations=n_iterations)
    _ = optimizer.maximize()

    return optimizer



gs_optimizer = example_2d_grid_search()
rs_optimizer = example_2d_random_search(n_iterations=9)
ga_optimizer = example_2d_ga_search(n_iterations=9)
gp_optimizer = example_2d_gp_search(n_iterations=9, gp_n_iter=25, gp_n_warmup=1000)
tpe_optimizer = example_2d_tpe_search(n_iterations=9)

optimizers = [gs_optimizer, rs_optimizer, ga_optimizer, gp_optimizer, tpe_optimizer]

for i in range(len(optimizers)):
    o = optimizers[i]
    #print(o.hyperparameter_set_per_timestep)
    sample_points = np.array([(x[1], y[1]) if x[0] == 'x' else (y[1], x[1]) for (x, y) in o.hyperparameter_set_per_timestep])
    print(sample_points)
    print("# Sample points: ", len(sample_points))
    param_ranges = [[example_2d_params[0].space[0], example_2d_params[0].space[1]],
                    [example_2d_params[1].space[0], example_2d_params[1].space[1]]]
    print("Param ranges: ", param_ranges)
    gen_example_2d_plot(sample_points, example_2d_eval_fn, param_ranges=param_ranges, name="example_plot_" + o.name)
