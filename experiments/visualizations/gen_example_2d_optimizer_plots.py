import random
import numpy as np
from autotune.optimizers import grid_search, ga_search, gp_search, tpe_search
from autotune.optimizers import random_search

from tests.test_data import sample_callback_fn, example_2d_params, example_2d_eval_fn
from utils import gen_example_2d_plot


seed=3
np.random.seed(seed)
random.seed(seed)

def example_2d_grid_search():
    print("-" * 53)
    print("Testing GS")

    optimizer = grid_search.GridSearchOptimizer(example_2d_params, example_2d_eval_fn, callback_fn=sample_callback_fn)
    _ = optimizer.maximize()

    return optimizer


def example_2d_ga_search(n_iterations=2000):
    print("-" * 53)
    print("2D Example GA fn")


    optimizer = ga_search.GeneticAlgorithmOptimizer(example_2d_params, example_2d_eval_fn, callback_fn=sample_callback_fn,
                                                    n_pops=8, n_iterations=n_iterations, elite_pops_fraction=0.2,
                                                    random_seed=seed)
    _ = optimizer.maximize()

    return optimizer


def example_2d_gp_search(n_iterations=20, gp_n_warmup=1000, gp_n_iter=25, n_restarts_optimizer=2, seed=None):
    print("-" * 53)
    print("Testing GP")


    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": n_restarts_optimizer}
    n_init_points = 5
    optimizer = gp_search.GaussianProcessOptimizer(example_2d_params, example_2d_eval_fn, callback_fn=sample_callback_fn,
                                                   n_iterations=n_iterations, gp_n_warmup=gp_n_warmup,
                                                   gp_n_iter=gp_n_iter, n_init_points=n_init_points,
                                                   random_seed=seed, **gp_params)
    _ = optimizer.maximize()

    return optimizer


def example_2d_random_search(n_iterations=2000, seed=None):
    print("-" * 53)
    print("Testing RS")

    optimizer = random_search.RandomSearchOptimizer(example_2d_params, example_2d_eval_fn, callback_fn=sample_callback_fn,
                                                    random_seed=seed, n_iterations=n_iterations)
    _ = optimizer.maximize()

    return optimizer


def example_2d_tpe_search(n_iterations=2000, seed=None):
    print("-" * 53)
    print("Testing TPE")

    optimizer = tpe_search.TPEOptimizer(example_2d_params, example_2d_eval_fn, callback_fn=sample_callback_fn,
                                        n_iterations=n_iterations, random_seed=seed)
    _ = optimizer.maximize()
    return optimizer


gs_optimizer = example_2d_grid_search()
rs_optimizer = example_2d_random_search(n_iterations=15, seed=seed)
ga_optimizer = example_2d_ga_search(n_iterations=25)
gp_optimizer = example_2d_gp_search(n_iterations=10, gp_n_iter=25, gp_n_warmup=1000, seed=seed)
tpe_optimizer = example_2d_tpe_search(n_iterations=15, seed=seed)

optimizers = [gs_optimizer, rs_optimizer, ga_optimizer, gp_optimizer, tpe_optimizer]

for i in range(len(optimizers)):
    o = optimizers[i]

    #print(o.hyperparameter_set_per_timestep)
    sample_points = np.array([(x[1], y[1]) if x[0] == 'x' else (y[1], x[1]) for (x, y) in o.hyperparameter_set_per_timestep])
    if type(o) is gp_search.GaussianProcessOptimizer:
        random_sample_points = np.array([(x[1], y[1]) if x[0] == 'x' else (y[1], x[1]) for (x, y) in
                                         rs_optimizer.hyperparameter_set_per_timestep])
        sample_points = np.concatenate([sample_points, random_sample_points[:5]], axis=0)
    print(sample_points)
    print("# Sample points: ", len(sample_points))
    param_ranges = [[example_2d_params[0].space[0], example_2d_params[0].space[1]],
                    [example_2d_params[1].space[0], example_2d_params[1].space[1]]]
    print("Param ranges: ", param_ranges)
    gen_example_2d_plot(sample_points, lambda params: 6.2-example_2d_eval_fn(params), param_ranges=param_ranges, name="example_plot_" + o.name)
