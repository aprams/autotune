import random
import numpy as np
from autotune.optimizers import ga_search

from tests.test_data import sample_params, sample_eval_fn, sample_callback_fn


def test_ga_search(n_iterations=2000):
    print("-" * 53)
    print("Testing GA")
    np.random.seed(0)
    random.seed(0)

    optimizer = ga_search.GeneticAlgorithmSearch(sample_params, sample_eval_fn, callback_fn=sample_callback_fn,
                                                 n_pops=8, n_iterations=n_iterations, elite_pops_fraction=0.2,
                                                 random_seed=0)
    results = optimizer.maximize()
    assert(results[0][1] >= 13.86)
    print("Optimizer wall timings: ", optimizer.wall_time_per_opt_timestep)
    print("Optimizer cpu timings: ", optimizer.cpu_time_per_opt_timestep)
    return optimizer
