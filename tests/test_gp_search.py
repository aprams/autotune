import random
import numpy as np
from autotune.optimizers import gp_search

from tests.test_data import sample_params, sample_eval_fn, sample_callback_fn


def test_gp_search(n_iterations=20, gp_n_warmup=100000, gp_n_iter=25, n_restarts_optimizer=5):
    print("-" * 53)
    print("Testing GP")

    np.random.seed(0)
    random.seed(0)

    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": n_restarts_optimizer}
    optimizer = gp_search.GaussianProcessOptimizer(sample_params, sample_eval_fn, callback_fn=sample_callback_fn,
                                                   n_iterations=n_iterations, gp_n_warmup = gp_n_warmup,
                                                   gp_n_iter=gp_n_iter, **gp_params, random_seed=0)
    results = optimizer.maximize()
    assert(results[0][1] > 13.70)
    print("Optimizer wall timings: ", optimizer.wall_time_per_opt_timestep)
    print("Optimizer cpu timings: ", optimizer.cpu_time_per_opt_timestep)

    return optimizer



# warmup: 10^5, gp_n_iter: 250, n_restarts_optimizer=5 => 28.57s
# warmup: 10^2, gp_n_iter: 250, n_restarts_optimizer=5 => 29.505s
# warmup: 10^5, gp_n_iter: 5, n_restarts_optimizer=5 => 4.363s
# warmup: 10^5, gp_n_iter: 5, n_restarts_optimizer=2 => 3.837s
# warmup: 10^5, gp_n_iter: 250, n_restarts_optimizer=2 => 30.330s
# warmup: 10^5, gp_n_iter: 25, n_restarts_optimizer=5 => 6.110s
# warmup: 10^4, gp_n_iter: 25, n_restarts_optimizer=5 => 3.552s
# warmup: 10^5, gp_n_iter: 15, n_restarts_optimizer=5 => 5.549

