import random
import numpy as np
from optimizers import gp_search

from test.test_data import sample_params, sample_eval_fn, sample_callback_fn, example_2d_params, example_2d_eval_fn
from .utils import save_plotted_progress

def test_gp_search(n_iterations=20, gp_n_warmup=100000, gp_n_iter=25, n_restarts_optimizer=5):
    print("-" * 53)
    print("Testing GP")

    np.random.seed(0)
    random.seed(0)

    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": n_restarts_optimizer}
    optimizer = gp_search.GaussianProcessOptimizer(sample_params, sample_eval_fn, callback_fn=sample_callback_fn,
                                                   n_iterations=n_iterations, gp_n_warmup = gp_n_warmup,
                                                   gp_n_iter=gp_n_iter, **gp_params)
    results = optimizer.maximize()
    assert(results[0][1] > 13.70)

    return optimizer

