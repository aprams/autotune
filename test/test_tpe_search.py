import random
import numpy as np
from optimizers import tpe_search

from test.test_data import sample_params, sample_eval_fn, sample_callback_fn, example_2d_params, example_2d_eval_fn
from .utils import save_plotted_progress

def test_tpe_search(n_iterations=100):
    print("-" * 53)
    print("Testing TPE")

    np.random.seed(0)
    random.seed(0)

    optimizer = tpe_search.TPEOptimizer(sample_params, sample_eval_fn, callback_fn=sample_callback_fn,
                                                   n_iterations=n_iterations)
    results = optimizer.maximize()
    assert(results[0][1] > 13.70)

    return optimizer
