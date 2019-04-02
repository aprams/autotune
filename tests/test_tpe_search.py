import random
import numpy as np
from autotune.optimizers import tpe_search

from tests.test_data import sample_params, sample_eval_fn, sample_callback_fn


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
