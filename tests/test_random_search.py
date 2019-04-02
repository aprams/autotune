import random
import numpy as np
from autotune.optimizers import random_search

from tests.test_data import sample_params, sample_eval_fn, sample_callback_fn


def test_random_search(n_iterations=1000):
    print("-" * 53)
    print("Testing RS")

    random.seed(0)
    np.random.seed(0)
    optimizer = random_search.RandomSearchOptimizer(sample_params, sample_eval_fn, callback_fn=sample_callback_fn,
                                                    n_iterations=n_iterations)
    results = optimizer.maximize()
    assert(results[0][1] >= 10.19)

    return optimizer
