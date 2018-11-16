import random
import numpy as np
from optimizers import ga_search

from test.test_data import sample_params, sample_eval_fn, sample_callback_fn

def test_ga_search():
    np.random.seed(0)
    random.seed(0)

    optimizer = ga_search.GeneticAlgorithmSearch(sample_params, sample_eval_fn, callback_fn=sample_callback_fn,
                                                 n_pops=8, n_unique_samples=1000, elite_pops_fraction=0.2)
    results = optimizer.maximize()
    assert(results[0][1] >= 13.86)
