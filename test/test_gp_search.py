import random
import numpy as np
from optimizers import gp_search

from test.test_data import sample_params, sample_eval_fn, sample_callback_fn
#from .utils import save_plotted_progress

def test_gp_search():
    np.random.seed(0)
    random.seed(0)

    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 5}
    optimizer = gp_search.GaussianProcessOptimizer(sample_params, sample_eval_fn, callback_fn=sample_callback_fn,
                                                   n_iterations=50, gp_n_warmup = 10, gp_n_iter=2, **gp_params)
    results = optimizer.maximize()
    assert(results[0][1] >= 12.006)

    #save_plotted_progress(optimizer.eval_fn_per_timestep, 'gp')
