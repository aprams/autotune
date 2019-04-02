from autotune.optimizers import grid_search

from tests.test_data import sample_params, sample_eval_fn, sample_callback_fn


def test_grid_search():
    print("-" * 53)
    print("Testing GS")

    optimizer = grid_search.GridSearchOptimizer(sample_params, sample_eval_fn, callback_fn=sample_callback_fn)
    results = optimizer.maximize()
    assert(results[0][1] == 14.0)

    return optimizer

