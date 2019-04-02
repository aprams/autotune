from param_space import Param
from typing import Callable

from .hyper_param_opt import AbstractHyperParameterOptimizer


class RandomSearchOptimizer(AbstractHyperParameterOptimizer):
    name="RandomSearch"
    def __init__(self, hyper_param_list: list, eval_fn: Callable, callback_fn: Callable=None, verbose: int = 1,
                 n_iterations=5000, random_seed=None, name="RandomSearch"):
        self.n_iterations = n_iterations
        super().__init__(hyper_param_list, eval_fn, callback_fn, verbose)
        self.name = name

    def _sample_random_hyperparam_set(self):
        sampled_params = {}
        for param in self.hyper_param_list:
            sampled_params[param.name] = param.sample()
        return sampled_params

    def _create_hyperparam_set_generator(self):
        def generator() -> Param:
            for i in range(self.n_iterations):
                yield self._sample_random_hyperparam_set()

        return generator

