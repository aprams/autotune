from typing import Callable
from abc import ABC, abstractmethod
import time
import random
import numpy as np

class AbstractHyperParameterOptimizer(ABC):
    name = "abstract"

    def __init__(self, hyper_param_list: list, eval_fn: Callable, callback_fn: Callable=None, n_iterations=None,
                 verbose: int=0, should_call_eval_fn=True, random_seed=None, name="abstract"):
        self.hyper_param_list = hyper_param_list
        self.eval_fn = eval_fn
        self.callback_fn = callback_fn if callback_fn is not None else lambda: None
        self.params_to_results_dict = {}
        self.eval_fn_per_timestep = []
        self.hyperparameter_set_per_timestep = []
        self.cpu_time_per_opt_timestep = []
        self.wall_time_per_opt_timestep = []
        self.verbose = verbose
        self.name = name
        self.should_call_eval_fn = should_call_eval_fn

        self.last_wall_time = None
        self.last_cpu_time = None
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(random_seed)

        random.seed(random_seed)
        np.random.seed(random_seed)


    def initialize(self):
        pass

    @abstractmethod
    def _create_hyperparam_set_generator(self):
        pass

    def maximize(self) -> list:
        generator = self._create_hyperparam_set_generator()
        self._on_pre_hyp_opt_step()
        for next_hyperparam_set in generator():
            self._on_post_hyp_opt_step()
            if self.should_call_eval_fn:
                eval_metric = self.eval_fn(next_hyperparam_set)

                self._add_sampled_point(next_hyperparam_set, eval_metric)
            self._on_optimizer_step_done(self.hyperparameter_set_per_timestep[-1], self.eval_fn_per_timestep[-1])
            self._on_pre_hyp_opt_step()
        self._on_optimizer_done()


    def _add_sampled_point(self, hyperparameter_set: dict, eval_metric: float):
        self.params_to_results_dict[frozenset(hyperparameter_set.items())] = eval_metric
        self.eval_fn_per_timestep += [eval_metric]
        self.hyperparameter_set_per_timestep += [frozenset(hyperparameter_set.items())]

    def _on_optimizer_step_done(self, hyperparameter_set: dict, eval_metric: float):
        if self.verbose >= 2:
            print("Parameter set {0} yielded result metric of: {1}".format(hyperparameter_set, eval_metric))

    def _on_optimizer_done(self):
        if self.verbose >= 1:
            print("Optimizer finished")
            sorted_results = sorted(self.params_to_results_dict.items(), key=lambda kv: kv[1], reverse=True)
            if self.verbose == 1:
                print("======")
                print(len(sorted_results))
                for k, v in sorted_results[0:10]:
                    print("{0}: {1}".format(k, v))
                print("======")
            return sorted_results
        if self.verbose >= 2:
            for hyperparameter_set, eval_metric in self.params_to_results_dict.items():
                print("Parameter set {0} yielded result metric of: {1}".format(dict(hyperparameter_set), eval_metric))

    def _on_pre_hyp_opt_step(self):
        self.last_cpu_time = time.clock()
        self.last_wall_time = time.time()

    def _on_post_hyp_opt_step(self):
        self.cpu_time_per_opt_timestep += [time.clock() - self.last_cpu_time]
        self.wall_time_per_opt_timestep += [time.time() - self.last_wall_time]

