import random
import functools
from autotune.param_space import Real, Integer, Bool
from typing import Callable

from .hyper_param_opt import AbstractHyperParameterOptimizer
from hyperopt import tpe, Trials, fmin, hp
import numpy as np

class TPEOptimizer(AbstractHyperParameterOptimizer):
    name = "TPE"
    def __init__(self, hyper_param_list: list, eval_fn: Callable, callback_fn: Callable=None, n_iterations=50,
                 verbose: int=0, random_seed=None, name="TPE", n_startup_jobs=5, n_EI_candidates=24):
        self.n_iterations = n_iterations
        super().__init__(hyper_param_list=hyper_param_list, eval_fn=eval_fn, callback_fn=callback_fn,
                         n_iterations=n_iterations, verbose=verbose, should_call_eval_fn=False,
                         random_seed=random_seed, name=name)
        is_hpo_space = False if type(hyper_param_list) is list else True
        self.tpe_space = self._create_tpe_space_from_param_space(hyper_param_list) if not is_hpo_space else hyper_param_list
        self.is_hpo_space = is_hpo_space
        self.name = name

        tpe.suggest = functools.partial(tpe.suggest, n_startup_jobs=n_startup_jobs, n_EI_candidates=n_EI_candidates)

    def transform_raw_param_samples(self, pop):
        #print("TPE params:", pop.keys())
        param_dict = {}
        for i in range(len(self.hyper_param_list)):
            cur_hyp_param = self.hyper_param_list[i]
            param_dict[cur_hyp_param.name] = \
                cur_hyp_param.transform_raw_sample(pop[cur_hyp_param.name])
        return param_dict

    def _create_tpe_space_from_param_space(self, hyper_param_list):
        tpe_space = {}
        for param in hyper_param_list:
            tmp_bound = None
            if type(param) is Real:
                tmp_bound = (param.space[0], param.space[1])
            elif type(param) is Integer:
                tmp_bound = (0, len(param.space) - 1e-5)
            elif type(param) is Bool:
                tmp_bound = (0, 2.0 - 1e-5)
            assert tmp_bound is not None
            tpe_space[param.name] = hp.uniform(param.name, tmp_bound[0], tmp_bound[1])
        return tpe_space

    def _create_hyperparam_set_generator(self):
        def tpe_generator():
            bayes_trials = Trials()
            # Optimize
            for i in range(self.n_iterations):
                fmin(fn=lambda params: -self.eval_fn(params),
                     space=self.tpe_space, algo=tpe.suggest, max_evals=i + 1,
                     trials=bayes_trials,
                     rstate=self.random_state)
                yield {x: bayes_trials.vals[x][-1] for x in bayes_trials.vals.keys() if i in bayes_trials.idxs[x]}, -bayes_trials.results[-1]['loss']

        return tpe_generator

    def maximize(self) -> dict:
        generator = self._create_hyperparam_set_generator()

        self._on_pre_hyp_opt_step()
        for next_hyperparam_set, eval_metric in generator():
            self._on_post_hyp_opt_step()
            next_hyperparam_set_dict = next_hyperparam_set
            self._add_sampled_point(next_hyperparam_set_dict, eval_metric)
            self._on_optimizer_step_done(next_hyperparam_set_dict, eval_metric)
            self._on_pre_hyp_opt_step()
        return self._on_optimizer_done()

