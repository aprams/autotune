from param_space import Param, Real, Integer, Bool
from typing import Callable

from .hyper_param_opt import AbstractHyperParameterOptimizer


import logging
logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.core import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


class BOHB_Worker(Worker):
    def __init__(self, run_id, nameserver=None, nameserver_port=None, logger=None, host=None, id=None, timeout=None,
                 callback_fn=None):
        super().__init__(run_id, nameserver=None, nameserver_port=None, logger=None, host=None, id=None, timeout=None)
        self.callback_fn = callback_fn


class BOHBOptimizer(AbstractHyperParameterOptimizer):
    def __init__(self, hyper_param_list: list, eval_fn: Callable, callback_fn: Callable, verbose: int = 1):
        super().__init__(hyper_param_list, eval_fn, callback_fn, verbose)
        self.tpe_space = self._create_tpe_space_from_param_space(hyper_param_list)
        self.name = "BOHB"

    def transform_raw_param_samples(self, pop):
        print("Hyper params:", pop.keys())
        param_dict = {}
        for i in range(len(self.hyper_param_list)):
            cur_hyp_param = self.hyper_param_list[i]
            param_dict[cur_hyp_param.name] = \
                cur_hyp_param.transform_raw_sample(pop[cur_hyp_param.name])
        return param_dict

    def _create_bhob_space_from_param_space(self, hyper_param_list):
        cs = CS.ConfigurationSpace()

        for param in hyper_param_list:
            tmp_bound = None
            if type(param) is Real:
                tmp_bound = (param.space[0], param.space[1])
            elif type(param) is Integer:
                tmp_bound = (0, len(param.space) - 1e-5)
            elif type(param) is Bool:
                tmp_bound = (0, 2.0 - 1e-5)
            assert tmp_bound is not None
            tmp_param = CSH.UniformFloatHyperparameter(param.name, lower=tmp_bound[0], upper=tmp_bound[1])
            cs.add_hyperparameters([tmp_param])
        return cs

    def _create_hyperparam_set_generator(self):
        def tpe_generator():
            bayes_trials = Trials()

            # Optimize
            for i in range(self.n_iterations):
                fmin(fn=lambda params: -self.eval_fn(self.transform_raw_param_samples(params)),
                     space=self.tpe_space, algo=tpe.suggest, max_evals=i + 1, trials=bayes_trials)
                yield [bayes_trials.vals[x][-1] for x in bayes_trials.vals.keys()], bayes_trials.results[-1]['loss']
        return tpe_generator
