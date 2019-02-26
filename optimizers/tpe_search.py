from param_space import Param, Real, Integer, Bool
from typing import Callable

from .hyper_param_opt import AbstractHyperParameterOptimizer
from hyperopt import tpe, Trials, fmin, hp

class TPEOptimizer(AbstractHyperParameterOptimizer):
    name = "TPE"
    def __init__(self, hyper_param_list: list, eval_fn: Callable, callback_fn: Callable, verbose: int = 1,
                 n_iterations=50):
        self.n_iterations = n_iterations
        super().__init__(hyper_param_list, eval_fn, callback_fn, verbose, should_call_eval_fn=False)
        self.tpe_space = self._create_tpe_space_from_param_space(hyper_param_list)
        self.name = "TPE"

    def transform_raw_param_samples(self, pop):
        print("pop:", pop.keys())
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
                fmin(fn=lambda params: -self.eval_fn(self.transform_raw_param_samples(params)),
                     space=self.tpe_space, algo=tpe.suggest, max_evals=i + 1, trials=bayes_trials)
                yield [bayes_trials.vals[x][-1] for x in bayes_trials.vals.keys()], bayes_trials.results[-1]['loss']
        return tpe_generator

    def maximize(self) -> dict:
        generator = self._create_hyperparam_set_generator()
        for next_hyperparam_set, eval_metric in generator():
            next_hyperparam_set_dict = {}
            for i in range(len(next_hyperparam_set)):
                next_hyperparam_set_dict[self.hyper_param_list[i].name] = next_hyperparam_set[i]
            self._add_sampled_point(next_hyperparam_set_dict, eval_metric)
            self._on_optimizer_step_done(next_hyperparam_set_dict, eval_metric)

        self._on_optimizer_done()
        print("======")
        sorted_results = sorted(self.params_to_results_dict.items(), key=lambda kv: kv[1], reverse=True)
        print(len(sorted_results))
        for k, v in sorted_results[0:10]:
            print("{0}: {1}".format(k, v))
        print("======")
        return sorted_results