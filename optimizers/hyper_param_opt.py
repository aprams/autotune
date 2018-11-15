from abc import ABC, abstractmethod


class AbstractHyperParameterOptimizer(ABC):
    def __init__(self, hyper_param_list: list, eval_fn: function, callback_fn: function, verbose: int=2):
        self.hyper_param_list = hyper_param_list
        self.eval_fn = eval_fn
        self.callback_fn = callback_fn
        self.params_to_results_dict = {}
        self.params_to_extra_output_dict = {}
        self.verbose = verbose

    def initialize(self):
        pass

    @abstractmethod
    def _create_hyperparam_set_generator(self):
        pass

    def maximize(self) -> dict:
        for next_hyperparam_set in self._create_hyperparam_set_generator():
            eval_metric, extra_output = self.eval_fn(next_hyperparam_set)
            self.params_to_extra_output_dict[next_hyperparam_set] = extra_output

            self._add_sampled_point(next_hyperparam_set, eval_metric)
            self._on_optimizer_step_done(next_hyperparam_set, eval_metric)

        self._on_optimizer_done()
        return self.params_to_results_dict

    def _add_sampled_point(self, hyperparameter_set: list, eval_metric: float):
        self.params_to_results_dict[hyperparameter_set] = eval_metric

    def _on_optimizer_step_done(self, hyperparameter_set: list, eval_metric: float):
        if self.verbose >= 1:
            print("Parameter set {0} yielded result metric of: {1}".format(hyperparameter_set, eval_metric))

    def _on_optimizer_done(self):
        if self.verbose >= 1:
            print("Optimizer finished")
        if self.verbose >= 2:
            for hyperparameter_set, eval_metric in self.params_to_results_dict:
                print("Parameter set {0} yielded result metric of: {1}".format(hyperparameter_set, eval_metric))

