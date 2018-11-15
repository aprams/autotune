from .hyper_param_opt import AbstractHyperParameterOptimizer
from param_space import Param

class GridSearchOptimizer(AbstractHyperParameterOptimizer):

    def _get_all_param_combinations(self, dicts: list=None, tmp_dict: dict=None, i: int=0):
        """
        Recursively creates a list of dicts holding all possible combinations of parameters from cfg.parameter_space
        :param dicts: list of multiple dict instances
        :param tmp_dict: instance of current parameter dictionary
        :param i: parameter index
        :return: list of parameter dictionaries
        """
        if i == 0:
            tmp_dict = {}
            dicts = []
        if i == len(self.hyper_param_list):
            dicts.extend([dict])
        else:
            param = list(self.hyper_param_list())[i]
            for j in param.create_generator():
                tmp_dict[param] = j
                self._get_all_param_combinations(dicts, tmp_dict.copy(), i + 1)
        return dicts

    def _create_hyperparam_set_generator(self):
        hyperparam_combinations = self._get_all_param_combinations()

        def generator() -> Param:
            for x in hyperparam_combinations:
                yield x

        return generator