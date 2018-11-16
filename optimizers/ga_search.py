import numpy as np

from param_space import Param
from typing import Callable

from .hyper_param_opt import AbstractHyperParameterOptimizer


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class GeneticAlgorithmSearch(AbstractHyperParameterOptimizer):
    def __init__(self, hyper_param_list: list, eval_fn: Callable, callback_fn: Callable, verbose: int = 1, n_pops=5,
                 n_unique_samples=100, elite_pops_fraction = 0.2):
        super().__init__(hyper_param_list, eval_fn, callback_fn, verbose)
        self.n_pops = n_pops
        self.pops = self.gen_random_pops(self.n_pops)
        self.best_pops = self.pops
        self.n_elite_pops = int(self.n_pops * elite_pops_fraction)
        self.n_unique_samples = n_unique_samples
        self.cur_gen_pops = []
        self.last_tested_pop_result = -np.inf

    def _create_hyperparam_set_generator(self):
        return self.ga_search

    def run_pop_policy(self, pop):
        """
        Evaluates a single pop
        :param pop:
        :return:
        """
        pop_params = self.transform_raw_param_samples_for_pop(pop)
        param_eval_metric = self.eval_fn(pop_params)
        return param_eval_metric

    def crossover(self, pop1, pop2):
        """
        Breeds two pops
        :return: new pop, mix of the parents
        """
        pop_idx = np.random.randint(len(pop1))
        new_pop = np.where(pop_idx, pop1, pop2)
        return new_pop

    def mutate(self, pop, rate=0.3):
        """
        Mutates a single pop with the given rate
        :param pop: pop to mutate
        :param rate: Probability per gene to mutate
        :return: mutated pop
        """
        rand_results = np.random.rand(len(pop))
        rand_pop = self.gen_random_pops(1)[0]  # np.random.randint(action_space_size, size=[len(pop)])
        pop = np.where(rand_results > rate, pop, rand_pop)
        return pop

    def transform_raw_param_samples_for_pop(self, pop):
        """
        Generates a param_dict from a pop's genes
        :param pop: pop to get param_dict for
        :return: param_dict containing net, learning rate, etc.
        """
        param_dict = {}
        for i in range(len(self.hyper_param_list)):
            param_dict[list(self.hyper_param_list)[i].name] = \
                self.hyper_param_list[i].transform_raw_sample(pop[i])
        return param_dict

    def gen_random_pops(self, n_pops):
        """
        Generates a number of randomly initialized pops
        :param n_pops: number of pops to generate
        :return: array of pops
        """
        rand_pops = []
        for i in range(n_pops):
            pop = [param.raw_sample() for param in self.hyper_param_list]
            rand_pops += [pop]
        return np.array(rand_pops)

    def breed_new_generation(self, pop_losses):
        n_pops = len(self.pops)
        pops_shape = [len(self.pops), len(self.pops[0])]
        mutated_pops = np.zeros(shape=pops_shape, dtype=np.float32)
        best_pops_idx = np.argsort(-pop_losses)[:self.n_elite_pops]

        # Crossover & mutate selected pops
        for i in np.arange(n_pops - self.n_elite_pops):
            crossover_pop_idx = np.random.choice(pops_shape[0], 2, p=softmax(pop_losses), replace=False) #/ np.sum(pop_losses))
            crossover_pops = self.pops[crossover_pop_idx]
            new_pop = self.crossover(crossover_pops[0], crossover_pops[1])
            new_pop = self.mutate(new_pop)
            mutated_pops[i] = new_pop

        mutated_pops[-self.n_elite_pops:] = self.pops[best_pops_idx]
        self.best_pops = self.pops[best_pops_idx]
        self.pops = mutated_pops

    def _add_sampled_point(self, hyperparameter_set: list, eval_metric: float):
        super()._add_sampled_point(hyperparameter_set, eval_metric)
        self.last_tested_pop_result = eval_metric

    def get_cached_result(self, pop):
        pop_idx = frozenset(pop.items())
        if pop_idx in self.params_to_results_dict.keys():
            return self.params_to_results_dict[pop_idx]
        else:
            return None

    def ga_search(self):
        """
        Computes the optimal parameters for a dict of data dicts (indexed by image_size) using GAs
        :param param_eval_fn: function which evaluates a given set of parameters on a given dataset
        :param data_dict: data dict containing data & labels
        :param n_pops: number of pops per generation
        :param n_iterations: number of generations to breed
        :param param_eval_metric: dict key for param_eval_fn's output which will be used as a loss
        :return: dict of optimal hyperparameters
        """

        n_unique_pop_counter = 0
        i_generation = 0
        # Determine number of classes
        while True:
            pop_losses = np.zeros(shape=[self.n_pops])
            # Run environment for pop
            for pop_idx in np.arange(self.n_pops):
                pop = self.pops[pop_idx]
                transformed_pop = self.transform_raw_param_samples_for_pop(pop)
                cached_result = self.get_cached_result(transformed_pop)
                if cached_result is None:
                    n_unique_pop_counter += 1
                    yield transformed_pop
                    loss = self.last_tested_pop_result
                else:
                    loss = cached_result
                pop_losses[pop_idx] = loss
                if n_unique_pop_counter >= self.n_unique_samples:
                    return

            #print("Average loss in episode {}: {:0.5f}".format(i_generation, np.sum(pop_losses) / self.n_pops))
            #print("Best loss in episode {}: {:0.5f}".format(i_generation, np.max(pop_losses)))

            self.breed_new_generation(pop_losses=pop_losses)