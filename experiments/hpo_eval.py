import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], 'hpo_dataset'))
sys.path.insert(1, sys.path[0])

import pickle
import utils
import config
import os
import multiprocessing as mp

from hpo_dataset import hpo_utils
from hpo_space import tpe_combined_spaces, tpe_spaces
from autotune.optimizers import ga_search, tpe_search, gp_search
from autotune.optimizers import random_search


N_DATASETS = 42  # 42
N_REPS_PER_OPTIMIZER = 10  # 10
N_OPT_STEPS = 100  # 100

opt_and_params = [(random_search.RandomSearchOptimizer, {}),
                  #(tpe_search.TPEOptimizer, {'n_startup_jobs': 5, 'n_EI_candidates': 5, 'name': 'TPE_short'}),
                  (tpe_search.TPEOptimizer, {'n_startup_jobs': 5, 'name': 'TPE'}),
                  #(tpe_search.TPEOptimizer, {'n_startup_jobs': 5, 'n_EI_candidates': 50, 'name': 'TPE_long'}),
                  (gp_search.GaussianProcessOptimizer, {'gp_n_iter': 25, 'gp_n_warmup': 1000, 'name': 'GP_short'}),
                  #(gp_search.GaussianProcessOptimizer, {'gp_n_iter': 100, 'name': 'GP_medium'}),
                  #(gp_search.GaussianProcessOptimizer, {'gp_n_iter': 250, 'name': 'GP_long'}),
                  (ga_search.GeneticAlgorithmSearch, {}),
                  ]#(grid_search.GridSearchOptimizer, {) ]

classifier_indexed_params, classifier_param_spaces, classifier_combined_spaces, classifiers = hpo_utils.load_hpo_data()

# Experiment resulst always like:
# [N_ITERS_PER_OPT], ..., {opt_name}, [T]?
# e.g.: [n_iters], {classifier}, {opt_name}, [T]?

def worker(i):
    results = {}
    print("=" * 46)
    print("Worker ", i, "started")
    for classifier in classifier_indexed_params.keys():
        results[classifier] = {}
        for optimizer, opt_params in opt_and_params:
            opt_name = opt_params['name'] if 'name' in opt_params else optimizer.name
            print("Worker {0} evaluating optimizer {1} with params {2}".format(i, optimizer.name, opt_params))
            results[classifier][opt_name] = [[] for _ in range(N_DATASETS)]
            for dataset_idx in range(N_DATASETS):
                def eval_fn(tpe_params):
                    params = utils.prune_invalid_params_for_classifier(utils.flatten(tpe_params), classifier)
                    loss = -classifier_indexed_params[classifier][frozenset(params.items())][dataset_idx]
                    return loss

                if optimizer == tpe_search.TPEOptimizer:
                    tmp_opt = optimizer(tpe_spaces[classifier], eval_fn,
                                        n_iterations=N_OPT_STEPS, random_seed=i, verbose=0)
                else:
                    tmp_opt = optimizer(classifier_param_spaces[classifier], eval_fn,
                                        n_iterations=N_OPT_STEPS, random_seed=i, verbose=0)

                _ = tmp_opt.maximize()
                tmp_results = list(zip(tmp_opt.hyperparameter_set_per_timestep, tmp_opt.eval_fn_per_timestep,
                                       tmp_opt.cpu_time_per_opt_timestep, tmp_opt.wall_time_per_opt_timestep))
                results[classifier][opt_name][dataset_idx] = tmp_results

    with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, 'hpo_dataset_optimizer_results_{0}.pickle'.format(i)),
              'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results

pool = mp.Pool(config.N_MP_PROCESSES)
pool_results = pool.map(worker, range(N_REPS_PER_OPTIMIZER))

new_optimizer_results = {}

for classifier in classifier_indexed_params.keys():
    new_optimizer_results[classifier] = {}  # manager.dict()
    for optimizer, opt_params in opt_and_params:
        opt_name = opt_params['name'] if 'name' in opt_params else optimizer.name
        new_optimizer_results[classifier][opt_name] = [[[] for _ in range(N_REPS_PER_OPTIMIZER)] for _ in range(N_DATASETS)]
        for i in range(N_REPS_PER_OPTIMIZER):
            for dataset_idx in range(N_DATASETS):
                new_optimizer_results[classifier][opt_name][dataset_idx][i] = pool_results[i][classifier][opt_name][dataset_idx]

optimizer_results = new_optimizer_results


# Combined classifier and param search

def combined_worker(i):
    results = {}
    print("=" * 46)
    print("combined_worker ", i, "started")
    for optimizer, opt_params in opt_and_params:
        print("Combined worker {0} evaluating optimizer {1} with params {2}".format(i, optimizer.name, opt_params))
        opt_name = opt_params['name'] if 'name' in opt_params else optimizer.name
        results[opt_name] = [[] for _ in range(N_DATASETS)]

        for dataset_idx in range(N_DATASETS):
            def eval_fn(tpe_params):
                params = utils.flatten(tpe_params)
                tmp_c = params['classifier']
                c = classifiers[tmp_c] if type(tmp_c) is int else tmp_c
                params = utils.prune_invalid_params_for_classifier(params, c)
                loss = -classifier_indexed_params[c][frozenset(params.items())][dataset_idx]
                return loss

            if optimizer == tpe_search.TPEOptimizer:
                tmp_opt = optimizer(tpe_combined_spaces, eval_fn,
                                    n_iterations=N_OPT_STEPS, random_seed=i, verbose=0)
            else:
                tmp_opt = optimizer(classifier_combined_spaces, eval_fn,
                                    n_iterations=N_OPT_STEPS, random_seed=i, verbose=0)
            _ = tmp_opt.maximize()
            tmp_results = list(zip(tmp_opt.hyperparameter_set_per_timestep, tmp_opt.eval_fn_per_timestep,
                                   tmp_opt.cpu_time_per_opt_timestep, tmp_opt.wall_time_per_opt_timestep))
            results[opt_name][dataset_idx] = tmp_results
        with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, 'combined_hpo_dataset_optimizer_results_{0}.pickle'.format(i)),
                  'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return results


pool = mp.Pool(config.N_MP_PROCESSES)
pool_results = pool.map(combined_worker, range(N_REPS_PER_OPTIMIZER))

new_combined_optimizer_results = {}

for optimizer, opt_params in opt_and_params:
    opt_name = opt_params['name'] if 'name' in opt_params else optimizer.name
    new_combined_optimizer_results[opt_name] = {}  # manager.dict()

    new_combined_optimizer_results[opt_name] = [[[] for _ in range(N_REPS_PER_OPTIMIZER)] for _ in range(N_DATASETS)]
    for i in range(N_REPS_PER_OPTIMIZER):
        for dataset_idx in range(N_DATASETS):
            new_combined_optimizer_results[opt_name][dataset_idx][i] = pool_results[i][opt_name][dataset_idx]

combined_optimizer_results = new_combined_optimizer_results

with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, 'hpo_dataset_optimizer_results.pickle'), 'wb') as handle:
    pickle.dump(optimizer_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, 'combined_hpo_dataset_optimizer_results.pickle'),
                  'wb') as handle:
    pickle.dump(combined_optimizer_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
