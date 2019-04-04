import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import pickle
from autotune.optimizers import ga_search, tpe_search, gp_search
from autotune.optimizers import random_search
from preprocess_hpo_dataset import create_index_param_space
from hyperopt.pyll import scope
from hyperopt import hp
from utils import get_loss_ranges_per_classifier_dataset
from autotune import param_space
import config
import os


def random_seed_fn(i):
    return i


opt_and_params = [(random_search.RandomSearchOptimizer, {}),
                  #(tpe_search.TPEOptimizer, {'n_startup_jobs': 5, 'n_EI_candidates': 5, 'name': 'TPE_short'}),
                  (tpe_search.TPEOptimizer, {'n_startup_jobs': 5, 'name': 'TPE'}),
                  #(tpe_search.TPEOptimizer, {'n_startup_jobs': 5, 'n_EI_candidates': 50, 'name': 'TPE_long'}),
                  #(gp_search.GaussianProcessOptimizer, {'gp_n_iter': 25, 'name': 'GP_short'}),
                  #(gp_search.GaussianProcessOptimizer, {'gp_n_iter': 100, 'name': 'GP_medium'}),
                  #(gp_search.GaussianProcessOptimizer, {'gp_n_iter': 250, 'name': 'GP_long'}),
                  (ga_search.GeneticAlgorithmSearch, {}),
                  ]#(grid_search.GridSearchOptimizer, {) ]

pca = {'preprocessing': 1, 'pca:keep_variance':
    hp.quniform('pca:keep_variance', 0, 1, 1)} #2

penalty_and_loss = hp.choice('penalty_and_loss',
                             [{'liblinear:penalty': 0, 'liblinear:loss': 0},
                              #{'liblinear:penalty': 'l2', 'liblinear:loss': 'l1'},
                              {'liblinear:penalty': 1, 'liblinear:loss': 0}]) # 2
liblinear_LOG2_C = scope.int(hp.quniform('liblinear:LOG2_C', 0, 20, 1)) # 21
liblinear = {'classifier': 'liblinear', 'liblinear:penalty_and_loss': penalty_and_loss, 'liblinear:C': liblinear_LOG2_C}
# 1, 3, 21 = 63

libsvm_LOG2_C = scope.int(hp.quniform('libsvm_svc:C', 0, 20, 1)) # 21
libsvm_LOG2_gamma = scope.int(hp.quniform('libsvm_svc:gamma', 0, 18, 1)) # 18/19
libsvm_svc = {'classifier': 'libsvm_svc', 'libsvm_svc:C': libsvm_LOG2_C, 'libsvm_svc:gamma': libsvm_LOG2_gamma}
# 21 * 19 = 399
criterion = hp.choice('random_forest:criterion', [0, 1]) # 2
max_features = scope.int(hp.quniform('random_forest:max_features', 0, 9, 1)) # 10
min_samples_split = scope.int(hp.quniform('random_forest:min_samples_split', 0, 4, 1)) # 5
random_forest = {'classifier': 'random_forest', 'random_forest:criterion': criterion, 'random_forest:max_features': max_features, 'random_forest:min_samples_split': min_samples_split}
# 2 * 10 * 5 = 100

preprocessors = {'None': 0, 'pca': pca} # 3
classifiers_params = {'libsvm_svc': libsvm_svc, # 399 * 3 = 1197
               'liblinear': liblinear, # 42 * 3 = 126
               'random_forest': random_forest # 100 * 3 = 300
                }

tpe_combined_spaces = {'classifier': hp.choice('classifier', classifiers_params.values()),
         'preprocessing': hp.choice('preprocessing', preprocessors.values())}
tpe_spaces = {x: {x:classifiers_params[x],
                  'preprocessing': hp.choice('preprocessing', preprocessors.values())
                  } for x in classifiers_params.keys()}

# Load preprocessed data dicts
with open(os.path.join(config.HPO_FOLDER, 'preprocessed_data.pickle'), 'rb') as handle:
    classifier_indexed_params = pickle.load(handle)

with open(os.path.join(config.HPO_FOLDER, 'preprocessed_param_values.pickle'), 'rb') as handle:
    classifier_param_values = pickle.load(handle)

classifier_param_spaces = {}
for k in classifier_indexed_params.keys():
    print(k)
    param_list = create_index_param_space(classifier_param_values[k])
    for p in param_list:
        print(p.name, p.space)
    classifier_param_spaces[k] = param_list

classifier_combined_spaces = []
classifier_combined_spaces += [
    param_space.Integer(space=list(range(len(list(classifier_param_spaces.keys())))), name='classifier')]
list(classifier_param_spaces.keys())
for c in classifier_param_spaces.keys():
    for p in classifier_param_spaces[c]:
        classifier_combined_spaces += [p]
classifiers = list(classifier_param_spaces.keys())

print("=" * 46)
print("Combined parameter space")
for p in classifier_combined_spaces:
    print(p.name, p.space)

N_DATASETS = 42  # 42
N_REPS_PER_OPTIMIZER = 10  # 10
N_OPT_STEPS = 15  # 100
import multiprocessing as mp
manager = mp.Manager()

loss_ranges_per_classifier_dataset = get_loss_ranges_per_classifier_dataset(classifier_indexed_params, max_n_datasets=N_DATASETS)


#print("Result dict:")
#print(optimizer_results)
#print(optimizer_results[k] for k in optimizer_results.keys())

def worker(i):
    optimizer_results = {}

    for optimizer, opt_params in opt_and_params:
        opt_name = opt_params['name'] if 'name' in opt_params else optimizer.name
        optimizer_results[opt_name] = {}#manager.dict()
        for classifier in classifier_indexed_params.keys():
            print("yee", classifier)
            optimizer_results[opt_name][classifier] = [[] for _ in range(N_DATASETS)]
            #[[[] for _ in range(N_REPS_PER_OPTIMIZER)] for _ in
                                                  #     range(N_DATASETS)]
    print("=" * 46)
    print("REP ", i)
    for optimizer, opt_params in opt_and_params:
        print("Evaluating optimizer", optimizer, "with params", opt_params)
        opt_name = opt_params['name'] if 'name' in opt_params else optimizer.name
        for classifier in classifier_indexed_params.keys():
            print("Evaluating classifier", classifier, i)
            for dataset_idx in range(N_DATASETS):
                def eval_fn(tpe_params):
                    params = {}
                    for k in tpe_params:
                        if type(tpe_params[k]) == int:
                            params[k] = tpe_params[k]
                            continue
                        elif type(tpe_params[k]) == dict:
                            for x in tpe_params[k]:
                                if type(tpe_params[k][x]) is dict:
                                    for y in tpe_params[k][x]:
                                        params[y] = tpe_params[k][x][y]
                                    continue
                                else:
                                    params[x] = tpe_params[k][x]
                        else:
                            raise Exception('unhandled type')
                    if 'classifier' in params:
                        del params['classifier']

                    if params['preprocessing'] == 0 and 'pca:keep_variance' in params:
                        del params['pca:keep_variance']
                    #print(params)
                    loss = -classifier_indexed_params[classifier][frozenset(params.items())][dataset_idx]
                    #print("TPE params {0} yielded a loss of {1}".format(params, loss))
                    return loss


                if optimizer == tpe_search.TPEOptimizer:
                    tmp_opt = optimizer(tpe_spaces[classifier], eval_fn,
                                        n_iterations=N_OPT_STEPS, random_seed=random_seed_fn(i), verbose=0)
                else:
                    tmp_opt = optimizer(classifier_param_spaces[classifier], eval_fn,
                                        n_iterations=N_OPT_STEPS, random_seed=random_seed_fn(i), verbose=0)

                _ = tmp_opt.maximize()
                tmp_results = list(zip(tmp_opt.hyperparameter_set_per_timestep, tmp_opt.eval_fn_per_timestep,
                                       tmp_opt.cpu_time_per_opt_timestep, tmp_opt.wall_time_per_opt_timestep))
                optimizer_results[opt_name][classifier][dataset_idx] = tmp_results

    with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, 'hpo_dataset_optimizer_results_{0}.pickle'.format(i)),
              'wb') as handle:
        pickle.dump(optimizer_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return optimizer_results

pool = mp.Pool(10)
results = pool.map(worker, range(N_REPS_PER_OPTIMIZER))
print(results)
print(len(results))
print(len(results[0]))

new_optimizer_results = {}

def transpose_worker_results(results):
    pass

for optimizer, opt_params in opt_and_params:
    opt_name = opt_params['name'] if 'name' in opt_params else optimizer.name
    new_optimizer_results[opt_name] = {}  # manager.dict()
    for classifier in classifier_indexed_params.keys():
        print("yee", classifier)
        new_optimizer_results[opt_name][classifier] = [[[] for _ in range(N_REPS_PER_OPTIMIZER)] for _ in range(N_DATASETS)]
        for i in range(N_REPS_PER_OPTIMIZER):
            for dataset_idx in range(N_DATASETS):
                new_optimizer_results[opt_name][classifier][dataset_idx][i] = results[i][opt_name][classifier][dataset_idx]

optimizer_results = new_optimizer_results

# Combined classifier and param search
combined_optimizer_results = {}
for i in range(N_REPS_PER_OPTIMIZER):
    print("=" * 46)
    print("REP ", i)
    for optimizer, opt_params in opt_and_params:
        print("Evaluating optimizer", optimizer, "with params", opt_params)
        opt_name = opt_params['name'] if 'name' in opt_params else optimizer.name
        if i == 0:
            combined_optimizer_results[opt_name] = [[[] for _ in range(N_REPS_PER_OPTIMIZER)] for _ in range(N_DATASETS)]

        for dataset_idx in range(N_DATASETS):
            def eval_fn(tpe_params):
                params = {}
                for k in tpe_params:
                    if type(tpe_params[k]) == int:
                        params[k] = tpe_params[k]
                        continue
                    elif type(tpe_params[k]) == dict:
                        for x in tpe_params[k]:
                            if type(tpe_params[k][x]) is dict:
                                for y in tpe_params[k][x]:
                                    params[y] = tpe_params[k][x][y]
                                continue
                            else:
                                params[x] = tpe_params[k][x]
                    else:
                        raise Exception('unhandled type')
                classifier = params['classifier']
                del params['classifier']
                if type(classifier) is int:
                    classifier = classifiers[classifier]
                final_params = {}
                for k in params:
                    is_valid_key_for_classifier = str.startswith(k, classifier) or str.startswith(k, 'preprocessing') \
                                                  or str.startswith(k, 'pca')
                    if is_valid_key_for_classifier:
                        final_params[k] = params[k]
                params = final_params

                if params['preprocessing'] == 0 and 'pca:keep_variance' in params:
                    del params['pca:keep_variance']

                loss = -classifier_indexed_params[classifier][frozenset(params.items())][dataset_idx]
                return loss

            if optimizer == tpe_search.TPEOptimizer:
                tmp_opt = optimizer(tpe_combined_spaces, eval_fn,
                                    n_iterations=N_OPT_STEPS, random_seed=random_seed_fn(i), verbose=0)
            else:
                tmp_opt = optimizer(classifier_combined_spaces, eval_fn,
                                    n_iterations=N_OPT_STEPS, random_seed=random_seed_fn(i), verbose=0)
            _ = tmp_opt.maximize()
            tmp_results = list(zip(tmp_opt.hyperparameter_set_per_timestep, tmp_opt.eval_fn_per_timestep,
                                   tmp_opt.cpu_time_per_opt_timestep, tmp_opt.wall_time_per_opt_timestep))
            combined_optimizer_results[opt_name][dataset_idx][i] = tmp_results
        with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, 'combined_hpo_dataset_optimizer_results_{0}.pickle'.format(i)),
                  'wb') as handle:
            pickle.dump(combined_optimizer_results, handle, protocol=pickle.HIGHEST_PROTOCOL)



optimizer_results['meta'] = {}
optimizer_results['meta']['loss_ranges'] = loss_ranges_per_classifier_dataset

with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, 'hpo_dataset_optimizer_results.pickle'), 'wb') as handle:
    pickle.dump(optimizer_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, 'combined_hpo_dataset_optimizer_results.pickle'),
                  'wb') as handle:
    pickle.dump(combined_optimizer_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
