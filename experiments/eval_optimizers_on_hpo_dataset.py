import pickle
import numpy as np
from optimizers import grid_search, random_search, ga_search, gp_search, tpe_search
from preprocess_hpo_dataset import create_index_param_space
from hyperopt.pyll import scope
from hyperopt import hp, fmin, tpe
from utils import get_loss_ranges_per_classifier_dataset
import param_space
import functools
import config
import os


def random_seed_fn(i):
    return i

optimizers = [random_search.RandomSearchOptimizer, tpe_search.TPEOptimizer,# gp_search.GaussianProcessOptimizer,
               ga_search.GeneticAlgorithmSearch,
             ]#grid_search.GridSearchOptimizer ]

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
classifier_combined_spaces += [param_space.Integer(space=list(range(len(list(classifier_param_spaces.keys())))), name='classifier')]
list(classifier_param_spaces.keys())
for c in classifier_param_spaces.keys():
    for p in classifier_param_spaces[c]:
        classifier_combined_spaces += [p]
classifiers = list(classifier_param_spaces.keys())

print("=" * 46)
print("Combined parameter space")
for p in classifier_combined_spaces:
    print(p.name, p.space)

n_datasets = 3 # 42
n_repititions_per_optimizer = 10 # 10
optimizer_steps = 50 # 100
optimizer_results = {}
tpe_startup_jobs = 5
tpe.suggest = functools.partial(tpe.suggest, n_startup_jobs=tpe_startup_jobs)


loss_ranges_per_classifier_dataset = get_loss_ranges_per_classifier_dataset(classifier_indexed_params, max_n_datasets=n_datasets)

# per classifier tests
for optimizer in optimizers:
    print("=" * 46)
    print("Evaluating optimizer", optimizer)
    optimizer_results[optimizer.name] = {}
    for classifier in classifier_indexed_params.keys():
        print("Evaluating classifier", classifier)
        optimizer_results[optimizer.name][classifier] = []
        for dataset_idx in range(n_datasets):
            tmp_agg_results = []
            for i in range(n_repititions_per_optimizer):
                def eval_fn(params):
                    modified_params = dict(params)
                    if modified_params['preprocessing'] == 0:
                        del modified_params['pca:keep_variance']
                    if optimizer == tpe_search.TPEOptimizer:
                        for k in modified_params:
                            print(k, modified_params[k])
                            modified_params[k] = round(modified_params[k])
                    # minimize val error
                    return -classifier_indexed_params[classifier][frozenset(modified_params.items())][dataset_idx]

                def tpe_eval_fn(tpe_params):
                    #print("TPE PARAMS: ", tpe_params)
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
                    del params['classifier']
                    #print(params)
                    loss = -classifier_indexed_params[classifier][frozenset(params.items())][dataset_idx]
                    #print("TPE params {0} yielded a loss of {1}".format(params, loss))
                    return loss

                def sample_callback_fn(**params):
                    #print(params)
                    pass

                if optimizer == tpe_search.TPEOptimizer:
                    tmp_opt = optimizer(tpe_spaces[classifier], tpe_eval_fn, #callback_fn=sample_callback_fn,
                                        n_iterations=optimizer_steps, random_seed=random_seed_fn(i), verbose=0)
                else:
                    tmp_opt = optimizer(classifier_param_spaces[classifier], eval_fn, #callback_fn=sample_callback_fn,
                                        n_iterations=optimizer_steps, random_seed=random_seed_fn(i), verbose=0)

                _ = tmp_opt.maximize()
                tmp_results = list(zip(tmp_opt.hyperparameter_set_per_timestep, tmp_opt.eval_fn_per_timestep,
                                       tmp_opt.cpu_time_per_opt_timestep, tmp_opt.wall_time_per_opt_timestep))
                tmp_agg_results += [tmp_results]
            optimizer_results[optimizer.name][classifier] += [tmp_agg_results]


# Combined classifier and param search
combined_optimizer_results = {}
for optimizer in optimizers:
    print("=" * 46)
    print("Evaluating optimizer", optimizer)
    combined_optimizer_results[optimizer.name] = []
    for dataset_idx in range(n_datasets):
        tmp_agg_results = []
        for i in range(n_repititions_per_optimizer):
            def eval_fn(params):
                modified_params = dict(params)
                final_params = {}
                classifier_idx = params['classifier']
                classifier = classifiers[classifier_idx]
                if modified_params['preprocessing'] == 0:
                    del modified_params['pca:keep_variance']
                for k in modified_params:
                    is_valid_key_for_classifier = str.startswith(k, classifier) or str.startswith(k, 'preprocessing') \
                                                  or str.startswith(k, 'pca')
                    if is_valid_key_for_classifier:
                        final_params[k] = modified_params[k]

                # minimize val error
                return -classifier_indexed_params[classifier][frozenset(final_params.items())][dataset_idx]

            def tpe_eval_fn(tpe_params):
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
                loss = -classifier_indexed_params[classifier][frozenset(params.items())][dataset_idx]
                return loss

            def sample_callback_fn(**params):
                pass

            if optimizer == tpe_search.TPEOptimizer:
                tmp_opt = optimizer(tpe_combined_spaces, tpe_eval_fn, #callback_fn=sample_callback_fn,
                                    n_iterations=optimizer_steps, random_seed=random_seed_fn(i), verbose=0)
            else:
                tmp_opt = optimizer(classifier_combined_spaces, eval_fn, #callback_fn=sample_callback_fn,
                                    n_iterations=optimizer_steps, random_seed=random_seed_fn(i), verbose=0)
            _ = tmp_opt.maximize()
            tmp_results = list(zip(tmp_opt.hyperparameter_set_per_timestep, tmp_opt.eval_fn_per_timestep,
                                   tmp_opt.cpu_time_per_opt_timestep, tmp_opt.wall_time_per_opt_timestep))
            tmp_agg_results += [tmp_results]
        combined_optimizer_results[optimizer.name] += [tmp_agg_results]


optimizer_results['meta'] = {}
optimizer_results['meta']['loss_ranges'] = loss_ranges_per_classifier_dataset

with open('../experiment_results/hpo_dataset_optimizer_results.pickle', 'wb') as handle:
    pickle.dump(optimizer_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('../experiment_results/combined_hpo_dataset_optimizer_results.pickle', 'wb') as handle:
    pickle.dump(combined_optimizer_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

