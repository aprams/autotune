from hyperopt.pyll import scope
from hyperopt import hp

pca = {'preprocessing': 1, 'pca:keep_variance':
    hp.quniform('pca:keep_variance', 0, 1, 1)} #2

penalty_and_loss = hp.choice('penalty_and_loss',
                             [{'liblinear:penalty': 0, 'liblinear:loss': 0},
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
