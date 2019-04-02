from hyperopt import hp
from hyperopt.pyll import scope

pca = {'preprocessing': 'pca', 'pca:keep_variance': scope.int(
    hp.quniform('pca:keep_variance', 0, 1, 1))} #2

penalty_and_loss = hp.choice('penalty_and_loss',
                             [{'liblinear:penalty': 'l1', 'liblinear:loss': 'l2'},
                              #{'liblinear:penalty': 'l2', 'liblinear:loss': 'l1'},
                              {'liblinear:penalty': 'l2', 'liblinear:loss': 'l2'}]) # 2
liblinear_LOG2_C = scope.int(hp.quniform('liblinear:LOG2_C', -5, 15, 1)) # 21
liblinear = {'classifier': 'liblinear', 'liblinear:penalty_and_loss': penalty_and_loss, 'liblinear:LOG2_C': liblinear_LOG2_C}
# 1, 3, 21 = 63

libsvm_LOG2_C = scope.int(hp.quniform('libsvm_svc:LOG2_C', -5, 15, 1)) # 21
libsvm_LOG2_gamma = scope.int(hp.quniform('libsvm_svc:LOG2_gamma', -15, 3, 1)) # 18/19
libsvm_svc = {'classifier': 'libsvm_svc', 'libsvm_svc:LOG2_C': libsvm_LOG2_C, 'libsvm_svc:LOG2_gamma': libsvm_LOG2_gamma}
# 21 * 19 = 399
criterion = hp.choice('random_forest:criterion', ['gini', 'entropy']) # 2
max_features = scope.int(hp.quniform('random_forest:max_features', 1, 10, 1)) # 10
min_samples_split = scope.int(hp.quniform('random_forest:min_samples_split', 0, 4, 1)) # 5
random_forest = {'classifier': 'random_forest', 'random_forest:criterion': criterion, 'random_forest:max_features': max_features, 'random_forest:min_samples_split': min_samples_split}
# 2 * 10 * 5 = 100

preprocessors = {'None': 'None', 'pca': pca} # 3
classifiers = {'libsvm_svc': libsvm_svc, # 399 * 3 = 1197
               'liblinear': liblinear, # 42 * 3 = 126
               'random_forest': random_forest # 100 * 3 = 300
                }

space = {'classifier': hp.choice('classifier', classifiers.values()),
         'preprocessing': hp.choice('preprocessing', preprocessors.values())}

# 1224 = 8 * 9 * 17


# liblinear: 0 - 125 = 126
# RF: 126 - 425 = 300
# libsvm_svc: 1622 - 426 = 1197