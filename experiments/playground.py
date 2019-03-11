"""

param1 = Real('a', 1, 2)
param2 = Real('b', 1, 2)

param3 = Conditional('c', cond_var=param1,
                     [(lambda param: param > 1, 1),
                      (lambda param: param <= 1, 2)])
"""

from hyperopt import hp, fmin, tpe
from optimizers.tpe_search import TPEOptimizer
space = hp.choice('classifier_type', [
    {
        'type': 'naive_bayes',
    },
    {
        'type': 'svm',
        'C': hp.lognormal('svm_C', 0, 1),
        'kernel': hp.choice('svm_kernel', [
            {'ktype': 'linear'},
            {'ktype': 'RBF', 'width': hp.lognormal('svm_rbf_width', 0, 1)},
            ]),
    },
    {
        'type': 'dtree',
        'criterion': hp.choice('dtree_criterion', ['gini', 'entropy']),
        'max_depth': hp.choice('dtree_max_depth',
            [None, hp.qlognormal('dtree_max_depth_int', 3, 1, 1)]),
        'min_samples_split': hp.qlognormal('dtree_min_samples_split', 2, 1, 1),
    },
    ])

def objective(args):
    print(args)
    if args['type'] == 'naive_bayes':
        print('yey')
        return -10.0
    if args['type'] == 'svm':
        return 2.0

    return 1.0

#best = fmin(objective,
#    space=space,
#    algo=tpe.suggest,
#    max_evals=100)
print(type(space))
print(type(objective))
opt = TPEOptimizer(space, eval_fn=objective, callback_fn=lambda: print('-'))
opt.maximize()