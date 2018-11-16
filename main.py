import param_space
from optimizers import grid_search, random_search

bool_var = param_space.Bool(name='bool_var')
bool_var_2 = param_space.Bool(name='bool_var_2')

real_var = param_space.Real([-5, 5], name='real_var')
real_var_2 = param_space.Real([-5, 0], projection_fn=lambda x: 10**x, name='real_var_2')

int_var = param_space.Integer([0, 1, 2], name='int_var')
int_var_2 = param_space.Integer([0, 1, 2], projection_fn=lambda x: x**2, name='int_var_2')

params = [bool_var, bool_var_2, real_var, real_var_2, int_var, int_var_2]


def eval_fn(params):
    return params['int_var'] * params['real_var'] + params['int_var_2'] * params['real_var_2']


def callback_fn(**params):
    print(params)


optimizer = grid_search.GridSearchOptimizer(params, eval_fn, callback_fn=callback_fn)
optimizer.maximize()


optimizer = random_search.RandomSearchOptimizer(params, eval_fn, callback_fn=callback_fn)
optimizer.maximize()
