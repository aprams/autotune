import param_space

bool_var = param_space.Bool(name='bool_var')
bool_var_2 = param_space.Bool(name='bool_var_2')

real_var = param_space.Real([-5, 5], name='real_var', n_points_to_sample=20)
real_var_2 = param_space.Real([-5, 0], projection_fn=lambda x: 10**x, name='real_var_2', n_points_to_sample=20)

int_var = param_space.Integer([0, 1, 2], name='int_var')
int_var_2 = param_space.Integer([0, 1, 2], projection_fn=lambda x: x**2, name='int_var_2')

sample_params = [bool_var, bool_var_2, real_var, real_var_2, int_var, int_var_2]


def sample_eval_fn(params):
    return params['int_var'] * params['real_var'] + params['int_var_2'] * params['real_var_2']


def sample_callback_fn(**params):
    print(params)