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


example_2d_x_var = param_space.Real([1, 5], name='x',  n_points_to_sample=5)
example_2d_y_var = param_space.Real([1, 5], name='y',  n_points_to_sample=5)

example_2d_params = [example_2d_x_var, example_2d_y_var]

def example_2d_eval_fn(params):
    a1 = 76.333
    a2 = -47.0
    a3 = 11.66667
    a4 = -1.0
    offset = -39
    return offset + a1 * params['x'] + a2 * params['x']**2 + a3 * params['x']**3 + a4 * params['x']**4