import os
import pickle
import math
import config
import random
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from autotune import param_space
from scipy.stats import beta
from utils import plot_results, plot_results_multiple


prefix="structure"

os.makedirs(os.path.join(config.PLOT_FOLDER, prefix), exist_ok=True)

with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, prefix + '_eval_fns_per_timestep.pickle'), 'rb') as handle:
    w_d_idxd_results = pickle.load(handle)

# plot sample functions
def eval_fn(xs, params):
    result = 1
    def projection_fn(x, a, b, min_point, **kwargs):  # multiplier):
        return math.fabs(beta.ppf(x, a, b) - min_point)
    for i in range(len(params)):
        param = params[i]
        result *= (1 + projection_fn(xs[i], param['a'], param['b'], param['min_point'])) * param['multiplier']
    return - result

n_sample_points = 200
for seed in range(100):
    d = 2
    params=[]
    for i in range(d):
        random.seed(seed)
        tmp_param = {}
        tmp_param['a'] = random.uniform(0.1, 5)
        tmp_param['b'] = random.uniform(0.1, 5)
        tmp_param['min_point'] = random.uniform(0, 1)
        tmp_param['multiplier'] = random.uniform(0.5, 2.0)
        params.append(tmp_param)

    _x = param_space.Real([0, 1], name='x', n_points_to_sample=n_sample_points)
    _y = param_space.Real([0, 1], name='y', n_points_to_sample=n_sample_points)

    _samples = []
    for x in _x.create_generator()():
        for y in _y.create_generator()():
            _samples += [(x, y, -eval_fn([x, y], params))]
    _samples = np.array(_samples)

    # Make the plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(_samples[:, 0], _samples[:, 1], _samples[:, 2], cmap=plt.cm.jet, linewidth=0.0)

    ax.view_init(30, -135)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(x)')

    #ax.set_xticks([-5, 0, 5, 10])
    #ax.set_yticks([0, 5, 10, 15])
    os.makedirs(os.path.join(config.PLOT_FOLDER, 'structure', 'sample_funs'), exist_ok=True)
    plt.savefig(os.path.join(config.PLOT_FOLDER, 'structure', 'sample_funs', './sample_fun_' + str(seed)))


print(w_d_idxd_results.keys())
print(w_d_idxd_results[list(w_d_idxd_results.keys())[0]].keys())

for k, eval_fns_per_timestep in w_d_idxd_results.items():
    print(k)
    k_dict = dict(k)
    w_d_id = "w{0}d{1}".format(k_dict['width'], k_dict['depth'])
    plot_results_multiple(eval_fns_per_timestep, dataset_idx=None, save_file_name_prefix=prefix + "/" + w_d_id + 'cum_all')

    tpe_keys = [k for k in list(eval_fns_per_timestep.keys()) if k.lower().startswith('tpe')]
    gp_keys = [k for k in list(eval_fns_per_timestep.keys()) if k.lower().startswith('gp')]

    print(tpe_keys)
    print(gp_keys)

    plot_results({x: eval_fns_per_timestep[x] for x in tpe_keys}, dataset_idx=None, save_file_name=prefix + '/' + w_d_id +'cum_tpe_log_x_y', use_log_scale_x=True,
                 use_log_scale_y=True)

    plot_results({x: eval_fns_per_timestep[x] for x in gp_keys}, dataset_idx=None, save_file_name=prefix + '/' + w_d_id + 'cum_gp_log_x_y', use_log_scale_x=True,
                 use_log_scale_y=True)