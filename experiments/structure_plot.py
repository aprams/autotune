import os
import pickle

import config
from utils import plot_results, plot_results_multiple

prefix="structure"

os.makedirs(os.path.join(config.PLOT_FOLDER, prefix), exist_ok=True)

with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, prefix + '_eval_fns_per_timestep.pickle'), 'rb') as handle:
    w_d_idxd_results = pickle.load(handle)

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