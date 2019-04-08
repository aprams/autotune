import os
import pickle

import config
from utils import plot_results

prefix="structure_"

with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, prefix + 'eval_fns_per_timestep.pickle'), 'rb') as handle:
    eval_fns_per_timestep = pickle.load(handle)

with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, prefix + 'results.pickle'), 'rb') as handle:
    results = pickle.load(handle)



plot_results(eval_fns_per_timestep, dataset_idx=None, save_file_name=prefix + 'cum_all_log_x_y', use_log_scale_x=True,
             use_log_scale_y=True)

plot_results(eval_fns_per_timestep, dataset_idx=None, save_file_name=prefix + 'cum_all_log_x', use_log_scale_x=True,
             use_log_scale_y=False)

plot_results(eval_fns_per_timestep, dataset_idx=None, save_file_name=prefix + 'cum_all', use_log_scale_x=False,
             use_log_scale_y=False)

tpe_keys = [k for k in list(eval_fns_per_timestep.keys()) if k.lower().startswith('tpe')]
gp_keys = [k for k in list(eval_fns_per_timestep.keys()) if k.lower().startswith('gp')]

print(tpe_keys)
print(gp_keys)

plot_results({x: eval_fns_per_timestep[x] for x in tpe_keys}, dataset_idx=None, save_file_name=prefix + 'cum_tpe_log_x_y', use_log_scale_x=True,
             use_log_scale_y=True)

plot_results({x: eval_fns_per_timestep[x] for x in gp_keys}, dataset_idx=None, save_file_name=prefix + 'cum_gp_log_x_y', use_log_scale_x=True,
             use_log_scale_y=True)