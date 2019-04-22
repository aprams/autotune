import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import config
from utils import plot_results, plot_results_multiple, results_to_numpy, plot_cpu_time_per_optimizer, \
    plot_avg_rank_per_timestep


prefix = "branin"

os.makedirs(os.path.join(config.PLOT_FOLDER, prefix), exist_ok=True)

with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, prefix + '_eval_fns_per_timestep.pickle'), 'rb') as handle:
    eval_fns_per_timestep = pickle.load(handle)

with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, prefix + '_results.pickle'), 'rb') as handle:
    results = pickle.load(handle)


# Average rank plotting
avg_rank_plot_save_path = os.path.join(config.PLOT_FOLDER, "branin/avg_rank_per_timestep")
filtered_eval_fns = eval_fns_per_timestep.copy()
print(filtered_eval_fns.keys())
filtered_eval_fns.pop('gp_medium', None)
filtered_eval_fns.pop('TPE_medium', None)
filtered_eval_fns.pop('tpe_medium', None)
filtered_eval_fns.pop('gp_long', None)
filtered_eval_fns.pop('TPE_long', None)
plot_avg_rank_per_timestep(filtered_eval_fns, save_path=avg_rank_plot_save_path, legend_loc="lower left")


# CPU time plotting:
cpu_time_plot_save_path = os.path.join(config.PLOT_FOLDER, "branin/cpu_comparison")
plot_cpu_time_per_optimizer(results, save_path=cpu_time_plot_save_path)

cpu_time_plot_save_path_gp = cpu_time_plot_save_path + "_gp"
plot_cpu_time_per_optimizer({x: results[x] for x in results.keys() if x.lower().startswith('gp')},
                            save_path=cpu_time_plot_save_path_gp, y_scale='linear')

cpu_time_plot_save_path_tpe = cpu_time_plot_save_path + "_tpe"
plot_cpu_time_per_optimizer({x: results[x] for x in results.keys() if x.lower().startswith('tpe')},
                            save_path=cpu_time_plot_save_path_tpe, y_scale='linear')



# Result plotting
plot_results_multiple(eval_fns_per_timestep, dataset_idx=None, save_file_name_prefix=prefix + '/cum_all')
plot_results_multiple(filtered_eval_fns, dataset_idx=None, save_file_name_prefix=prefix + '/cum_all_filtered')

tpe_keys = [k for k in list(eval_fns_per_timestep.keys()) if k.lower().startswith('tpe')]
gp_keys = [k for k in list(eval_fns_per_timestep.keys()) if k.lower().startswith('gp')]

print(tpe_keys)
print(gp_keys)
tpe_dict = {x: eval_fns_per_timestep[x] for x in tpe_keys}
if "TPE" in tpe_dict.keys():
    tpe_dict['TPE_medium'] = tpe_dict['TPE']
    del tpe_dict['TPE']
if "tpe_medium" in tpe_dict.keys():
    tpe_dict['TPE_medium'] = tpe_dict['tpe_medium']
    del tpe_dict['tpe_medium']
plot_results(tpe_dict, dataset_idx=None, save_file_name=prefix + '/cum_tpe_log_x_y', use_log_scale_x=True,
             use_log_scale_y=True)

gp_dict = {x: eval_fns_per_timestep[x] for x in gp_keys}
if "gp" in gp_dict:
    gp_dict['gp_long'] = gp_dict['gp']
    del gp_dict['gp']
plot_results(gp_dict, dataset_idx=None, save_file_name=prefix + '/cum_gp_log_x_y', use_log_scale_x=True,
             use_log_scale_y=True)

