import os
import psutil


N_MP_PROCESSES = psutil.cpu_count(logical=False)

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))


EXPERIMENT_RESULTS_FOLDER = "experiments/results"
EXPERIMENT_RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), EXPERIMENT_RESULTS_FOLDER)
if not os.path.exists(EXPERIMENT_RESULTS_FOLDER):
    os.makedirs(EXPERIMENT_RESULTS_FOLDER)

HPO_FOLDER = "experiments/hpo_dataset"
HPO_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), HPO_FOLDER)
if not os.path.exists(HPO_FOLDER):
    os.makedirs(HPO_FOLDER)

PLOT_FOLDER = os.path.join(ROOT_FOLDER, "experiments/visualizations/plots/")
PLOT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), PLOT_FOLDER)
os.makedirs(PLOT_FOLDER, exist_ok=True)

_OPTIMIZER_CONVERSION_NAMES = {
    'TPE' : 'TPE_short'
}

_OPTIMIZER_TO_COLOR_DICT = {
    'GA' : '#ff7f0e',
    'gp_short': '#d62728',
    'gp_medium': '#e377c2',
    'gp_long': '#9467bd',
    'tpe_short': '#2ca02c',
    'tpe_medium': '#8c564b',
    'tpe_long': '#7f7f7f',
    'RandomSearch': '#1f77b4',
}


_OPTIMIZER_DISPLAY_NAMES = {
    'GA' : 'GA',
    'RandomSearch' : 'RS',
    'gp_short' : 'GP_short',
    'gp_medium' : 'GP_medium',
    'gp_long' : 'GP_long',
    'tpe_short' : 'TPE_short',
    'tpe_medium' : 'TPE_medium',
    'tpe_long' : 'TPE_long',
}
