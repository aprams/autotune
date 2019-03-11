import os

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))

PLOT_FOLDER = os.path.join(ROOT_FOLDER, "plots")

EXPERIMENT_RESULTS_FOLDER = "experiment_results"
EXPERIMENT_RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), EXPERIMENT_RESULTS_FOLDER)
if not os.path.exists(EXPERIMENT_RESULTS_FOLDER):
    os.makedirs(EXPERIMENT_RESULTS_FOLDER)