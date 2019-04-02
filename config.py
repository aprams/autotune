import os

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))

PLOT_FOLDER = os.path.join(ROOT_FOLDER, "visualizations/plots/")
PLOT_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), PLOT_FOLDER)
os.makedirs(PLOT_FOLDER, exist_ok=True)

EXPERIMENT_RESULTS_FOLDER = "experiments/results"
EXPERIMENT_RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), EXPERIMENT_RESULTS_FOLDER)
if not os.path.exists(EXPERIMENT_RESULTS_FOLDER):
    os.makedirs(EXPERIMENT_RESULTS_FOLDER)

HPO_FOLDER = "experiments/hpo_dataset"
HPO_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), HPO_FOLDER)
if not os.path.exists(HPO_FOLDER):
    os.makedirs(HPO_FOLDER)
