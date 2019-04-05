import os
import config
import pickle
from preprocess_hpo_dataset import create_index_param_space
from autotune import param_space


def load_hpo_data():
    # Load preprocessed data dicts
    with open(os.path.join(config.HPO_FOLDER, 'preprocessed_data.pickle'), 'rb') as handle:
        classifier_indexed_params = pickle.load(handle)

    with open(os.path.join(config.HPO_FOLDER, 'preprocessed_param_values.pickle'), 'rb') as handle:
        classifier_param_values = pickle.load(handle)

    classifier_param_spaces = {}
    for k in classifier_indexed_params.keys():
        print(k)
        param_list = create_index_param_space(classifier_param_values[k])
        for p in param_list:
            print(p.name, p.space)
        classifier_param_spaces[k] = param_list

    classifier_combined_spaces = []
    classifier_combined_spaces += [
        param_space.Integer(space=list(range(len(list(classifier_param_spaces.keys())))), name='classifier')]
    list(classifier_param_spaces.keys())
    for c in classifier_param_spaces.keys():
        for p in classifier_param_spaces[c]:
            classifier_combined_spaces += [p]
    classifiers = list(classifier_param_spaces.keys())
    for p in classifier_combined_spaces:
        print(p.name, p.space)

    return classifier_indexed_params, classifier_param_spaces, classifier_combined_spaces, classifiers