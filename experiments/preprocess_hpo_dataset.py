import yaml
import math
from autotune import param_space
import pickle


def create_index_param_space(param_value_dict):
    params = []
    for k in param_value_dict.keys():
        tmp_param = param_space.Integer(space=list(range(len(param_value_dict[k]))), name=k)
        params += [tmp_param]

    return params


if __name__ == '__main__':
    # Load configurations
    with open("../hpo_dataset/configurations.yml", 'r') as stream:
        hpo_params = yaml.load(stream)

    # Load and clean results
    with open("../hpo_dataset/results.yml", 'r') as stream:
        hpo_results = yaml.load(stream)
        cleaned_hpo_results = {}
        # clean results as there are nan's and shorter ones
        for k in hpo_results.keys():
            len_ok = len(hpo_results[k]) == len(hpo_params)
            no_nans = True
            for k2 in hpo_results[k]:
                if math.isnan(hpo_results[k][k2]):
                    no_nans = False
                    break

            if len_ok and no_nans:
                #print("Key {0} was ok".format(k))
                cleaned_hpo_results[k] = hpo_results[k]
            else:
                #print("Key {0} was not ok, len_ok: {1}, no_nans: {2}".format(k, len_ok, no_nans))
                pass
        hpo_results = cleaned_hpo_results
        print("HPO results has {0} experiments".format(len(list(hpo_results.keys()))))

    # Prepare dicts
    classifiers = ['libsvm_svc', 'liblinear', 'random_forest']
    classifier_param_values = {}
    classifier_param_spaces = {}

    classifier_dicts = {}
    for x in classifiers:
        classifier_dicts[x] = {}
        classifier_param_values[x] = {}

    for i in range(len(hpo_params)):
        tmp_dict = hpo_params[i]
        tmp_classifier = tmp_dict['classifier']
        del tmp_dict['classifier']

        # insert results into dict
        tmp_results = []
        for j in range(len(list(hpo_results.keys()))):
            result_key = list(hpo_results.keys())[j]
            tmp_results += [hpo_results[result_key][i]]

        # update classifier values
        for key in list(tmp_dict.keys()):
            if key not in classifier_param_values[tmp_classifier]:
                print("Adding param key", key)
                classifier_param_values[tmp_classifier][key] = []
            if tmp_dict[key] not in classifier_param_values[tmp_classifier][key]:
                print("Adding param key {0} for value {1} ".format(key, tmp_dict[key]))
                classifier_param_values[tmp_classifier][key] += [tmp_dict[key]]

        classifier_dicts[tmp_classifier][frozenset(tmp_dict.items())] = tmp_results

    for i in range(len(classifiers)):
        k = classifiers[i]
        print("Classifier {0} has {1} configurations with".format(k, len(classifier_dicts[k].items())))
        print("Classifier {0} params:")
        param_list = create_index_param_space(classifier_param_values[k])
        for p in param_list:
            print(p.name, p.space)

        classifier_param_spaces[k] = param_list

    classifier_indexed_params = {}

    for c in classifiers:
        classifier_indexed_params[c] = {}
        for k, v in classifier_dicts[c].items():
            key_dict = dict(k)
            indexed_dict = {}
            for k2, v2 in key_dict.items():
                indexed_dict[k2] = classifier_param_values[c][k2].index(v2)
            classifier_indexed_params[c][frozenset(indexed_dict.items())] = v

    for c in classifier_indexed_params:
        print("-" * 36)
        print("-" * 36)
        print(c)
        for k, v in classifier_indexed_params[c].items():
            print("Indexed; Key: {0}, Value: {1}".format(k, v))

    with open('../hpo_dataset/preprocessed_data.pickle', 'wb') as handle:
        pickle.dump(classifier_indexed_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../hpo_dataset/preprocessed_param_values.pickle', 'wb') as handle:
        pickle.dump(classifier_param_values, handle, protocol=pickle.HIGHEST_PROTOCOL)