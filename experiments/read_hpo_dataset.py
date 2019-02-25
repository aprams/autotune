import yaml
import param_space


def create_index_param_space(param_value_dict):
    params = []
    for k in param_value_dict.keys():
        tmp_param = param_space.Integer(space=list(range(len(param_value_dict[k]))), name=k)
        params += [tmp_param]

    return params

with open("../hpo_dataset/configurations.yml", 'r') as stream:
    hpo_params = yaml.load(stream)

with open("../hpo_dataset/results.yml", 'r') as stream:
    hpo_results = yaml.load(stream)
    print(hpo_results.keys())
    print("HPO results has {0} experiments".format(len(list(hpo_results.keys()))))

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
            print("Adding param key ", key)
            classifier_param_values[tmp_classifier][key] = []
        if tmp_dict[key] not in classifier_param_values[tmp_classifier][key]:
            print("Adding param value {0} for key {1} ".format(tmp_dict[key], key))
            classifier_param_values[tmp_classifier][key] += [tmp_dict[key]]


    classifier_dicts[tmp_classifier][frozenset(tmp_dict.items())] = tmp_results

print(classifier_dicts)
print(classifier_param_values)

for i in range(len(classifiers)):
    k = classifiers[i]
    print("Classifier {0} has {1} configurations with".format(k, len(classifier_dicts[k].items())))
    print("Classifier {0} params:")
    param_list = create_index_param_space(classifier_param_values[k])
    for p in param_list:
        print(p.name, p.space)

    classifier_param_spaces[k] = param_list

