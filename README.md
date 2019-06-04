# autotune

This project provides an implementation and comparison of popular hyperparameter optimization strategies. Currently the following optimizers are supported:
- Grid Search
- Random Search
- Genetic Algorithms
- Gaussian Processes (adapting [BayesianOptimization](https://github.com/fmfn/BayesianOptimization))
- Tree-structured Parzen Estimators (adapting [HyperOpt](https://github.com/hyperopt/hyperopt))

## Structure

- autotune/ - contains the optimizer implementations
- experiments/ - contains a number of experiments on three datasets (call first call the xy_eval.py, then xy_plot.py for any experiment)
- tests/ - basic tests for the optimizers to ensure their functionality and initially recorded performance

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Installing

In order to test and start developing, you will need to install the requirements, preferably in a virtual environment:

```
virtualenv -p python3.6 .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the notebooks by calling
```
jupyter notebook
```
and selecting the notebook.

### Example Code

```
from autotune import param_space
from autotune.optimizers import gp_search

a = param_space.Real([-5, 5], name='real_var', n_points_to_sample=20)
b = param_space.Real([-5, 0], projection_fn=lambda x: 10 ** x, name='real_var_2', n_points_to_sample=20)

def sample_eval_fn(params):
    return params['real_var'] * params['real_var_2']

optimizer = gp_search.GaussianProcessOptimizer([a, b], sample_eval_fn, n_iterations=10, random_seed=0)
results = optimizer.maximize()
print("Best value of {0} achieved with parameters {1}".format(results[0][1], dict(results[0][0]))) # sorted list of parameter combination -> eval_fn output mapping
```

## Built With

* [BayesianOptimization](https://github.com/fmfn/BayesianOptimization) - Framework for Gaussian Processes based optimization
* [HyperOpt](https://github.com/hyperopt/hyperopt/) - Hyperparameter Optimization framework, used for TPEs in this project
* [Numpy](https://pypi.org/project/numpy/) - NumPy is the fundamental package for array computing with Python.
* [Scipy](https://pypi.org/project/scipy/) - Scientific Library for Python
* [Matplotlib](https://pypi.org/project/matplotlib/) - Python plotting package
* [Seaborn](https://pypi.org/project/seaborn/) - Statistic data visualization package


## Authors

* **Alexander Prams** - [aprams](https://github.com/aprams)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* BayesianOptimization and HyperOpt authors for their great frameworks, which made the development of autotune much easier


