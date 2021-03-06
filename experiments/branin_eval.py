import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import multiprocessing as mp
import utils
import random
import config
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle

from utils import gen_example_2d_plot, branin, plot_results
from autotune import param_space
from autotune.optimizers import grid_search, ga_search, gp_search, tpe_search
from autotune.optimizers import random_search



if __name__ == '__main__':
    do_recreate_fun = False
    sample_callback_fn = None

    branin_x = param_space.Real([-5, 10], name='x', n_points_to_sample=200)
    branin_y = param_space.Real([0, 15], name='y', n_points_to_sample=200)

    branin_param_space = [branin_x, branin_y]
    branin_eval_fn = lambda params: -branin()(params)

    branin_samples = []

    for x in branin_x.create_generator()():
        for y in branin_y.create_generator()():
            branin_samples += [(x, y, -branin_eval_fn({'x': x, 'y': y}))]
    branin_samples = np.array(branin_samples)

    if do_recreate_fun:
        import seaborn as sns
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        sns.reset_orig()
        sns.reset_defaults()
        import importlib

        importlib.reload(mpl)
        importlib.reload(plt)
        importlib.reload(sns)

        # Make the plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(branin_samples[:, 0], branin_samples[:, 1], branin_samples[:, 2], cmap=plt.cm.jet, linewidth=0.0)

        ax.view_init(30, -135)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f(X, Y)')

        ax.set_xticks([-5, 0, 5, 10])
        ax.set_yticks([0, 5, 10, 15])

        plt.savefig(os.path.join(config.PLOT_FOLDER, './branin_fun'))
        sys.exit()

    def branin_grid_search(random_seed=None):
        print("-" * 53)
        print("Testing GS")

        optimizer = grid_search.GridSearchOptimizer(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn)
        _ = optimizer.maximize()

        return optimizer


    def branin_ga_search(n_iterations=2000, random_seed=None):
        print("-" * 53)
        print("2D Example GA fn")

        optimizer = ga_search.GeneticAlgorithmOptimizer(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn,
                                                        n_pops=8, n_iterations=n_iterations, elite_pops_fraction=0.2,
                                                        random_seed=random_seed)
        _ = optimizer.maximize()

        return optimizer


    def branin_gp_search(n_iterations=20, gp_n_warmup=100000, gp_n_iter=25, n_restarts_optimizer=5, name='gp', random_seed=None):
        print("-" * 53)
        print("Testing GP")

        gp_params = {"alpha": 1e-5, "n_restarts_optimizer": n_restarts_optimizer}
        optimizer = gp_search.GaussianProcessOptimizer(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn,
                                                       n_iterations=n_iterations, gp_n_warmup=gp_n_warmup,
                                                       gp_n_iter=gp_n_iter, name=name, **gp_params, random_seed=random_seed)
        _ = optimizer.maximize()

        return optimizer


    def branin_random_search(n_iterations=2000, random_seed=None):
        print("-" * 53)
        print("Testing RS")

        optimizer = random_search.RandomSearchOptimizer(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn,
                                                        n_iterations=n_iterations, random_seed=random_seed)
        _ = optimizer.maximize()

        return optimizer


    def branin_tpe_search(n_iterations=2000, n_EI_candidates=24, name='TPE', random_seed=None):
        print("-" * 53)
        print("Testing TPE")

        optimizer = tpe_search.TPEOptimizer(branin_param_space, branin_eval_fn, callback_fn=sample_callback_fn,
                                            n_iterations=n_iterations, n_EI_candidates=n_EI_candidates,
                                            name=name, random_seed=random_seed)
        _ = optimizer.maximize()

        return optimizer


    N_BRANIN_ITERS = 100
    N_ITERS_PER_OPT = 10
    def worker(i):
        optimizers = [
        #branin_grid_search(),
        branin_random_search(n_iterations=N_BRANIN_ITERS, random_seed=i),
        branin_ga_search(n_iterations=N_BRANIN_ITERS, random_seed=i*2),
        branin_gp_search(n_iterations=N_BRANIN_ITERS, random_seed=i*3, gp_n_iter=25, gp_n_warmup=1000, name='gp_short'),
        branin_gp_search(n_iterations=N_BRANIN_ITERS, random_seed=i*4, gp_n_iter=100, gp_n_warmup=10000, name='gp_medium'),
        branin_gp_search(n_iterations=N_BRANIN_ITERS, random_seed=i*5, gp_n_iter=250, gp_n_warmup=100000, name='gp_long'),
        branin_tpe_search(n_iterations=N_BRANIN_ITERS, random_seed=i*6, name='TPE_medium'),
        branin_tpe_search(n_iterations=N_BRANIN_ITERS, random_seed=i*7, n_EI_candidates=5, name='TPE_short'),
        branin_tpe_search(n_iterations=N_BRANIN_ITERS, random_seed=i*8, n_EI_candidates=100, name='TPE_long'),
        ]


        results = {}
        for o in optimizers:
            results[o.name] = list(zip(o.hyperparameter_set_per_timestep, o.eval_fn_per_timestep,
                                       o.cpu_time_per_opt_timestep, o.wall_time_per_opt_timestep))
        return results

    pool = mp.Pool(config.N_MP_PROCESSES)
    results = pool.map(worker, range(N_ITERS_PER_OPT))

    transposed_results = {}
    for opt_name in results[0]:
        transposed_results[opt_name] = [[] for _ in range(N_ITERS_PER_OPT)]
        for i in range(N_ITERS_PER_OPT):
            transposed_results[opt_name][i] = results[i][opt_name]
    results = transposed_results
    eval_fns_per_timestep = utils.results_to_numpy(results, negative=True)

    # results shape : {opt_name}[N_ITERS_PER_OPT][4][..]

    for opt_name in results.keys():
        r = results[opt_name]
        sample_points = np.array([(x[1], y[1]) if x[0] == 'x' else (y[1], x[1]) for (x, y) in np.array(r)[...,0][0]])
        #print(sample_points)
        param_ranges = [[branin_param_space[0].space[0], branin_param_space[0].space[1]],
                        [branin_param_space[1].space[0], branin_param_space[1].space[1]]]
        gen_example_2d_plot(sample_points, branin_eval_fn, param_ranges=param_ranges, name="branin_plot_" + opt_name)


    with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, 'branin_results.pickle'), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, 'branin_eval_fns_per_timestep.pickle'), 'wb') as handle:
        pickle.dump(eval_fns_per_timestep, handle, protocol=pickle.HIGHEST_PROTOCOL)
