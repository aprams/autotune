import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import multiprocessing as mp
import utils
import random
import config
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
from scipy.stats import beta
from functools import reduce
from hyperopt import hp, space_eval
import hyperopt.pyll
from itertools import product

from utils import gen_example_2d_plot, branin, plot_results
from autotune import param_space
from autotune.optimizers import grid_search, ga_search, gp_search, tpe_search
from autotune.optimizers import random_search


if __name__ == '__main__':
    do_recreate_fun = False
    sample_callback_fn = None


    N_BRANIN_ITERS = 100
    N_ITERS_PER_OPT = 10

    def worker(args):#i, depth, width):
        try:
            i, depth, width = args
            print("Worker args; i: {0}, depth: {1}, width: {2}".format(i, depth, width))
            def projection_fn(x, a, b, min_point, **kwargs):# multiplier):
                return math.fabs(beta.ppf(x, a, b) - min_point)


            def gen_structured_space(depth=2, width=2, prefix=""):
                space = []
                tpe_space = {}
                params = {}
                for i in range(width):
                    name = prefix + str(i)
                    if depth != 0:
                        tmp_results = gen_structured_space(depth-1, width, name)
                        space.append(tmp_results[0])
                        params = {**tmp_results[2], **params}
                        tmp_tpe_result = tmp_results[1]
                        tpe_space[name] = tmp_tpe_result
                    else:
                        a=random.uniform(0.1, 5)
                        b=random.uniform(0.1, 5)
                        min_point = random.uniform(0, 1)
                        multiplier = random.uniform(0.5, 2.0)
                        x = param_space.Real([0, 1], projection_fn=None, name=name,
                                                        n_points_to_sample=200)
                        space.append(x)
                        tpe_x = hp.uniform(name, 0, 1)# * int(name[2])#projection_fn(, a, b, min_point)
                        tpe_space[name] = tpe_x
                        params[name]= {'a': a, 'b': b, 'min_point': min_point, 'multiplier': multiplier}
                if depth > 0:
                    tpe_space = hp.choice(prefix, tpe_space.values())
                return space, tpe_space, params

            DEPTH = depth
            WIDTH = width

            structured_space_idx = [param_space.Integer(name='idx_' + str(i),
                                                        space=list(range(WIDTH))) for i in range(DEPTH)]
            structured_space, structured_tpe_space, structure_params = gen_structured_space(depth=DEPTH, width=WIDTH)
            structured_space = list(utils.flatten_list(structured_space)) + structured_space_idx

            def structured_eval_fn(params):
                tmp_dict = params
                is_tpe = not any([x.startswith('idx_') for x in list(params.keys())])
                if not is_tpe:
                    idx = {x: params[x] for x in params.keys() if type(x) == str and x.startswith('idx_')}.items()
                    idx = sorted(idx, key=lambda x: x[0])
                    idx = [x[1] for x in idx]
                    idx = reduce(lambda x, y: str(x) + str(y), idx) if len(idx) > 1 else str(idx[0])
                    tmp_dict = {k: params[k] for k in list(params.keys()) if k.startswith(idx)}
                    #print(idx)
                    #print(tmp_dict)
                projected_params = [(1.0 + projection_fn(v, **structure_params[k])) * structure_params[k]['multiplier']
                                    for k, v in tmp_dict.items()]
                #print(projected_params)
                result = reduce(lambda x, y: x * y, projected_params)
                #print(result)
                assert(result <= 100)
                return - result


            def structured_ga_search(n_iterations=2000, random_seed=None):
                optimizer = ga_search.GeneticAlgorithmOptimizer(structured_space, structured_eval_fn, callback_fn=sample_callback_fn,
                                                                n_pops=8, n_iterations=n_iterations, elite_pops_fraction=0.2,
                                                                random_seed=random_seed)
                _ = optimizer.maximize()

                return optimizer


            def structured_gp_search(n_iterations=20, gp_n_warmup=100000, gp_n_iter=25, n_restarts_optimizer=5, name='gp', random_seed=None):
                gp_params = {"alpha": 1e-5, "n_restarts_optimizer": n_restarts_optimizer}
                optimizer = gp_search.GaussianProcessOptimizer(structured_space, structured_eval_fn, callback_fn=sample_callback_fn,
                                                               n_iterations=n_iterations, gp_n_warmup=gp_n_warmup,
                                                               gp_n_iter=gp_n_iter, name=name, **gp_params, random_seed=random_seed)
                _ = optimizer.maximize()

                return optimizer


            def structured_random_search(n_iterations=2000, random_seed=None):
                optimizer = random_search.RandomSearchOptimizer(structured_space, structured_eval_fn, callback_fn=sample_callback_fn,
                                                                n_iterations=n_iterations, random_seed=random_seed)
                _ = optimizer.maximize()

                return optimizer


            def structured_tpe_search(n_iterations=2000, n_EI_candidates=24, name='TPE', random_seed=None):
                optimizer = tpe_search.TPEOptimizer(structured_tpe_space, structured_eval_fn, callback_fn=sample_callback_fn,
                                                    n_iterations=n_iterations, n_EI_candidates=n_EI_candidates,
                                                    name=name, random_seed=random_seed)
                _ = optimizer.maximize()

                return optimizer


            optimizers = [
            structured_random_search(n_iterations=N_BRANIN_ITERS, random_seed=i),
            structured_ga_search(n_iterations=N_BRANIN_ITERS, random_seed=i*2),
            structured_gp_search(n_iterations=N_BRANIN_ITERS, random_seed=i*3, gp_n_iter=25, gp_n_warmup=10000, name='gp_short'),
            #structured_gp_search(n_iterations=N_BRANIN_ITERS, random_seed=i*4, gp_n_iter=100, gp_n_warmup=100000, name='gp_medium'),
            #structured_gp_search(n_iterations=N_BRANIN_ITERS, random_seed=i*5, gp_n_iter=250, gp_n_warmup=100000),
            #structured_tpe_search(n_iterations=N_BRANIN_ITERS, random_seed=i*6),
            structured_tpe_search(n_iterations=N_BRANIN_ITERS, random_seed=i*7, n_EI_candidates=5, name='TPE_short'),
            #structured_tpe_search(n_iterations=N_BRANIN_ITERS, random_seed=i*8, n_EI_candidates=100, name='TPE_long'),
            ]


            results = {}
            for o in optimizers:
                results[o.name] = list(zip(o.hyperparameter_set_per_timestep, o.eval_fn_per_timestep,
                                           o.cpu_time_per_opt_timestep, o.wall_time_per_opt_timestep))
            results['depth'] = depth
            results['width'] = width
            results['structure_params'] = structure_params
            return results
        except Exception as e:
            print('Exception: %s', e)



    MIN_DEPTH = 0
    MAX_DEPTH = 3

    MIN_WIDTH = 1
    MAX_WIDTH = 3

    pool = mp.Pool(config.N_MP_PROCESSES)
    it_range = product(range(N_ITERS_PER_OPT), range(MIN_DEPTH, MAX_DEPTH+1), range(MIN_WIDTH, MAX_WIDTH+1))
    print("Total combinations: ", N_ITERS_PER_OPT * (MAX_DEPTH - MIN_DEPTH + 1) * (MAX_WIDTH - MIN_WIDTH + 1))
    results = pool.map(worker, it_range)

    w_d_idxd_results = {}
    w_d_idxd_structure_params = {}
    for x in results:
        w = x['width']
        d = x['depth']
        structure_params = x['structure_params']
        print(structure_params)
        tmp_result = dict(x)
        del tmp_result['width']
        del tmp_result['depth']
        del tmp_result['structure_params']
        k = frozenset({'width': w, 'depth': d}.items())
        if k not in w_d_idxd_results:
            w_d_idxd_results[k] = []
            w_d_idxd_structure_params[k] = []
        w_d_idxd_results[k].append(tmp_result)
        w_d_idxd_structure_params[k].append(structure_params)

    for k, r in w_d_idxd_results.items():
        transposed_results = {}
        for opt_name in r[0]:
            transposed_results[opt_name] = [[] for _ in range(N_ITERS_PER_OPT)]
            for i in range(N_ITERS_PER_OPT):
                transposed_results[opt_name][i] = r[i][opt_name]
        tmp_r = transposed_results
        eval_fns_per_timestep = utils.results_to_numpy(tmp_r, negative=True)
        w_d_idxd_results[k] = eval_fns_per_timestep

    with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, 'structure_eval_fns_per_timestep.pickle'), 'wb') as handle:
        pickle.dump(w_d_idxd_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(config.EXPERIMENT_RESULTS_FOLDER, 'structure_eval_structure_params.pickle'), 'wb') as handle:
        pickle.dump(w_d_idxd_structure_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
