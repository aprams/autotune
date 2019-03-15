from test.test_ga_search import test_ga_search
from test.test_gp_search import test_gp_search
from test.test_grid_search import test_grid_search
from test.test_random_search import test_random_search
from test.test_tpe_search import test_tpe_search

from utils import save_plotted_progress


ga_optimizer = test_ga_search(n_iterations=2000)
gs_optimizer = test_grid_search()
rs_optimizer = test_random_search(n_iterations=2000)
gp_optimizer = test_gp_search(n_iterations=250, gp_n_iter=250, gp_n_warmup=100000)
tpe_optimizer = test_tpe_search(n_iterations=1000)

optimizers = [ga_optimizer, gs_optimizer, rs_optimizer, gp_optimizer, tpe_optimizer]
all_cum_max_data = []

for i in range(len(optimizers)):
    o = optimizers[i]
    save_plotted_progress(o)


    cumulative_max_data = [max(o.eval_fn_per_timestep[0:i+1]) for i in range(len(o.eval_fn_per_timestep))]
    all_cum_max_data += [cumulative_max_data]
    save_plotted_progress(o, data=cumulative_max_data, name="cum_max_" + o.name, y_lim=[0, 14])

import matplotlib.pyplot as plt

for x in all_cum_max_data:
    plt.semilogx(x)
plt.legend([o.name for o in optimizers], loc='lower right')
plt.savefig('./plots/cum_max_all_log_x')

plt.clf()

for x in all_cum_max_data:
    plt.plot(x)
plt.legend([o.name for o in optimizers], loc='lower right')
plt.savefig('./plots/cum_max_all')
