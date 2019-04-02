import numpy as np
import os
import config
import matplotlib.pyplot as plt

np.random.seed(2)

x = np.linspace(0, 15, num=100)
f = lambda x: np.sin(x)
y = np.random.normal(f(x), scale=0.35)

gamma = 0.8

quantile_idx = int((1.0-gamma) * len(y))
best_idx = np.argsort(y)[:quantile_idx]
worst_idx = np.argsort(y)[quantile_idx:]

plt.scatter(x[worst_idx], y[worst_idx], label="other observations")
plt.scatter(x[best_idx], y[best_idx], c='r', label="{0}% best samples".format(int(np.round((1.0-gamma) * 100))))
plt.legend(loc="upper right")
plt.xlabel('Hyperparameter value')
plt.ylabel('Function value')
plt.savefig(os.path.join(config.PLOT_FOLDER, "tpe_split"))
plt.show()