# source. http://katbailey.github.io/post/gaussian-processes-for-dummies/
import numpy as np
import config
import os
import matplotlib.pyplot as plt


def kernel(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)

n = 400

a = np.linspace(-1.5, 1.5, num=n).reshape([-1, 1])
b = np.linspace(-1.5, 1.5, num=n).reshape([-1, 1])
param = 0.1
grid = np.meshgrid(a, b)
grid = np.stack(grid).transpose([1, 2, 0]).reshape([-1, 2])
print(grid.shape)
orig_grid = grid
print("Calculating kernels")
kernel_x = kernel(a, a*0, param)
print("Finished kernel_x")
kernel_y = kernel(b, b*0, param)
print("Finished kernel_x")

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

# Make the plot
fig = plt.figure()
ax = fig.gca(projection='3d')
print(orig_grid.shape)
#print(orig_grid)
ax.plot_trisurf(orig_grid[:, 0], orig_grid[:, 1], (kernel_x * kernel_y.T).ravel(), cmap=plt.cm.jet, linewidth=0.2)#grid[0].ravel(), grid[1].ravel(), K.ravel(), cmap=plt.cm.jet, linewidth=0.2)

ax.view_init(30, -45)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x)')

#ax.set_xticks([-2, -1, 0, ])
#ax.set_yticks([-5, 0, 5])

plt.savefig(os.path.join(config.PLOT_FOLDER, "squared_exp_kernel"))

plt.show()