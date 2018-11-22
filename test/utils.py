import matplotlib.pyplot as plt

def save_plotted_progress(data, name):
    plt.plot(data)
    plt.savefig(name)
