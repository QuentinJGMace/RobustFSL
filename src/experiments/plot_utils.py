import matplotlib.pyplot as plt
import numpy as np


def plot_evolution_with_param(param_sets, results, param_name, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    x = np.array([param[param_name] for param in param_sets])
    y = np.array([r["mean_accuracies"] for r in results])
    ax.errorbar(x, y, label=param_name)
    ax.set_xlabel("Number of outliers")
    ax.set_ylabel("Accuracy")
    ax.plot(x, y, "o-")
    ax.legend()

    return fig, ax
