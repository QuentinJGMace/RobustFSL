import matplotlib.pyplot as plt
import numpy as np

import pickle
from sklearn.decomposition import PCA


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


def parse_experiment_file(file_path):
    params = []
    scores = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    results = []
    for line in lines:
        if "Running experiment with params" in line:
            results


def load_pca(path_pca: str):
    """Load a pca object from a file

    Args:
        path_pca (str): path to the pca object

    Returns:
        PCA: pca object
    """
    try:
        with open(path_pca, "rb") as f:
            pca = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"PCA object not found at {path_pca}")
    return pca


def plot_pca(
    pca: PCA,
    array: np.array,
    labels: np.array,
    save_path="random/pca/pca_support.png",
    plot=None,
    return_fig=False,
    markersize=1,
):
    """Plot the pca of the points in the array, coloraed by class

    Args:
        pca (PCA): pca object
        array (np.array): array of points to plot
        labels (np.array): array of labels
        save_path (str): path to save the figure

    Returns:
        fig, ax: if return_fig is True
    """

    if plot is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig, ax = plot
    if labels is not None:
        ax.scatter(
            pca.transform(array)[:, 0],
            pca.transform(array)[:, 1],
            c=labels,
            s=markersize,
        )
    else:
        ax.scatter(pca.transform(array)[:, 0], pca.transform(array)[:, 1], c="r", s=50)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if save_path is not None:
        fig.savefig(save_path)

    if return_fig:
        return fig, ax


def plot_outlier_detection(
    thetas,
    n_support,
    idx_outliers_support,
    idx_outliers_query,
    save_path=None,
    plot=None,
    return_fig=False,
):
    """Plot the outlier detection results

    Args:
        thetas (np.array): array of thetas
        indices_outliers (np.array): array of indices of outliers
        save_path (str): path to save the figure

    Returns:
        fig, ax: if return_fig is True
    """
    if plot is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    else:
        fig, ax = plot

    ax[0].plot(thetas[0, :n_support].cpu().numpy())
    ax[0].scatter(
        idx_outliers_support[0].cpu().numpy(),
        thetas[0, idx_outliers_support[0]].cpu().numpy(),
        c="r",
    )
    ax[0].set_title("Support set")
    ax[1].plot(thetas[0, n_support:].cpu().numpy())
    ax[1].scatter(
        idx_outliers_query[0].cpu().numpy(),
        thetas[0, idx_outliers_query[0] + n_support].cpu().numpy(),
        c="r",
    )
    ax[1].set_title("Query set")

    if save_path is not None:
        fig.savefig(save_path)

    if return_fig:
        return fig, ax


def plot_outlier_detection_per_class(
    thetas,
    n_support,
    idx_outliers_support,
    idx_outliers_query,
    save_path=None,
    plot=None,
    return_fig=False,
):

    n_class = thetas.size(2)

    if plot is None:
        fig, ax = plt.subplots(n_class, 2, figsize=(10, n_class * 5))
    else:
        fig, ax = plot

    for i in range(n_class):

        ax[i, 0].plot(thetas[0, :n_support, i].cpu().numpy())
        ax[i, 0].scatter(
            idx_outliers_support[0].cpu().numpy(),
            thetas[0, idx_outliers_support[0], i].cpu().numpy(),
            c="r",
        )
        ax[i, 0].set_title(f"Support set class {i}")
        ax[i, 1].plot(thetas[0, n_support:, i].cpu().numpy())
        ax[i, 1].scatter(
            idx_outliers_query[0].cpu().numpy(),
            thetas[0, idx_outliers_query[0] + n_support, i].cpu().numpy(),
            c="r",
        )
        ax[i, 1].set_title(f"Query set class {i}")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)

    if return_fig:
        return fig, ax
