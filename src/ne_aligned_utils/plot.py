import numpy as np
import matplotlib.pyplot as plt

def plot_aligned_embedding(embd, n0, fig_name=None, color=None, shuffle=True, seed=0, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
        return_fig = True
    else:
        return_fig = False

    colors = np.array(["blue", "red"])

    np.random.seed(seed)
    if shuffle:
        perm = np.random.permutation(embd.shape[0])
        dataset_mask = np.array([0]*n0 + [1]*(embd.shape[0]-n0))[perm]
        embd_plot = embd[perm]
    else:
        embd_plot = embd

    plot_color = color if color is not None else colors[dataset_mask]

    ax.scatter(*embd_plot.T,
               s=5.0,
               color=plot_color,
               linewidth=1,
               alpha=1,
               zorder=10)
    ax.set_aspect("equal")
    ax.axis("off")
    if return_fig:
        return fig, ax
    else:
        return ax