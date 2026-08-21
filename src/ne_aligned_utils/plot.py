import os

import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

# plotting utils

def plot_aligned_embedding(embd, n0, s=5, color=None, shuffle=True, seed=0, ax=None):
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
               s=s,
               c=plot_color,
               linewidth=1,
               alpha=1,
               zorder=10)
    ax.set_aspect("equal")
    ax.axis("off")
    if return_fig:
        return fig, ax
    else:
        return ax


# inspection plots for the load_*.ipynb notebooks


def cmap_colors(n):
    """`n` distinct colours -- an atlas has more cell types than `tab10` has colours."""
    if n <= 10:
        return np.asarray(plt.get_cmap("tab10").colors[:n])
    if n <= 20:
        return np.asarray(plt.get_cmap("tab20").colors[:n])
    return plt.get_cmap("gist_ncar")(np.linspace(0.0, 0.95, n))


def plot_panels(
    embds, out, codes=None, names=None, title="", fname=None, fig_dir=None, legend_title=""
):
    """3-panel scatter of `embds` = (batch 1, batch 2, joint), as returned by `prepare`.

    `codes=(codes1, codes2)` together with `names` colours by a label, `codes=None` by
    batch. Saved to `fig_dir/fname` if both are given.
    """
    e1, e2, ej = embds
    b1, b2 = out["batch_names"]

    if codes is None:
        c1, c2 = "r", "b"
        cj = np.array(["r"] * e1.shape[0] + ["b"] * e2.shape[0])
        handles = None
    else:
        colors = cmap_colors(len(names))
        c1, c2 = colors[codes[0]], colors[codes[1]]
        cj = np.concatenate((c1, c2))
        handles = [
            mlines.Line2D(
                [], [], linestyle="none", marker="o", markersize=5,
                markerfacecolor=colors[i], markeredgecolor="none", label=str(lbl),
            )
            for i, lbl in enumerate(names)
        ]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(*e1[:, :2].T, s=1, c=c1)
    ax[0].set_title(str(b1))
    ax[1].scatter(*e2[:, :2].T, s=1, c=c2)
    ax[1].set_title(str(b2))

    order = np.random.permutation(ej.shape[0])
    ax[2].scatter(*ej[order, :2].T, s=1, c=cj[order])
    ax[2].set_title("joint")

    for a in ax:
        a.set_aspect("equal")
        a.axis("off")
    if handles is not None:
        ax[2].legend(
            handles=handles, title=legend_title, loc=(1.0, 0.0), fontsize=7,
            ncol=1 if len(handles) <= 20 else 2,
        )

    fig.suptitle(title)
    if fname is not None and fig_dir is not None:
        fig.savefig(os.path.join(fig_dir, fname), dpi=300, bbox_inches="tight")
    return fig


def plot_variant(out, fig_dir=None, tsne=True, seed=0):
    """PCA50 (and t-SNE on it) panels of a `prepare` result.

    Coloured by batch, by cell type and by every column in `out["extra"]`. The embeddings
    are added to `out` under `"pca2"` / `"tsne"`.
    """
    # local imports so that importing this module stays cheap
    from openTSNE import TSNE
    from sklearn.decomposition import PCA

    np.random.seed(seed)
    dataset = out["dataset"]
    X1, X2 = out["X1"], out["X2"]

    n_comp = min(50, X1.shape[1], X1.shape[0], X2.shape[0])
    # the joint panel reuses what `prepare` saved, so it matches X1_pca<n>/X2_pca<n>
    if out.get("X1_pca") is not None and out.get("X2_pca") is not None:
        joint = np.concatenate((out["X1_pca"], out["X2_pca"]))
    else:
        joint = PCA(n_components=n_comp).fit_transform(np.concatenate((X1, X2)))
    pcas = (
        PCA(n_components=n_comp).fit_transform(X1),
        PCA(n_components=n_comp).fit_transform(X2),
        joint,
    )

    embeddings = [("pca2", pcas, f"PCA2 of {dataset}")]
    if tsne:
        tsnes = tuple(np.asarray(TSNE(verbose=True, random_state=seed).fit(p)) for p in pcas)
        embeddings.append(("tsne", tsnes, f"tSNE on PCA{n_comp} of {dataset}"))

    for tag, embds, title in embeddings:
        plot_panels(embds, out, title=title, fname=f"{dataset}_{tag}.png", fig_dir=fig_dir)
        plot_panels(
            embds, out, codes=(out["y1"], out["y2"]), names=out["unique_cell_types"],
            title=title, fname=f"{dataset}_by_type_{tag}.png", fig_dir=fig_dir,
            legend_title="Cell types",
        )
        for col, lab in out.get("extra", {}).items():
            plot_panels(
                embds, out, codes=(lab["codes1"], lab["codes2"]), names=lab["names"],
                title=title, fname=f"{dataset}_by_{col}_{tag}.png", fig_dir=fig_dir,
                legend_title=col,
            )
        out[tag] = embds

    return out