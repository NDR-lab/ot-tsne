"""Receiving-degree (hubness) diagnostics for the synthetic notebooks, in the paper style."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ne_aligned_utils.evaluation.hubness import GRAPH_LABELS
from ne_aligned_utils.paper import layout, registry
from ne_aligned_utils.paper.figures.synthetic import GALLERY_BATCH_COLOURS, PANEL_TITLE_PT


def occurrence_histograms(points, *, title=None):
    """Shared linear x and logarithmic y; zero-degree points remain in the histogram."""
    fig, axes = plt.subplots(
        1,
        len(GRAPH_LABELS),
        figsize=layout.full_width(1.9),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    finite = points.normalized_k_occurrence.dropna()
    bins = np.linspace(0, max(2.0, float(finite.max()) * 1.01) if len(finite) else 2.0, 31)
    for ax, (graph, label) in zip(axes, GRAPH_LABELS.items(), strict=True):
        for batch, group in points.loc[points.graph == graph].groupby("receiving_batch"):
            values = group.normalized_k_occurrence.dropna()
            if len(values):
                ax.hist(
                    values,
                    bins=bins,
                    weights=np.full(len(values), 1 / len(values)),
                    histtype="step",
                    color=GALLERY_BATCH_COLOURS[int(batch)],
                    label=f"Batch {batch}",
                    linewidth=1.0,
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No receiving edges",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=PANEL_TITLE_PT,
                    wrap=True,
                )
        ax.axvline(1, color="0.65", linestyle=":", linewidth=0.6)
        ax.set(title=label, xlabel=r"$N_k/\overline{N_k}$", yscale="log")
        ax.legend(fontsize=PANEL_TITLE_PT, frameon=False)
    axes[0].set_ylabel("Fraction of points")
    if title:
        fig.suptitle(title, fontsize=PANEL_TITLE_PT)
    return fig


def receiving_coverage_figure(summary, modes):
    """Separate each receiving batch; SD is across corruption seeds, not points."""
    fig, axes = plt.subplots(
        len(modes),
        2,
        figsize=layout.full_width(2.1 * len(modes)),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for row, mode in enumerate(modes):
        for batch, ax in enumerate(axes[row]):
            selected = summary.loc[(summary["mode"] == mode) & (summary.receiving_batch == batch)]
            for graph, label in GRAPH_LABELS.items():
                group = selected.loc[selected.graph == graph].sort_values("magnitude")
                if group.empty:
                    continue  # Directed graph has exactly one receiving batch.
                style = registry.resolve("ot" if graph == "sparse_ot" else "joint")
                marker = {
                    "knn_0_to_1": ">",
                    "knn_1_to_0": "<",
                    "shared_knn": "s",
                    "mutual_knn": "o",
                    "sparse_ot": style.marker,
                }[graph]
                ax.errorbar(
                    group.magnitude,
                    group.receiving_coverage_mean,
                    yerr=group.receiving_coverage_sd.fillna(0),
                    label=label,
                    color=style.color,
                    marker=marker,
                    linestyle="--" if graph == "mutual_knn" else "-",
                    linewidth=0.9,
                    markersize=3,
                    capsize=1.5,
                )
            ax.set(
                title=f"{mode.replace('_', ' ').capitalize()} · receiving batch {batch}",
                xlabel=r"Batch effect strength, $\lambda$",
                ylim=(-0.03, 1.05),
            )
            ax.legend(fontsize=PANEL_TITLE_PT, frameon=False)
        axes[row, 0].set_ylabel("Receiving coverage")
    return fig
