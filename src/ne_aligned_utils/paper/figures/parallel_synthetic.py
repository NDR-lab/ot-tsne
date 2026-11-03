"""Shared, saved-artifact figures for both modes of the parallel-PC1 experiment."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ne_aligned_utils.experiments.parallel import load_parallel_run
from ne_aligned_utils.experiments.synthetic import embedding_method_id
from ne_aligned_utils.paper import layout, registry
from ne_aligned_utils.paper.figures import synthetic as drawing
from ne_aligned_utils.paper.inputs.synthetic import load_coordinates


@dataclass(frozen=True)
class ParallelPlotConfig:
    magnitude: float = 10.0
    corruption_seed: int = 0
    embedding_seed: int = 0
    rel_reg: float = 1.0


def policy_figure(cost, modes, rel_reg):
    fig, axes = plt.subplots(
        len(modes), 2, figsize=layout.full_width(4.0), layout="constrained", squeeze=False
    )
    colors = drawing.GALLERY_BATCH_COLOURS
    for row, mode in enumerate(modes):
        for policy, style, color, label in zip(
            ("fixed_at_lambda_0", "adaptive_per_magnitude"),
            ("--", "-"),
            colors,
            ("Fixed ε", "Adaptive ε"),
        ):
            part = cost.loc[
                (cost["mode"] == mode)
                & (cost.epsilon_policy == policy)
                & np.isclose(cost.rel_reg, rel_reg)
            ]
            if part.empty:
                raise ValueError(f"Missing {mode}/{policy}/rel_reg={rel_reg} diagnostics")
            for ax, metric in zip(axes[row], ("epsilon_over_reference", "relative_ot_plan_change")):
                table = drawing.mean_sd(part, metric)
                dashed = style == "--"
                ax.errorbar(
                    table.magnitude,
                    table["mean"],
                    yerr=table.sd.fillna(0),
                    label=label,
                    color=color,
                    marker="o",
                    markersize=3,
                    markerfacecolor="white" if dashed else color,
                    markeredgewidth=0.8,
                    linewidth=1.1,
                    linestyle=(0, (4, 2.5)) if dashed else "-",
                    capsize=1.5,
                )
        axes[row, 0].set(
            title=f"{mode.capitalize()} · regularization", ylabel=r"$\epsilon/\epsilon_0$"
        )
        axes[row, 1].set(
            title=f"{mode.capitalize()} · plan change", ylabel="Relative OT-plan change"
        )
        axes[row, 0].legend(fontsize=drawing.PANEL_TITLE_PT, handlelength=4.0)
        for ax in axes[row]:
            ax.set_xlabel(r"Batch effect strength, $\lambda$")
    return fig


def affinity_figure(affinity, knn, modes, rel_reg):
    fig, axes = plt.subplots(
        len(modes), 2, figsize=layout.full_width(4.0), layout="constrained", squeeze=False
    )
    ot = registry.resolve("ot")
    dark, light = registry.resolve("joint-pca"), registry.resolve("joint")
    specs = (
        (
            "eot_shared_or_digit_purity",
            "eot_shared_or_nonempty_row_fraction",
            "Sparse OT",
            ot.color,
            ot.marker,
        ),
        (
            "shared_correct_class_neighbor_fraction",
            "shared_nonempty_row_fraction",
            "Shared kNN",
            dark.color,
            "o",
        ),
        (
            "mutual_correct_class_neighbor_fraction",
            "mutual_nonempty_row_fraction",
            "Mutual kNN",
            light.color,
            "o",
        ),
    )
    for row, mode in enumerate(modes):
        directed = affinity.loc[
            (affinity["mode"] == mode)
            & (affinity.epsilon_policy == "adaptive_per_magnitude")
            & np.isclose(affinity.rel_reg, rel_reg)
        ]
        for purity, coverage, label, color, marker in specs:
            frame = directed if label == "Sparse OT" else knn.loc[knn["mode"] == mode]
            for ax, field in zip(axes[row], (purity, coverage)):
                drawing.add_curve(ax, frame, field, label=label, color=color, marker=marker)
        for ax in axes[row]:
            ax.set(xlabel=r"Batch effect strength, $\lambda$", ylim=(-0.03, 1.05))
        axes[row, 0].set(title=f"{mode.capitalize()} · graph purity", ylabel="Neighbor purity")
        axes[row, 1].set(title=f"{mode.capitalize()} · coverage", ylabel="Nonempty row fraction")
        axes[row, 0].legend(fontsize=drawing.PANEL_TITLE_PT, handlelength=4.0)
    return fig


def metrics_figure(summary, modes, rel_reg):
    """Compare both epsilon policies and both MultiMAP preprocessing variants in each mode."""
    fig, axes = plt.subplots(
        len(modes), 2, figsize=layout.full_width(4.5), layout="constrained", squeeze=False
    )
    for row, mode in enumerate(modes):
        selected = summary.loc[
            (summary["mode"] == mode)
            & ((summary.method != "ot") | np.isclose(summary.rel_reg, rel_reg))
        ]
        for _, group in selected.groupby("method_id", sort=False):
            sample = group.iloc[0]
            if sample.method == "ot":
                key = "ot"
                linestyle = "--" if sample.epsilon_policy == "fixed_at_lambda_0" else "-"
            elif sample.method == "multimap":
                key = "multimap-k15" if sample.effective_k == 15 else "multimap"
                linestyle = "--" if sample.preprocessing == "uncentered" else "-"
            else:
                key = "scanorama-50"
                linestyle = "--" if sample.preprocessing == "uncentered" else "-"
            style = registry.resolve(key)
            group = group.sort_values("magnitude")
            legend_label = sample.display_method.replace(f" · rel_reg={rel_reg:g}", "")
            if sample.method == "multimap":
                name = "MultiMAP k=15" if sample.effective_k == 15 else "MultiMAP"
                legend_label = f"{name} · {sample.preprocessing}"
            for ax, metric in zip(axes[row], ("purity", "kbet")):
                ax.errorbar(
                    group.magnitude,
                    group[metric],
                    yerr=group[f"{metric}_std"].fillna(0),
                    label=legend_label,
                    color=style.color,
                    marker=style.marker,
                    linestyle=(0, (4, 2.5)) if linestyle == "--" else "-",
                    linewidth=1.1,
                    markersize=3,
                    markerfacecolor="white" if linestyle == "--" else style.color,
                    markeredgewidth=0.8,
                    capsize=1.5,
                )
        axes[row, 0].set(title=f"{mode.capitalize()} · embedding purity", ylabel="Neighbor purity")
        axes[row, 1].set(title=f"{mode.capitalize()} · batch mixing", ylabel="kBET")
        axes[row, 1].set_ylim(0, 1.05)
        if not np.isfinite(selected.kbet).any():
            axes[row, 1].text(
                0.5,
                0.5,
                "kBET unavailable\n(insufficient observations per digit)",
                transform=axes[row, 1].transAxes,
                ha="center",
                va="center",
                fontsize=drawing.PANEL_TITLE_PT,
            )
            magnitudes = selected.magnitude.to_numpy()
            axes[row, 1].set_xlim(min(0.0, magnitudes.min()), max(1.0, magnitudes.max()) * 1.05)
        for ax in axes[row]:
            ax.set_xlabel(r"Batch effect strength, $\lambda$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        fontsize=drawing.PANEL_TITLE_PT,
        handlelength=4.0,
    )
    return fig


def embedding_gallery(run_dir, selection):
    """Read saved coordinates, keeping both noise modes and MultiMAP variants visible."""
    import json

    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())["config"]
    with np.load(run_dir / "data/construction.npz", allow_pickle=False) as data:
        labels0, labels1 = data["labels0"], data["labels1"]
    labels = np.r_[labels0, labels1]
    batches = np.r_[np.zeros(len(labels0), int), np.ones(len(labels1), int)]
    ot_k = config["ot"]["ot_effective_k"]
    specs = [
        ("Joint t-SNE", embedding_method_id("joint-tsne", "tsne")),
        (
            "OT-tSNE",
            embedding_method_id("ot", "tsne", "adaptive_per_magnitude", selection.rel_reg, ot_k),
        ),
    ]
    competitors = config["competitors"]
    if competitors["run_multimap"]:
        k = 15 if competitors["multimap_profile"] == "k15" else None
        for preprocessing in competitors["multimap_preprocessing"]:
            backend = (
                "native-defaults" if preprocessing == "centered" else "uncentered-svd-defaults"
            )
            specs.append(
                (
                    f"{'MultiMAP k=15' if k == 15 else 'MultiMAP'}\n{preprocessing}",
                    embedding_method_id("multimap", backend, k=k),
                )
            )
    if competitors["run_scanorama_50_tsne"]:
        specs.append(("Scanorama→t-SNE", embedding_method_id("scanorama-50-tsne", "tsne")))
    modes = config["data"]["modes"]
    fig, axes = plt.subplots(
        2 * len(modes),
        len(specs),
        figsize=layout.full_width(4.6),
        squeeze=False,
        layout="constrained",
    )
    for row, mode in enumerate(modes):
        for col, (name, method_id) in enumerate(specs):
            path = (
                run_dir
                / mode
                / "embeddings"
                / method_id
                / f"lambda_{selection.magnitude:g}"
                / f"corruption_seed_{selection.corruption_seed}"
                / f"embedding_seed_{selection.embedding_seed}.npz"
            )
            coordinates = load_coordinates(path, len(labels))
            for offset, colors in enumerate(
                (np.asarray(drawing.GALLERY_BATCH_COLOURS)[batches], labels)
            ):
                ax = axes[2 * row + offset, col]
                kwargs = {} if offset == 0 else {"cmap": "tab10", "vmin": 0, "vmax": 9}
                ax.scatter(*coordinates.T, c=colors, s=2, linewidths=0, rasterized=True, **kwargs)
                ax.set(xticks=[], yticks=[])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if offset == 0:
                    ax.set_title(name, fontsize=drawing.PANEL_TITLE_PT)
                if col == 0:
                    ax.set_ylabel(f"{mode.capitalize()}\n{'Batch' if offset == 0 else 'Digit'}")
    return fig


def build_parallel_figures(run_dir, *, output_dir=None, selection=None):
    """Canonical comparison builder: saved results only, no method execution."""
    selection = selection or ParallelPlotConfig()
    runs = load_parallel_run(run_dir)
    modes = runs.manifest["config"]["data"]["modes"]
    with layout.paper_style():
        figures = {
            "epsilon_policies": policy_figure(runs.cost, modes, selection.rel_reg),
            "cross_batch_affinities": affinity_figure(
                runs.affinity, runs.knn, modes, selection.rel_reg
            ),
            "embeddings": embedding_gallery(run_dir, selection),
            "embedding_metrics": metrics_figure(runs.summary, modes, selection.rel_reg),
        }
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for name, fig in figures.items():
                layout.save_figure(
                    fig, f"parallel_pc1_{name}", root=output_dir, preview=False, close=False
                )
    return figures
