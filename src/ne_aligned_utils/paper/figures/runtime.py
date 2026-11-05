"""Runtime figures: all methods per dataset, and OT-tSNE's stage breakdown."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ne_aligned_utils.paper import layout
from ne_aligned_utils.paper.rendering import runtime as presentation, runtime_drawing
from ne_aligned_utils.paper.rendering.runtime import RuntimeFigureStyle, RuntimeRenderContext
from ne_aligned_utils.paper.summaries import runtime as summaries


def ot_breakdown(
    table: pd.DataFrame,
    *,
    method_label: str,
    segments: Sequence[tuple[str, str, str]],
    figsize: tuple[float, float],
) -> Figure:
    """Absolute seconds (left) and stage shares (right) from one breakdown table."""
    fig, (ax_abs, ax_share) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.72, 1.0]}, layout="none"
    )

    y = np.arange(len(table))
    totals = table["time_total"].to_numpy(dtype=float)

    left_abs = np.zeros(len(table))
    left_share = np.zeros(len(table))
    for field, label, colour in segments:
        widths = table[field].to_numpy(dtype=float)
        ax_abs.barh(y, widths, left=left_abs, height=0.62, color=colour, label=label, linewidth=0.0)
        shares = 100.0 * widths / totals
        ax_share.barh(y, shares, left=left_share, height=0.62, color=colour, linewidth=0.0)
        left_abs = left_abs + widths
        left_share = left_share + shares

    pad = float(left_abs.max()) * 0.015
    for index, total in enumerate(totals):
        ax_abs.text(
            float(total) + pad,
            index,
            f"{float(total):,.0f} s",
            va="center",
            ha="left",
            fontsize=7.0,
        )
    # Label the OT phase share directly on its segment.
    for index, row in enumerate(table.to_dict("records")):
        share = 100.0 * float(row["time_pre_embedding"]) / float(row["time_total"])
        ax_share.text(
            share / 2.0,
            index,
            f"{share:.0f}%",
            va="center",
            ha="center",
            fontsize=7.0,
            color="#ffffff",
        )

    labels = [
        f"{row['dataset_label']}\n$n$ = {int(row['n_obs']):,}" for row in table.to_dict("records")
    ]
    for ax in (ax_abs, ax_share):
        ax.set_yticks(y)
        ax.set_ylim(-0.6, len(table) - 0.4)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0.0, pad=2.0)
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.5)

    ax_abs.set_yticklabels(labels, fontsize=7.5)
    ax_abs.set_xlim(0.0, float(left_abs.max()) * 1.19)
    ax_abs.set_xlabel("Runtime (s)")

    ax_share.set_yticklabels([])
    ax_share.set_xlim(0.0, 100.0)
    ax_share.set_xticks([0, 50, 100])
    ax_share.set_xticklabels(["0", "50", "100%"])
    ax_share.set_xlabel("Share of total")

    handles, legend_labels = ax_abs.get_legend_handles_labels()
    # Place the key in the unused upper-right corner of the absolute-duration panel.
    ax_abs.legend(
        handles,
        legend_labels,
        loc="upper right",
        ncol=1,
        frameon=False,
        fontsize=7.5,
        handlelength=1.0,
        handletextpad=0.4,
        labelspacing=0.3,
        borderaxespad=0.5,
    )
    fig.suptitle(
        f"{method_label}: optimal transport and downstream t-SNE",
        fontsize=8.0,
        x=0.012,
        y=0.985,
        ha="left",
        va="top",
    )
    # Explicit margins keep the heading above the legend and leave room for
    # two-line dataset labels; they must not compete with constrained layout.
    fig.subplots_adjust(left=0.18, right=0.965, top=0.905, bottom=0.175, wspace=0.10)

    return fig


def all_methods(
    medians: pd.DataFrame,
    *,
    render_context: RuntimeRenderContext,
    style: RuntimeFigureStyle,
    method_order: Mapping[str, Sequence[str]],
    datasets: tuple[str, ...],
    sizes: Mapping[str, int],
    xlim: tuple[float, float],
    figsize: tuple[float, float],
    device_marks: tuple[str, str],
    gpu_medians: pd.DataFrame | None = None,
) -> Figure:
    """Per-dataset median runtimes, one panel per method family, rows as given.

    GPU readings, when supplied, share the dataset glyph and row of their CPU reading.
    """
    ordered = datasets
    offsets = runtime_drawing.dataset_offsets(ordered, row_spread=style.row_spread)
    labels = [
        runtime_drawing.dataset_label(
            dataset, sizes.get(dataset), labels=render_context.dataset_labels
        )
        for dataset in ordered
    ]
    ncol, dataset_fraction, legend_fraction = runtime_drawing.legend_geometry(
        labels, figsize, style=style
    )

    families = render_context.families
    family_order = render_context.family_order
    markers = render_context.dataset_markers
    counts = [len(families[family]) for family in family_order]
    fig, axes = runtime_drawing.family_axes(counts, legend_fraction, figsize)

    lookup = {
        (str(row["method_key"]), str(row["dataset"])): row for row in medians.to_dict("records")
    }
    gpu_lookup: dict[tuple[str, str], dict] = {}
    if gpu_medians is not None and len(gpu_medians):
        gpu_lookup = {
            (str(row["method_key"]), str(row["dataset"])): row
            for row in gpu_medians.to_dict("records")
        }

    for ax, family in zip(axes, family_order):
        keys = list(method_order[family])
        show_phases = family in style.phase_families
        for row_index, key in enumerate(keys):
            y = len(keys) - 1 - row_index
            for dataset in ordered:
                record = lookup.get((key, dataset))
                if record is None:
                    continue
                colour = style.palette[dataset]
                marker = markers[dataset]
                y_point = y + offsets[dataset]
                total = float(record["time_total"])
                pre = float(record["time_pre_embedding"])
                if (
                    show_phases
                    and np.isfinite(pre)
                    and np.isfinite(float(record["time_embedding"]))
                ):
                    ax.plot(
                        [pre, total],
                        [y_point, y_point],
                        color=colour,
                        linewidth=0.5,
                        zorder=1,
                        solid_capstyle="butt",
                    )
                    ax.scatter(
                        [pre],
                        [y_point],
                        marker=marker,
                        s=style.hollow_size,
                        facecolors="none",
                        edgecolors=colour,
                        linewidths=0.6,
                        zorder=2,
                    )
                ax.scatter(
                    [total],
                    [y_point],
                    marker=marker,
                    s=style.point_size,
                    color=colour,
                    linewidths=0.0,
                    zorder=3,
                )
                gpu = gpu_lookup.get((key, dataset))
                if gpu is None:
                    continue
                g_total = float(gpu["time_total"])
                # Draw GPU totals only; short phase intervals would overlap at this scale.
                ax.scatter(
                    [g_total],
                    [y_point],
                    marker=marker,
                    s=style.point_size,
                    color=colour,
                    linewidths=0.0,
                    zorder=3,
                )
                gpu_mark = runtime_drawing.device_mark(
                    ax, g_total, y_point, device_marks[1], colour, style=style
                )
                runtime_drawing.device_mark(
                    ax, total, y_point, device_marks[0], colour, style=style
                )
                # Draw the bridge across the gap, avoiding the CPU phase pair's line and markers.
                ax.plot(
                    [
                        runtime_drawing.right_of(fig, ax, gpu_mark),
                        pre if np.isfinite(pre) else total,
                    ],
                    [y_point, y_point],
                    color=colour,
                    linewidth=0.5,
                    linestyle=style.device_bridge_dash,
                    zorder=1,
                    solid_capstyle="butt",
                )
        runtime_drawing.style_family_axis(
            ax,
            keys,
            xlim,
            style.family_labels[family],
            bottom=family == family_order[-1],
            method_labels=render_context.method_labels,
            style=style,
        )

    axes[-1].set_xlabel("Runtime to final 2D embeddings (s)")
    runtime_drawing.draw_figure_one_legends(
        fig,
        axes,
        ordered,
        labels=labels,
        markers=markers,
        style=style,
        ncol=ncol,
        drop=dataset_fraction,
        device_key=runtime_drawing.device_key_text(device_marks, device_names=style.device_names)
        if gpu_lookup
        else "",
    )
    return fig


def captions(frame: pd.DataFrame, *, context: RuntimeRenderContext) -> dict[str, str]:
    """Draft captions whose facts are read from the plotted rows, so they cannot go stale."""
    plotted = frame.loc[frame["included_in_plot"]]
    first = plotted.iloc[0]
    allocation = (
        f"{int(first['threads_requested'])} threads, {first['device'].upper()}, "
        f"{first['blas_name']}"
    )
    gene_space = sorted(
        {
            context.method_labels[k]
            for k in plotted.loc[plotted["n_features_input"] > 50, "method_key"]
        }
    )
    widths = sorted({int(w) for w in plotted["n_features_input"]})
    pre, emb, total = (
        plotted[c].to_numpy(dtype=float)
        for c in ("time_pre_embedding", "time_embedding", "time_total")
    )
    both = np.isfinite(pre) & np.isfinite(emb)
    worst = (
        float(np.max(np.abs(pre[both] + emb[both] - total[both]) / total[both]))
        if both.any()
        else 0.0
    )
    child = ""
    overhead = pd.Series(dtype=float)
    if {"time_excluded_overhead", "time_wall_outer"} <= set(plotted.columns):
        overhead = plotted.groupby("method_key")["time_excluded_overhead"].median().dropna()
    if "multimap" in overhead.index:
        wall = plotted.loc[plotted["method_key"] == "multimap", "time_wall_outer"].median()
        child = (
            f" For MultiMAP this excluded start-up is {overhead['multimap']:.0f} s of a "
            f"{wall:.0f} s median wall time."
        )
    n_datasets = plotted["dataset"].nunique()
    return {
        "runtime_all_methods": (
            "Runtime to a final two-dimensional embedding, for every benchmarked method on each of "
            f"the {n_datasets} datasets. One coloured marker per dataset, at the median of three "
            "seeds. Rows are ordered by geometric-mean runtime, a sorting statistic that is not "
            "drawn. The x axis is logarithmic and identical in every panel. All points are "
            "`time_total`, the algorithm runtime: it excludes data loading, artifact writing and "
            "scoring, and for methods that run in a child process it also excludes interpreter "
            f"start-up and library import.{child} In the first and third panels the hollow marker "
            "is the integration stage and the filled marker adds the downstream t-SNE; both stages "
            f"are measured and add up to the total within {worst:.1%}. All runs used one "
            f"allocation ({allocation}). OT methods are shown at their reported regularisation. "
            f"Input widths differ by method ({' or '.join(map(str, widths))} features): "
            f"{', '.join(gene_space)} read selected genes, the others 50 principal components."
        ),
        "runtime_ot_tsne_breakdown": (
            "OT-tSNE runtime split between the optimal-transport stage and the downstream t-SNE. "
            "Left: absolute seconds, datasets ordered by number of cells, total annotated. Right: "
            "the same two stages as shares of each dataset's total. Each segment is the median "
            "over three seeds; per-seed values are in `runtime_ot_breakdown.csv`."
        ),
    }


def write_runtime_figures(
    frame: pd.DataFrame,
    *,
    root: Path,
    output_dir: Path,
    datasets: tuple[str, ...],
    gpu_medians: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write the runtime figures and their audit tables from preloaded timing rows."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    context = presentation.runtime_render_context()
    style = presentation.runtime_figure_style()
    frame = summaries.annotate(frame, context=context)
    report = summaries.validate(frame, root=Path(root), datasets=datasets, context=context)
    gpu = gpu_medians if gpu_medians is not None and len(gpu_medians) else None
    inputs = summaries.panel_inputs(frame, datasets=datasets, context=context)

    def save(name, figure):
        return layout.save_figure(
            figure, name, subdir=target.name, root=target.parent, formats=("png",)
        )

    written: dict[str, Path] = {}
    with layout.paper_style():
        written["runtime_all_methods"] = save(
            "runtime_all_methods",
            all_methods(
                **inputs,
                render_context=context,
                style=style,
                device_marks=presentation.DEVICE_MARKS_LETTERS,
                figsize=presentation.FIGURE_ONE_SIZE,
            ),
        )
        if gpu is not None:
            for name, (marks, figsize) in presentation.DEVICE_VARIANTS.items():
                written[name] = save(
                    name,
                    all_methods(
                        **inputs,
                        render_context=context,
                        style=style,
                        gpu_medians=gpu,
                        device_marks=marks,
                        figsize=figsize,
                    ),
                )
        table = summaries.ot_breakdown_table(frame, datasets=datasets, context=context)
        written["runtime_ot_tsne_breakdown"] = save(
            "runtime_ot_tsne_breakdown",
            ot_breakdown(
                table,
                method_label=context.method_labels["ot"],
                segments=presentation.OT_SEGMENTS,
                figsize=presentation.FIGURE_TWO_SIZE,
            ),
        )

    summary = frame.copy()
    hosts = summaries.environment_hosts(Path(root))
    summary["hostname"] = summary["env_fingerprint"].astype(str).map(hosts).fillna("")
    for column in summaries.SUMMARY_COLUMNS:
        if column not in summary.columns:
            summary[column] = np.nan
    tables = {
        "selected-timing-rows.csv": frame,
        "runtime_summary.csv": summary[list(summaries.SUMMARY_COLUMNS)],
        "runtime_ot_breakdown.csv": table,
        "dataset-medians.csv": inputs["medians"],
    }
    for name, data in tables.items():
        data.to_csv(target / name, index=False)
        written[name] = target / name
    (target / "runtime_validation.txt").write_text("\n".join(report) + "\n")
    (target / "runtime_captions.md").write_text(
        "\n\n".join(
            f"## {name}\n\n{text}"
            for name, text in sorted(captions(frame, context=context).items())
        )
        + "\n"
    )
    return written
