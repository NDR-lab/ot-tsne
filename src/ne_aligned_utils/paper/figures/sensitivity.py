"""Sensitivity trajectories: a bio metric against a mixing metric along one swept setting.

One panel per dataset. Each trajectory is one run configuration (see
:mod:`ne_aligned_utils.paper.summaries.sensitivity`), drawn through its seed means in the order
of the swept value. An arrowhead on every edge points the way the value grows, and marker area
grows with it too. The reported configuration is redrawn in red.

Colours and markers come from the paper registry. OT-tSNE variants keep their family colours and
are told apart by line style. Several trajectories of one method (for example one per k) share its
marker and get a light-to-dark ramp ending at its colour (for OT-tSNE, the
registry's OT colour scale).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.lines import Line2D

from ne_aligned_utils.paper import layout, registry
from ne_aligned_utils.paper.rendering.metrics import round_three_ticks
from ne_aligned_utils.paper.summaries.sensitivity import LINE

#: Axis pairs: name -> (x metric, y metric, x label, y label). Mixing on x, bio on y.
PAIRS: dict[str, tuple[str, str, str, str]] = {
    "purity-kbet": ("kbet", "purity", "kBET", "Neighbor purity"),
    "silhouette-ilisi": ("ilisi", "silhouette", "iLISI", "Silhouette"),
    "bio-mixing": ("mixing", "bio", "Batch mixing (iLISI, kBET)", "Bio conservation"),
}

#: Axis names of the settings a trajectory can sweep.
SWEEP_LABELS: dict[str, str] = {
    "rel_reg": "r",
    "effective_k": "k",
    "rel_reg_m": "marginal relaxation",
}

#: Line style per OT-tSNE variant; other methods draw solid.
DASHES: dict[str, object] = {
    "ot": "-",
    "ot-knndiag": "--",
    "ot-gramdiag": "-.",
    "ot-selfotmedian": ":",
    "ot-zerodiag": (0, (5, 1.5, 1, 1.5, 1, 1.5)),
    "ot-zerodiag-split": (0, (1, 1)),
}

#: The reported configuration's glyph.
REPORTED_COLOR = "#E31A1C"
#: Marker areas (pt^2) for the smallest and largest swept value. 20.25 is the 4.5 pt print floor.
SIZE_RANGE = (20.5, 30.0)
MARKER_ALPHA = 0.85
LINE_WIDTH = 0.7
#: Lightest step of a within-method ramp, as a blend towards white.
RAMP_LIGHTEST = 0.65


def _sizes(values: Sequence[float]) -> dict[float, float]:
    ordered = sorted(set(float(v) for v in values))
    if len(ordered) == 1:
        return {ordered[0]: SIZE_RANGE[1]}
    low, high = SIZE_RANGE
    return {v: low + (high - low) * i / (len(ordered) - 1) for i, v in enumerate(ordered)}


def _ot_scale() -> list[str]:
    """The registry's OT-tSNE colours, lightest first; the last is OT-tSNE's own."""
    ot = [m for m in registry.METHODS.values() if m.family == "ot"]
    return sorted({m.color for m in ot}, key=lambda c: -sum(to_rgb(c)))


def _ramp(color: str, n: int) -> list[str]:
    """``n`` shades ending at ``color``, lightest first.

    OT-tSNE's trajectories take evenly spaced steps of the OT colour scale; other methods blend
    their colour towards white.
    """
    base = np.array(to_rgb(color))
    if n == 1:
        return [to_hex(base)]
    scale = _ot_scale()
    if to_hex(base) == to_hex(scale[-1]):
        cmap = LinearSegmentedColormap.from_list("ot", scale)
        return [to_hex(cmap(t)) for t in np.linspace(0.0, 1.0, n)]
    blends = np.linspace(RAMP_LIGHTEST, 0.0, n)
    return [to_hex(base + (1.0 - base) * t) for t in blends]


def line_styles(points: pd.DataFrame) -> pd.DataFrame:
    """Per trajectory: method, label, colour, marker and line style, in legend order."""
    lines = points.drop_duplicates(LINE)[[LINE, "method", "line_label"]]
    order = registry.method_order(lines["method"].unique())
    lines = lines.assign(_rank=lines["method"].map(order.index)).sort_values(["_rank", LINE])
    rows = []
    for method, block in lines.groupby("method", sort=False):
        style = registry.resolve(method)
        colors = _ramp(style.color, len(block))
        for color, (_, line) in zip(colors, block.iterrows()):
            rows.append(
                {
                    LINE: line[LINE],
                    "method": method,
                    "line_label": line["line_label"],
                    "color": color,
                    "marker": style.marker,
                    "edge_color": style.edge_color,
                    "edge_width": style.edge_width,
                    "dashes": DASHES.get(method, "-"),
                }
            )
    return pd.DataFrame(rows)


def _arrows(ax, fig, xs, ys, color) -> None:
    """A small head at the middle of every edge, pointing towards the larger swept value.

    Each shaft is trimmed to a tip, so it does not paint over the line's dash pattern; the trim is
    measured on screen, because the two axes span different ranges.
    """
    for (x0, y0), (x1, y1) in zip(zip(xs[:-1], ys[:-1]), zip(xs[1:], ys[1:])):
        mid = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
        start = ax.transData.transform((x0, y0))
        end = ax.transData.transform(mid)
        length_pt = float(np.hypot(*(end - start))) * 72.0 / float(fig.dpi)
        ax.annotate(
            "",
            xy=mid,
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": LINE_WIDTH,
                "mutation_scale": 6,
                "shrinkA": max(0.0, length_pt - 1.5),
                "shrinkB": 0.0,
            },
            annotation_clip=False,
            zorder=3,
        )


def _is_reported(points: pd.DataFrame, reported: Mapping[str, object]) -> pd.Series:
    mask = pd.Series(True, index=points.index)
    for column, value in reported.items():
        if column not in points.columns:
            continue
        values = points[column]
        try:
            mask &= np.isclose(pd.to_numeric(values), float(value))
        except (TypeError, ValueError):
            mask &= values.astype(str) == str(value)
    return mask


def trajectory_grid(
    points: pd.DataFrame,
    name: str,
    *,
    sweep: str,
    pair: str,
    datasets: Sequence[str],
    output_root: Path,
    reported: Mapping[str, object] | None = None,
    dataset_labels: Mapping[str, str] | None = None,
    ncols: int | None = None,
    subdir: str | None = None,
) -> Path:
    """Draw ``pair`` for every dataset, one trajectory per run configuration along ``sweep``.

    :param points: output of :func:`~ne_aligned_utils.paper.summaries.sensitivity.sweep_points`.
    :param reported: settings of the reported configuration, for example
        ``{"method": "ot", "rel_reg": 1.0, "effective_k": 15}``. Matching points are drawn red.
        Settings absent from ``points`` are ignored.
    :param ncols: panels per row; default up to four.
    """
    if pair not in PAIRS:
        raise ValueError(f"pair must be one of {sorted(PAIRS)}")
    x, y, x_label, y_label = PAIRS[pair]
    datasets = list(datasets)
    missing = sorted(set(datasets) - set(points["dataset"]))
    if missing:
        raise ValueError(f"no trajectory points for {missing}")
    labels_for = dict(registry.DATASET_LABELS) | dict(dataset_labels or {})
    styles = line_styles(points)
    sizes = _sizes(points[sweep])
    reported_mask = _is_reported(points, reported or {})

    sweep_label = SWEEP_LABELS.get(sweep, sweep)
    legend_labels = [*styles["line_label"], f"increasing {sweep_label}"]
    has_reported = bool(reported_mask.any())
    if has_reported:
        legend_labels.append("reported configuration")

    columns = min(4, len(datasets)) if ncols is None else int(ncols)
    rows = -(-len(datasets) // columns)
    fig, axes, legend_ax, _geom = layout.grid_with_legend(columns, rows, legend_labels)
    flat = np.atleast_1d(axes).ravel()
    for spare in flat[len(datasets) :]:
        spare.remove()

    for index, (ax, dataset) in enumerate(zip(flat, datasets)):
        block = points[points["dataset"] == dataset]
        # Limits first: the arrow trims measure edges on screen.
        round_three_ticks(ax, block[f"{x}_mean"], block[f"{y}_mean"])
        for _, style in styles.iterrows():
            line = block[block[LINE] == style[LINE]].sort_values(sweep)
            if line.empty:
                continue
            xs, ys = line[f"{x}_mean"].to_numpy(), line[f"{y}_mean"].to_numpy()
            ax.plot(
                xs,
                ys,
                linestyle=style["dashes"],
                color=style["color"],
                linewidth=LINE_WIDTH,
                clip_on=False,
            )
            _arrows(ax, fig, xs, ys, style["color"])
            ax.scatter(
                xs,
                ys,
                s=[sizes[float(v)] for v in line[sweep]],
                c=style["color"],
                marker=style["marker"],
                edgecolors=style["edge_color"],
                linewidths=style["edge_width"],
                alpha=MARKER_ALPHA,
                zorder=2,
                clip_on=False,
            )
            mark = line[reported_mask.loc[line.index]]
            if not mark.empty:
                ax.scatter(
                    mark[f"{x}_mean"],
                    mark[f"{y}_mean"],
                    s=[sizes[float(v)] for v in mark[sweep]],
                    c=REPORTED_COLOR,
                    marker=style["marker"],
                    edgecolors=style["edge_color"],
                    linewidths=style["edge_width"],
                    zorder=4,
                    clip_on=False,
                )
        ax.set_title(labels_for.get(dataset, dataset))
        # Panels have their own scales, so every panel keeps its tick labels.
        ax.set_xlabel(x_label if index >= len(datasets) - columns else "")
        ax.set_ylabel(y_label if index % columns == 0 else "")
        layout.style_axes(ax)

    marker_size = SIZE_RANGE[1] ** 0.5
    handles = [
        Line2D(
            [],
            [],
            linestyle=style["dashes"],
            marker=style["marker"],
            markersize=marker_size,
            markeredgecolor=style["edge_color"],
            markeredgewidth=style["edge_width"],
            color=style["color"],
            linewidth=LINE_WIDTH,
        )
        for _, style in styles.iterrows()
    ]
    handles.append(
        Line2D([], [], linestyle="-", marker=">", markersize=3.4, color="#1f1f1f", linewidth=0.7)
    )
    if has_reported:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=registry.resolve("ot").marker,
                markersize=marker_size,
                color=REPORTED_COLOR,
            )
        )
    legend_ax.legend(
        handles,
        legend_labels,
        loc="center",
        ncol=layout.legend_columns(legend_labels),
        frameon=False,
    )
    return layout.save_figure(fig, name, subdir=subdir, root=output_root)


__all__ = ["DASHES", "PAIRS", "SWEEP_LABELS", "line_styles", "trajectory_grid"]
