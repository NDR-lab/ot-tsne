"""Metric scatter grids: one per dataset pair of metrics, and one pooled over datasets.

Both draw four family columns (OT-tSNE, integrated visualization, integrate-then-embed, no
integration) over a shared legend row, and repeat the OT-tSNE point in every column as an anchor.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ne_aligned_utils.paper import layout
from ne_aligned_utils.paper.rendering import headings, legends
from ne_aligned_utils.paper.rendering.context import RenderContext
from ne_aligned_utils.paper.rendering.metrics import (
    draw_metric_point,
    metric_span,
    round_three_ticks,
)

#: Legend-row placement shared by both grids (figure fractions).
LEGEND_LIFT = -0.003
LEGEND_SHIFT = {"ot": 0.016, "dedicated": 0.0402, "integration": 0.0085}
OT_ANCHOR_X = 0.075

#: Pooled figure rows, top first: (x metric, y metric, x label, y label).
POOLED_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("kbet", "purity", "kBET", "Neighbor purity"),
    ("ilisi", "silhouette", "iLISI", "Silhouette"),
)
POOLED_PANEL_ASPECT = 0.758
#: The silhouette row spans a narrower range, so it gets less height.
POOLED_LAST_ROW_SCALE = 0.85
POOLED_MARK_SCALE = 0.8
POOLED_TITLE_PT = 8.0
#: The integration column holds twelve marks: thinner outlines, and a translucent anchor.
CROWDED_FAMILY = "integration"
CROWDED_EDGE_PT = 0.25
CROWDED_ANCHOR_ALPHA = 0.8


def _family_grid(render_context: RenderContext, ordered, n_rows: int, panel_aspect: float):
    """``n_rows`` x four panels over a legend row measured from the methods it will list."""
    presentation = legends.LegendPresentation(
        render_context.methods,
        render_context.family_order,
        render_context.legend.integration_tool_order,
        dimension_size=render_context.legend.dimension_size,
    )
    fig, axes, legend_ax, _geom = layout.grid_with_legend(
        len(render_context.family_order),
        n_rows,
        [render_context.methods[k].label for k in ordered],
    )
    axes = np.atleast_2d(axes)
    reserved = layout.legend_height(
        legends.legend_rows(
            ordered, presentation=presentation, extra_rows=render_context.legend.extra_rows
        ),
        1,
    )
    # Rewrite the gridspec ratios (in inches) so the legend keeps ``reserved`` and the panels
    # shrink by ``panel_aspect``; resizing the figure alone would shrink the legend too.
    panels = (fig.get_figheight() - reserved) * float(panel_aspect)
    fig.set_figheight(reserved + panels)
    axes[0, 0].get_gridspec().set_height_ratios([*([panels / n_rows] * n_rows), reserved])
    return fig, axes, legend_ax, presentation


def _scale_row(fig, axes, row: int, scale: float) -> None:
    """Shrink one panel row by ``scale`` and the figure by the same amount."""
    gridspec = axes[0, 0].get_gridspec()
    ratios = list(gridspec.get_height_ratios())
    shrunk = ratios[row] * float(scale)
    fig.set_figheight(fig.get_figheight() - (ratios[row] - shrunk))
    ratios[row] = shrunk
    gridspec.set_height_ratios(ratios)


def _finish(fig, axes, legend_ax, ordered, presentation, *, tag_placement: str) -> None:
    legends.family_legends(
        fig,
        axes,
        legend_ax,
        legends.legend_order(ordered, presentation=presentation),
        lift=LEGEND_LIFT,
        shifts=LEGEND_SHIFT,
        ot_anchor_x=OT_ANCHOR_X,
        reference_anchor_x=None,
        presentation=presentation,
    )
    headings.place_family_tags(fig, list(axes[0]), placement=tag_placement)


def _family_title(ax, family, render_context, tag_placement):
    headings.set_family_title(
        ax,
        family,
        family_labels=render_context.family_labels,
        family_order=render_context.family_order,
        placement=tag_placement,
    )


def dataset_metric_grid(
    summary: pd.DataFrame,
    name: str,
    *,
    render_context: RenderContext,
    x: str,
    y: str,
    x_label: str,
    y_label: str,
    panel_aspect: float = 0.585,
    tag_placement: str = "top",
    subdir: str | None = None,
    anchor_method: str = "ot",
) -> Path:
    """One row per dataset: ``y`` against ``x``, each point a seed mean with its seed SD."""
    row_datasets = render_context.dataset_order
    if not row_datasets:
        raise ValueError("`datasets` must name at least one dataset.")
    empty = [d for d in row_datasets if not (summary["dataset"] == d).any()]
    if empty:
        raise ValueError(f"no summary rows for requested datasets: {empty}")
    summary = summary[summary["dataset"].isin(row_datasets)]
    ordered = render_context.order_methods(summary["method_key"].unique())
    fig, axes, legend_ax, presentation = _family_grid(
        render_context, ordered, len(row_datasets), panel_aspect
    )
    family_order = render_context.family_order
    for r, dataset in enumerate(row_datasets):
        row_block = summary[summary["dataset"] == dataset]
        anchor = row_block[row_block["method_key"] == anchor_method]
        row_xs, row_ys = (metric_span(row_block, column.removesuffix("_mean")) for column in (x, y))
        for c, family in enumerate(family_order):
            ax = axes[r, c]
            for _, row in row_block[row_block["family"] == family].iterrows():
                scatter = render_context.scatter_kwargs(row["method_key"], zorder=2)
                _draw(ax, row, x=x, y=y, scatter=scatter)
            if not anchor.empty:
                _draw(
                    ax,
                    anchor.iloc[0],
                    x=x,
                    y=y,
                    scatter=render_context.scatter_kwargs(anchor_method, zorder=4),
                )
            round_three_ticks(ax, row_xs, row_ys)
            if r == 0:
                _family_title(ax, family, render_context, tag_placement)
            ax.set_ylabel(y_label if c == 0 else "")
            ax.set_xlabel(x_label if r == len(row_datasets) - 1 else "")
            if c > 0:
                # The row shares one scale; keep the ticks, drop repeated numbers.
                ax.set_yticklabels([])
            if c == len(family_order) - 1:
                strip = ax.twinx()
                strip.set_yticks([])
                for side in ("top", "right", "bottom", "left"):
                    strip.spines[side].set_visible(False)
                strip.set_ylabel(
                    render_context.dataset_labels.get(dataset, dataset),
                    rotation=270,
                    labelpad=10,
                    va="center",
                )
            layout.style_axes(ax)
    _finish(fig, axes, legend_ax, ordered, presentation, tag_placement=tag_placement)
    return layout.save_figure(
        fig, name, appendix=False, subdir=subdir, root=render_context.output_root
    )


def _draw(ax, row, *, x, y, scatter):
    draw_metric_point(ax, row, x=x, y=y, scatter=scatter, error_color=scatter["c"])


def pooled_metric_grid(
    pooled: pd.DataFrame,
    name: str,
    *,
    render_context: RenderContext,
    pairs: Sequence[tuple[str, str, str, str]] = POOLED_PAIRS,
    tag_placement: str = "top",
    subdir: str | None = None,
    anchor_method: str = "ot",
) -> Path:
    """One row per metric pair; each point a mean over datasets, its bars the SD across datasets.

    Each row keeps its own scale and its own axis names. The x axis always spans [0, 1], so the two
    mixing metrics read on the same scale.
    """
    ordered = render_context.order_methods(pooled["method_key"].unique())
    anchor = pooled[pooled["method_key"] == anchor_method]
    family_order = render_context.family_order

    def scatter(key, family, **extra):
        kwargs = render_context.scatter_kwargs(key, **extra)
        kwargs["s"] = float(kwargs["s"]) * POOLED_MARK_SCALE
        if family == CROWDED_FAMILY and kwargs.get("edgecolors", "none") != "none":
            kwargs["linewidths"] = CROWDED_EDGE_PT
        return kwargs

    with plt.rc_context({"axes.titlesize": POOLED_TITLE_PT}):
        fig, axes, legend_ax, presentation = _family_grid(
            render_context, ordered, len(pairs), POOLED_PANEL_ASPECT
        )
        if len(pairs) > 1:
            _scale_row(fig, axes, len(pairs) - 1, POOLED_LAST_ROW_SCALE)
        for r, (x_stem, y_stem, x_label, y_label) in enumerate(pairs):
            x, y = f"{x_stem}_mean", f"{y_stem}_mean"
            row_xs = metric_span(pooled, x_stem)
            row_xs = (min(0.0, float(row_xs.min())), max(1.0, float(row_xs.max())))
            row_ys = metric_span(pooled, y_stem)
            for c, family in enumerate(family_order):
                ax = axes[r, c]
                for _, row in pooled[pooled["family"] == family].iterrows():
                    _draw(ax, row, x=x, y=y, scatter=scatter(row["method_key"], family, zorder=2))
                if not anchor.empty:
                    alpha = CROWDED_ANCHOR_ALPHA if family == CROWDED_FAMILY else 1.0
                    _draw(
                        ax,
                        anchor.iloc[0],
                        x=x,
                        y=y,
                        scatter=scatter(anchor_method, family, zorder=4, alpha=alpha),
                    )
                round_three_ticks(ax, row_xs, row_ys)
                if r == 0:
                    _family_title(ax, family, render_context, tag_placement)
                ax.set_ylabel(y_label if c == 0 else "")
                ax.set_xlabel(x_label)
                if c > 0:
                    ax.set_yticklabels([])
                layout.style_axes(ax)
        _finish(fig, axes, legend_ax, ordered, presentation, tag_placement=tag_placement)
    return layout.save_figure(
        fig, name, appendix=False, subdir=subdir, root=render_context.output_root
    )
