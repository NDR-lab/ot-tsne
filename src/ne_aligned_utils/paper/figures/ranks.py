"""Aggregate-rank and top-r figures from already-scored tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory

from ne_aligned_utils.paper import layout


@dataclass(frozen=True)
class RankStyle:
    """Rank-figure geometry and colours."""

    rank_ramp: Colormap
    rank_mark_size: float
    rank_ramp_offset_pt: float
    rank_ramp_gap_pt: float
    rank_ramp_thickness_pt: float
    rank_ramp_tick_pad_pt: float
    rank_panel_width: float
    rank_panel_gap: float
    rank_side_blocks: float
    rank_value_format: str
    rank_label_pad_pt: float
    rank_edge_pad_in: float

    def __post_init__(self):
        object.__setattr__(self, "rank_ramp", self.rank_ramp.copy())


@dataclass(frozen=True)
class RankComposer:
    """Draw a selected, already-ranked pool without modifying its data."""

    style: RankStyle
    method_labels: Mapping[str, str]
    block_titles: Mapping[str, str]

    def __post_init__(self):
        object.__setattr__(self, "method_labels", MappingProxyType(dict(self.method_labels)))
        object.__setattr__(self, "block_titles", MappingProxyType(dict(self.block_titles)))

    def _figure_width(self, n_panels: int) -> float:
        """Panels, the gutters between them, and the name column."""
        style = self.style
        panels = n_panels * style.rank_panel_width + (n_panels - 1) * style.rank_panel_gap
        return self.style.rank_side_blocks + panels

    def _rank_axis_ramp(self, ax, n_competitors: int, norm) -> None:
        """Lay the rank ramp along the x axis, in place of the axis line."""
        figure = ax.figure
        figure.canvas.draw()
        height_in = ax.get_window_extent().transformed(figure.dpi_scale_trans.inverted()).height
        top = -(self.style.rank_ramp_offset_pt + self.style.rank_ramp_gap_pt) / 72.0 / height_in
        bottom = top - self.style.rank_ramp_thickness_pt / 72.0 / height_in
        limits = (ax.get_xlim(), ax.get_ylim())
        ax.imshow(
            np.linspace(1, n_competitors, 256).reshape(1, -1),
            cmap=self.style.rank_ramp,
            norm=norm,
            aspect="auto",
            origin="lower",
            extent=(1, n_competitors, bottom, top),
            transform=blended_transform_factory(ax.transData, ax.transAxes),
            clip_on=False,
            zorder=4,
        )
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(
            axis="x",
            length=0,
            pad=self.style.rank_ramp_offset_pt
            + self.style.rank_ramp_gap_pt
            + self.style.rank_ramp_thickness_pt
            + self.style.rank_ramp_tick_pad_pt,
        )

    def _print_values(self, ax, rows, values, *, side: int) -> None:
        """Print each mark's rank beside it, on the worse-rank side."""
        inset = self.style.rank_label_pad_pt + self.style.rank_mark_size**0.5 / 2
        size = float(plt.rcParams["ytick.labelsize"])
        for row, value in zip(rows, values):
            if not np.isfinite(value):
                continue
            note = ax.annotate(
                self.style.rank_value_format.format(value),
                xy=(value, row),
                xytext=(side * inset, 0),
                textcoords="offset points",
                ha="left" if side == 1 else "right",
                va="center",
                fontsize=size,
                annotation_clip=False,
            )
            # Values may sit outside the axes; keep them visible and out of the layout.
            note.set_clip_on(False)
            note.set_in_layout(False)

    def _clear_rank_names(self, fig, axes) -> None:
        """Move the row names left of the values printed beside the marks."""
        fig.canvas.draw()
        inv = fig.dpi_scale_trans.inverted()
        overlap = 0.0
        for ax in axes:
            notes = [t for t in ax.texts if t.get_text()]
            names = [t for t in ax.get_yticklabels() if t.get_text()]
            if not notes or not names:
                continue
            left = min(t.get_window_extent().transformed(inv).x0 for t in notes)
            right = max(t.get_window_extent().transformed(inv).x1 for t in names)
            overlap = max(overlap, right - left)
        if overlap <= 0.0:
            return
        pad = float(plt.rcParams["ytick.major.pad"]) + overlap * 72.0 + self.style.rank_label_pad_pt
        for ax in axes:
            ax.tick_params(axis="y", pad=pad)

    def _clear_right_edge(self, fig, axes) -> None:
        """Pull the panels in from the right until the printed values fit on the page."""
        fig.canvas.draw()
        inv = fig.transFigure.inverted()
        right = max(
            (
                text.get_window_extent().transformed(inv).x1
                for ax in axes
                for text in ax.texts
                if text.get_text()
            ),
            default=0.0,
        )
        edge = 1.0 - self.style.rank_edge_pad_in / fig.get_figwidth()
        if right <= edge:
            return
        fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0 - (right - edge), 1.0))

    @staticmethod
    def _rank_axis(ax, n_competitors: int, *, reverse: bool) -> None:
        """Rank axis ticked at 1, the middle and the last place; 1 on the left unless reversed."""
        ticks = sorted({1, int(round((1 + n_competitors) / 2)), int(n_competitors)})
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        pad = 0.04 * (n_competitors - 1)
        low, high = 1 - pad, n_competitors + pad
        ax.set_xlim((high, low) if reverse else (low, high))

    def rank_aggregate(
        self,
        agg: pd.DataFrame,
        *,
        height: float = 4.3,
        blocks: Sequence[str] | None = None,
        order_per_block: bool = True,
        show_values: bool = True,
        reverse: bool = False,
    ) -> Figure:
        """One panel per score block (all metrics, bio conservation, batch mixing).

        By default each panel lists the methods in its own rank order and prints each mean rank
        beside its mark. ``order_per_block=False`` orders every panel by the first block instead.
        """
        wanted = self.block_titles if blocks is None else blocks
        present = set(agg["block"])
        blocks = [b for b in wanted if b in present]
        if not blocks:
            raise ValueError(f"no known metric block in the frame; found {sorted(present)}.")
        n_competitors = int(agg["n_competitors"].iloc[0])
        ranked = {
            b: agg[agg["block"] == b].sort_values("aggregate_rank")["method_key"].tolist()
            for b in blocks
        }
        if any(len(order) != n_competitors for order in ranked.values()):
            raise ValueError(
                f"the frame holds a different pool than the {n_competitors} methods it was ranked "
                "over. Exclude before ranking, not after."
            )

        fig = plt.figure(figsize=(self._figure_width(len(blocks)), height))
        grid = fig.add_gridspec(1, len(blocks))
        axes = [fig.add_subplot(grid[0, c]) for c in range(len(blocks))]
        fig.get_layout_engine().set(w_pad=self.style.rank_edge_pad_in)
        norm = Normalize(vmin=1, vmax=n_competitors)
        for ax, block in zip(axes, blocks):
            order = ranked[block] if order_per_block else ranked[blocks[0]]
            y = np.arange(len(order))
            panel = agg[agg["block"] == block].set_index("method_key")["aggregate_rank"]
            values = np.array([panel.get(k, np.nan) for k in order], dtype=float)
            ax.scatter(
                values,
                y,
                s=self.style.rank_mark_size,
                c=values,
                cmap=self.style.rank_ramp,
                norm=norm,
                edgecolors="none",
                clip_on=False,
                zorder=3,
            )
            ax.set_title(self.block_titles[block])
            self._rank_axis(ax, n_competitors, reverse=reverse)
            # Label the shared rank scale under the middle panel.
            ax.set_xlabel("Aggregate rank" if ax is axes[len(axes) // 2] else "")
            ax.set_ylim(len(order) - 0.5, -0.5)
            ax.set_yticks(y)
            named = order_per_block or ax is axes[0]
            ax.set_yticklabels([self.method_labels[k] for k in order] if named else [])
            ax.tick_params(axis="y", length=0)
            if show_values:
                self._print_values(ax, y, values, side=-1 if reverse else 1)
            layout.style_axes(ax)

        if show_values:
            if reverse:
                self._clear_rank_names(fig, axes)
            else:
                self._clear_right_edge(fig, axes)
        for ax in axes:
            self._rank_axis_ramp(ax, n_competitors, norm)
        return fig


@dataclass(frozen=True)
class TopKComposer:
    """Detached labels, stable tie order, palette and measured bar geometry."""

    method_labels: Mapping[str, str]
    method_order: tuple[str, ...]
    palette: Mapping[str, str]
    dataset_labels: Mapping[str, str]
    parts: tuple[tuple[str, str | None], ...]
    label_clearance: float
    panel_fraction: float
    legend_loc: str
    legend_pad: float

    def __post_init__(self):
        for name in ("method_labels", "palette", "dataset_labels"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(self, "method_order", tuple(self.method_order))
        object.__setattr__(self, "parts", tuple(tuple(part) for part in self.parts))

    def _trim_axis(self, axes) -> None:
        """Shrink the x axis to the counts printed at the bar ends, plus one label's clearance."""
        axes = list(axes)
        axes[0].figure.canvas.draw()
        ends = [
            text.get_window_extent().transformed(ax.transData.inverted()).x1
            for ax in axes
            for text in ax.texts
            if text.get_text() and text not in {ax.title, ax.xaxis.label, ax.yaxis.label}
        ]
        if not ends:
            return
        for ax in axes:
            ax.set_xlim(0, max(ends) + self.label_clearance)

    def _draw_bar_column(self, ax, order, wides, totals, *, hues, per_dataset, span) -> np.ndarray:
        """Draw one column of the top-three bars, and return its y positions."""
        y = np.arange(len(order))
        left = np.zeros(len(order))
        for dataset in per_dataset:
            for column, hatch in self.parts:
                values = wides[column].reindex(index=order)[dataset].to_numpy(dtype=float)
                ax.barh(
                    y,
                    values,
                    left=left,
                    height=0.72,
                    color=hues[dataset],
                    # A hairline of the surface between touching segments, so a boundary between two
                    # segments never reads as one wider segment. It also supplies the hatch color.
                    edgecolor="#ffffff",
                    linewidth=0.4,
                    hatch=hatch,
                    zorder=2,
                )
                left += values

        # The count at the bar end, in ink rather than in the bar's own color: a number is text and
        # wears text color, or it reads as one more encoded channel.
        for yi, value in zip(y, totals.reindex(order).to_numpy(dtype=float)):
            ax.text(
                value + 0.012 * span,
                yi,
                f"{value:g}",
                va="center",
                ha="left",
                fontsize=float(plt.rcParams["xtick.labelsize"]),
            )

        ax.set(yticks=y, yticklabels=[self.method_labels[k] for k in order])
        ax.set_ylim(len(order) - 0.5, -0.5)
        # Headroom for the direct labels, which would otherwise sit outside the axes and be clipped.
        # Trimmed to what they measure once drawn -- see :func:`_trim_axis`.
        ax.set_xlim(0, span * 1.10)
        # A count axis has no fractional values, so the ticks are integers.
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="y", length=0)
        layout.style_axes(ax)
        return y

    def top_k_finishes(
        self,
        counts: pd.DataFrame,
        *,
        palette: Mapping[str, str] | None = None,
        dataset_labels: Mapping[str, str] | None = None,
        family_labels: tuple[str, str] = ("Bio conservation", "Batch mixing"),
        height: float = 2.05,
        width: float | None = None,
        columns: int = 2,
        legend_ncol: int = 1,
        font_pt: float | None = None,
        drop_zeros: bool = True,
    ) -> Figure:
        """Horizontal bars: one per method, its length the number of top-r finishes."""
        top_k = int(counts["top_k"].iloc[0])
        order = self._bar_method_order(counts)
        totals = counts.groupby("method_key")["n_top_k"].sum().reindex(order)
        if drop_zeros:
            # Omit zero-finish bars; report excluded method names separately to retain pool context.
            order = [key for key in order if totals[key] > 0]
            totals = totals.reindex(order)
            counts = counts[counts["method_key"].isin(order)]
        sizes = (
            {}
            if font_pt is None
            else {
                key: float(font_pt)
                for key in (
                    "axes.labelsize",
                    "xtick.labelsize",
                    "ytick.labelsize",
                    "legend.fontsize",
                )
            }
        )
        with plt.rc_context(sizes):
            return self._draw_top_k(
                counts,
                order=order,
                totals=totals,
                palette=palette,
                dataset_labels=dataset_labels,
                family_labels=family_labels,
                height=height,
                width=width,
                columns=columns,
                legend_ncol=legend_ncol,
                top_k=top_k,
            )

    def _draw_top_k(
        self,
        counts,
        *,
        order,
        totals,
        palette,
        dataset_labels,
        family_labels,
        height,
        width,
        columns,
        legend_ncol,
        top_k,
    ) -> Figure:
        """Draw every count, label and legend under the caller's type-size context."""
        hues = dict(self.palette if palette is None else palette)
        per_dataset = [d for d in hues if d in set(counts["dataset"])]
        names = dict(self.dataset_labels if dataset_labels is None else dataset_labels)
        # One span for every column, computed over the whole pool: the columns are two halves of a
        # single ranking, so a bar must be measured against the same axis wherever it was dealt.
        span = max(1.0, float(totals.max()))
        wides = {
            column: counts.pivot_table(
                index="method_key", columns="dataset", values=column, aggfunc="sum", fill_value=0
            ).reindex(columns=per_dataset, fill_value=0)
            for column, _hatch in self.parts
        }

        fig = plt.figure(figsize=layout.full_width(height) if width is None else (width, height))
        # Equal ratios give the columns equal axes boxes, so one x limit
        # means one scale. Tick labels sit outside the box and do not eat into it.
        # Reserve a gridspec cell for the margin; constrained layout ignores figure-level right
        # margins.
        share = self.panel_fraction if columns == 1 and width is None else 1.0
        grid = fig.add_gridspec(
            1,
            columns + 1,
            width_ratios=[share / columns] * columns + [max(1.0 - share, 1e-6)],
        )
        axes = [fig.add_subplot(grid[0, i]) for i in range(columns)]
        fig.add_subplot(grid[0, columns]).axis("off")
        for ax, chunk in zip(axes, np.array_split(np.asarray(order, dtype=object), columns)):
            self._draw_bar_column(
                ax, list(chunk), wides, totals, hues=hues, per_dataset=per_dataset, span=span
            )

        handles = [
            Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markersize=5,
                markerfacecolor=hues[d],
                markeredgecolor="none",
            )
            for d in per_dataset
        ] + [
            # Neutral grey: these two entries are about the *texture*, and giving either a dataset's
            # hue would read as a claim about that dataset.
            Patch(facecolor="#9a9a9a", edgecolor="#ffffff", linewidth=0.4, hatch="////"),
            Patch(facecolor="#9a9a9a", edgecolor="#ffffff", linewidth=0.4),
        ]
        # The last column holds the tail of the ranking, so its lower rows are the short and empty
        # bars: the key goes there, in the white space the ranking itself opens up.
        # Anchor just past the longest bar in the rows the key covers, putting it beside
        # them. Which rows those are depends on how tall the key is, so it is measured rather than
        # guessed: draw it once, see what it spans, and move it.
        axes[-1].legend(
            handles=handles,
            labels=[names.get(d, d) for d in per_dataset] + list(family_labels),
            loc=self.legend_loc,
            ncol=legend_ncol,
            frameon=False,
            borderaxespad=self.legend_pad,
            # Remove invisible legend padding so measured clearance reflects visible ink.
            borderpad=0.0,
            handletextpad=0.4,
            columnspacing=1.0,
            labelspacing=0.35,
        )

        self._trim_axis(axes)
        # Center the shared axis name on the panel pair rather than the whole figure.
        label = fig.supxlabel(
            f"Top-{top_k} finishes", fontsize=float(plt.rcParams["axes.labelsize"])
        )
        fig.canvas.draw()
        boxes = [ax.get_position() for ax in axes]
        label.set_x(0.5 * (boxes[0].x0 + boxes[-1].x1))
        return fig

    def _bar_method_order(self, counts: pd.DataFrame) -> list[str]:
        """Methods by total placings, descending, ties broken by the registry's own order."""
        totals = counts.groupby("method_key")["n_top_k"].sum()
        canonical = {k: i for i, k in enumerate(self.method_order)}
        return sorted(totals.index, key=lambda k: (-float(totals[k]), canonical[k]))
