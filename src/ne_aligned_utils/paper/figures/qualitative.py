"""Embedding galleries: one dataset per figure, or every dataset stacked in one figure."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

from ne_aligned_utils.paper import layout


@dataclass(frozen=True)
class QualitativeStyle:
    """Detached gallery geometry, colors and deterministic draw-order seed."""

    qualitative_rows: tuple[tuple[str, str], ...]
    qualitative_margins: Mapping[str, float]
    qualitative_batch_colours: tuple[str, ...]
    qualitative_rule_at: float
    qualitative_legend_gap: float
    qualitative_batch_labels: tuple[str, ...]
    qualitative_handletextpad: float
    qualitative_columnspacing: float
    qualitative_group_band: float
    qualitative_group_lift: float
    qualitative_group_tick: float
    qualitative_group_text_gap: float
    qualitative_group_pt: float
    point_ref_size: float
    point_ref_n: int
    panel_trim: float
    seed: int

    def __post_init__(self):
        object.__setattr__(
            self, "qualitative_margins", MappingProxyType(dict(self.qualitative_margins))
        )
        for name in ("qualitative_rows", "qualitative_batch_colours", "qualitative_batch_labels"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class GalleryBlock:
    """One dataset's preloaded gallery inputs for the stacked figure."""

    name: str
    embeddings: Mapping[str, np.ndarray]
    caches: Mapping[str, Mapping]
    palette: np.ndarray
    labels: np.ndarray


#: Stacked-gallery geometry, inches.
STACK_TEXT_PT = 8.0
STACK_MARGIN_IN = 0.05
STACK_ROW_LABEL_IN = 0.20
STACK_COLUMN_GAP_IN = 0.04
STACK_HEADING_IN = 0.14
STACK_BLOCK_TITLE_IN = 0.15
STACK_ROW_GAP_IN = 0.02
STACK_BLOCK_GAP_IN = 0.10
STACK_BRACE_NAME_IN = 0.10
STACK_LEGEND_GAP_IN = 0.03
STACK_LEGEND_IN = 0.11


@dataclass(frozen=True)
class QualitativeComposer:
    """Draw two differently colored views of the same preloaded coordinates."""

    style: QualitativeStyle
    method_labels: Mapping[str, str]

    def __post_init__(self):
        object.__setattr__(self, "method_labels", MappingProxyType(dict(self.method_labels)))

    def embedding_panels(
        self,
        cached: Mapping[str, tuple[np.ndarray, dict]],
        palette: np.ndarray,
        *,
        columns: tuple[tuple[str, str | None], ...],
        title_pt: float = 8.0,
        groups: tuple[tuple[str, int], ...] | None = None,
    ) -> Figure:
        """One dataset's embeddings: methods across, batch over cell type down."""
        # Place batch and cell-type keys side by side to limit legend height.
        block = self._batch_legend()
        reserved = (
            self._measure_legend_block(block, layout.FULL_WIDTH)[1]
            + self.style.qualitative_legend_gap
        )
        band = self.style.qualitative_group_band if groups else 0.0
        geom = layout.square_grid(
            len(columns),
            len(self.style.qualitative_rows),
            legend=reserved + band,
            **self.style.qualitative_margins,
        )
        fig = plt.figure(figsize=geom.figsize)
        # Reserve gridspec rows for the legend and group band so neither overlaps panels.
        heights = [geom.figsize[1] - reserved - band, reserved]
        if groups:
            heights.insert(1, band)
        outer = fig.add_gridspec(len(heights), 1, height_ratios=heights)
        inner = outer[0].subgridspec(len(self.style.qualitative_rows), len(columns))
        axes = [
            [fig.add_subplot(inner[r, c]) for c in range(len(columns))]
            for r in range(len(self.style.qualitative_rows))
        ]
        if groups:
            group_ax = fig.add_subplot(outer[1])
            group_ax.axis("off")
        legend_ax = fig.add_subplot(outer[-1])
        legend_ax.axis("off")
        rng = np.random.default_rng(self.style.seed)

        for c, (key, heading) in enumerate(columns):
            emb, cache_c = cached[key]
            order = rng.permutation(len(emb))
            for r, (color_kind, row_label) in enumerate(self.style.qualitative_rows):
                if color_kind == "batch":
                    colors = np.asarray(self.style.qualitative_batch_colours)[
                        np.asarray(cache_c["batch_ids"], dtype=int)
                    ]
                else:
                    labels = np.concatenate(
                        [np.asarray(cache_c["labels0"][0]), np.asarray(cache_c["labels1"][0])]
                    ).astype(int)
                    colors = palette[labels]
                ax = axes[r][c]
                ax.scatter(
                    emb[order, 0],
                    emb[order, 1],
                    c=colors[order],
                    s=self._point_size(len(emb)),
                    alpha=0.7,
                    linewidths=0,
                    edgecolors="none",
                    zorder=-1,
                )
                ax.set(xticks=[], yticks=[])
                self._fill_panel(ax, emb)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if c == len(columns) - 1:
                    # Place coloring labels on the right, separate from panel coordinates.
                    ax.yaxis.set_label_position("right")
                    # Use the column-heading font size for row labels.
                    ax.set_ylabel(
                        row_label, rotation=270, va="bottom", labelpad=6, fontsize=title_pt
                    )
                if r == 0:
                    text = heading if heading is not None else self.method_labels[key]
                    ax.set_title(text, fontsize=title_pt)

        self._qualitative_legend(legend_ax, block)
        # Place the rule after constrained layout settles the axes positions.
        fig.canvas.draw()
        self._qualitative_rule(fig, axes)
        if groups:
            self._qualitative_groups(fig, axes, groups)
        return fig

    def _qualitative_rule(self, fig, axes) -> None:
        """The vertical rule setting the baseline column apart from the methods."""
        inv = fig.transFigure.inverted()
        boxes = [[ax.get_window_extent().transformed(inv) for ax in row] for row in axes]
        x = boxes[0][0].x1 + self.style.qualitative_rule_at * (boxes[0][1].x0 - boxes[0][0].x1)
        fig.add_artist(
            Line2D(
                [x, x],
                [boxes[-1][0].y0, boxes[0][0].y1],
                transform=fig.transFigure,
                color="#000000",
                linewidth=0.6,
            )
        )

    def _qualitative_groups(self, fig, axes, groups) -> None:
        """Brace each column group and name what it has in common."""
        total = sum(count for _name, count in groups)
        if total != len(axes[0]):
            raise ValueError(
                f"groups cover {total} columns but the figure has {len(axes[0])}: "
                f"{[name for name, _ in groups]}"
            )
        inverse = fig.transFigure.inverted()
        boxes = [ax.get_window_extent().transformed(inverse) for ax in axes[-1]]
        height = fig.get_figheight()
        rule = min(box.y0 for box in boxes) - self.style.qualitative_group_lift / height
        tick = self.style.qualitative_group_tick / height
        self._draw_braces(fig, boxes, groups, rule, tick)

    def _batch_legend(self, palette_colors=None):
        """``(handles, labels, ncol)`` for the batch key -- the only key this figure carries."""
        palette_colors = (
            self.style.qualitative_batch_colours if palette_colors is None else palette_colors
        )
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=4,
                markerfacecolor=color,
                markeredgecolor="none",
            )
            for color in palette_colors
        ]
        return (
            handles,
            list(self.style.qualitative_batch_labels),
            len(self.style.qualitative_batch_labels),
        )

    def _measure_legend_block(self, block, width: float) -> tuple[float, float]:
        """``(width, height)`` in inches of one key, measured by drawing it."""
        handles, labels, ncol = block
        scratch = plt.figure(figsize=(width, 1.0))
        ax = scratch.add_subplot(111)
        ax.axis("off")
        drawn = Legend(
            ax,
            handles,
            labels,
            loc="lower left",
            ncol=ncol,
            frameon=False,
            borderaxespad=0.0,
            borderpad=0.0,
            handletextpad=self.style.qualitative_handletextpad,
            columnspacing=self.style.qualitative_columnspacing,
            labelspacing=0.3,
            alignment="left",
        )
        ax.add_artist(drawn)
        extent = drawn.get_window_extent(scratch.canvas.get_renderer())
        plt.close(scratch)
        return extent.width / scratch.dpi, extent.height / scratch.dpi

    def _qualitative_legend(self, legend_ax, block) -> None:
        """The batch key, centered under the panels."""
        handles, labels, ncol = block
        drawn = Legend(
            legend_ax,
            handles,
            labels,
            loc="center",
            ncol=ncol,
            frameon=False,
            borderaxespad=0.0,
            # Remove border padding so the key fits its reserved band.
            borderpad=0.0,
            handletextpad=self.style.qualitative_handletextpad,
            columnspacing=self.style.qualitative_columnspacing,
            labelspacing=0.3,
        )
        legend_ax.add_artist(drawn)

    def _point_size(self, n: int) -> float:
        """Marker area for an embedding of ``n`` points, in points squared."""
        return float(self.style.point_ref_size * np.sqrt(self.style.point_ref_n / max(n, 1)))

    def _fill_panel(self, ax, emb: np.ndarray) -> None:
        """Frame a panel on the bulk of its points, at 1:1 aspect."""
        lo = np.quantile(emb, self.style.panel_trim, axis=0)
        hi = np.quantile(emb, 1.0 - self.style.panel_trim, axis=0)
        centre = (lo + hi) / 2
        half = float(np.max(hi - lo)) / 2 * 1.06
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_aspect("equal", adjustable="box")

    def stacked_panels(
        self,
        blocks: list[GalleryBlock],
        *,
        columns: tuple[tuple[str, str], ...],
        groups: tuple[tuple[str, int], ...],
        braces: str = "once",
    ) -> Figure:
        """Every dataset's gallery in one figure: headings once at the top, key once at the foot.

        ``braces="once"`` draws the column-group braces under the last block only; ``"each"``
        draws them under every block.
        """
        if braces not in {"once", "each"}:
            raise ValueError("braces must be 'once' or 'each'")
        style = self.style
        n_cols, n_rows = len(columns), len(style.qualitative_rows)
        panel_in = (
            layout.FULL_WIDTH
            - STACK_MARGIN_IN
            - STACK_ROW_LABEL_IN
            - (n_cols - 1) * STACK_COLUMN_GAP_IN
        ) / n_cols
        panels_in = n_rows * panel_in + (n_rows - 1) * STACK_ROW_GAP_IN
        brace_in = (
            style.qualitative_group_lift
            + style.qualitative_group_tick
            + style.qualitative_group_text_gap
            + STACK_BRACE_NAME_IN
        )
        per_block = brace_in if braces == "each" else 0.0
        foot = (0.0 if braces == "each" else brace_in) + STACK_LEGEND_GAP_IN + STACK_LEGEND_IN
        height = (
            2 * STACK_MARGIN_IN
            + STACK_HEADING_IN
            + len(blocks) * (STACK_BLOCK_TITLE_IN + panels_in + per_block)
            + (len(blocks) - 1) * STACK_BLOCK_GAP_IN
            + foot
        )
        with plt.rc_context({"figure.constrained_layout.use": False}):
            fig = plt.figure(figsize=(layout.FULL_WIDTH, height))
        width_in, height_in = fig.get_figwidth(), fig.get_figheight()
        rng = np.random.default_rng(style.seed)

        top = 1.0 - STACK_MARGIN_IN / height_in
        batch_rows = []
        for index, block in enumerate(blocks):
            top -= (STACK_BLOCK_TITLE_IN + (STACK_HEADING_IN if index == 0 else 0.0)) / height_in
            row_axes = []
            for c, (key, heading) in enumerate(columns):
                emb, cache = block.embeddings[key], block.caches[key]
                order = rng.permutation(len(emb))
                left = (STACK_MARGIN_IN + c * (panel_in + STACK_COLUMN_GAP_IN)) / width_in
                for r, (kind, row_label) in enumerate(style.qualitative_rows):
                    if kind == "batch":
                        colors = np.asarray(style.qualitative_batch_colours)[
                            np.asarray(cache["batch_ids"], dtype=int)
                        ]
                    else:
                        colors = block.palette[block.labels]
                    bottom = top - ((r + 1) * panel_in + r * STACK_ROW_GAP_IN) / height_in
                    ax = fig.add_axes([left, bottom, panel_in / width_in, panel_in / height_in])
                    ax.scatter(
                        emb[order, 0],
                        emb[order, 1],
                        c=colors[order],
                        s=self._point_size(len(emb)),
                        alpha=0.7,
                        linewidths=0,
                        edgecolors="none",
                        zorder=-1,
                    )
                    ax.set(xticks=[], yticks=[])
                    self._fill_panel(ax, emb)
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    if c == n_cols - 1:
                        ax.yaxis.set_label_position("right")
                        ax.set_ylabel(
                            row_label,
                            rotation=270,
                            va="bottom",
                            labelpad=6,
                            fontsize=STACK_TEXT_PT,
                        )
                    if index == 0 and r == 0:
                        ax.set_title(heading, fontsize=STACK_TEXT_PT, pad=3)
                    if r == 0:
                        row_axes.append(ax)
            batch_rows.append(row_axes)
            top -= (panels_in + per_block) / height_in
            if index < len(blocks) - 1:
                top -= STACK_BLOCK_GAP_IN / height_in

        self._stack_titles(fig, blocks, batch_rows)
        for row_axes in batch_rows if braces == "each" else batch_rows[-1:]:
            self._stack_braces(fig, row_axes, groups, panel_in)
        handles, labels, ncol = self._batch_legend()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, STACK_MARGIN_IN / height_in),
            ncol=ncol,
            frameon=False,
            fontsize=STACK_TEXT_PT,
            borderpad=0.0,
            borderaxespad=0.0,
            handletextpad=style.qualitative_handletextpad,
            columnspacing=style.qualitative_columnspacing,
        )
        return fig

    def _stack_titles(self, fig, blocks, batch_rows) -> None:
        """A bold letter at the left margin and the dataset name (n cells) over its panels."""
        fig.canvas.draw()
        inverse = fig.transFigure.inverted()
        renderer = fig.canvas.get_renderer()
        for index, (block, axes) in enumerate(zip(blocks, batch_rows)):
            boxes = [ax.get_window_extent().transformed(inverse) for ax in axes]
            top = max(box.y1 for box in boxes)
            if index == 0:
                titles = [
                    ax.title.get_window_extent(renderer).transformed(inverse).y1
                    for ax in axes
                    if ax.title.get_text()
                ]
                top = max(titles, default=top)
            at = top + 0.03 / fig.get_figheight()
            letter = "abcdefghijklmnopqrstuvwxyz"[index]
            fig.text(
                STACK_MARGIN_IN / fig.get_figwidth(),
                at,
                rf"$\mathbf{{{letter}}}$",
                ha="left",
                va="bottom",
                fontsize=STACK_TEXT_PT,
            )
            total = len(next(iter(block.embeddings.values())))
            fig.text(
                (boxes[0].x0 + boxes[-1].x1) / 2,
                at,
                rf"{block.name} ($n = {total:,}$)".replace(",", r"{,}"),
                ha="center",
                va="bottom",
                fontsize=STACK_TEXT_PT,
            )

    def _stack_braces(self, fig, batch_row, groups, panel_in: float) -> None:
        """Column-group braces beneath one block (measured from its batch row)."""
        fig.canvas.draw()
        inverse = fig.transFigure.inverted()
        height = fig.get_figheight()
        boxes = [ax.get_window_extent().transformed(inverse) for ax in batch_row]
        floor = min(box.y0 for box in boxes) - (panel_in + STACK_ROW_GAP_IN) / height
        rule = floor - self.style.qualitative_group_lift / height
        tick = self.style.qualitative_group_tick / height
        self._draw_braces(fig, boxes, groups, rule, tick)

    def _draw_braces(self, fig, boxes, groups, rule: float, tick: float) -> None:
        """One bracket per column group along ``rule`` (figure fraction), its name beneath."""
        height = fig.get_figheight()
        start = 0
        for name, count in groups:
            left, right = boxes[start].x0, boxes[start + count - 1].x1
            start += count
            fig.add_artist(
                Line2D(
                    [left, left, right, right],
                    [rule + tick, rule, rule, rule + tick],
                    transform=fig.transFigure,
                    color="#000000",
                    linewidth=0.6,
                    solid_joinstyle="miter",
                )
            )
            fig.text(
                (left + right) / 2,
                rule - self.style.qualitative_group_text_gap / height,
                name,
                ha="center",
                va="top",
                fontsize=self.style.qualitative_group_pt,
            )
