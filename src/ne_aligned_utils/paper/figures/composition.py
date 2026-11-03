"""Cell-type composition by batch, one block per dataset.

Left panel: cells per type, stacked by batch. Right panel: each type's share of its batch, the two
batches side by side. Types present in both batches come first, so a dataset with types only one
batch carries shows them together at the foot of its block.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ne_aligned_utils.paper import layout

TEXT_PT = 7.0
#: Height of one cell-type row, and how much of it the bar fills.
BAR_IN = 0.108
BAR_FRACTION = 0.78
ROW_TITLE_IN = 0.115
TICKS_IN = 0.135
ROW_GAP_IN = 0.07
HEADER_IN = 0.13
FOOTER_IN = 0.14
MARGIN_IN = 0.04
RIGHT_MARGIN_IN = 0.10
NAME_PAD_IN = 0.05
PANEL_GAP_IN = 0.30
#: Datasets per figure: seven blocks do not fit one page, so they are split by size.
SPLIT_AT = 4

BATCH_COLOURS: tuple[str, str] = ("#377eb8", "#e66101")
BATCH_LABELS: tuple[str, str] = ("Batch 0", "Batch 1")
COLUMN_TITLES = ("Cell counts by batch", "Cell-type proportions by batch")
AXIS_NAMES = ("Cells", "Fraction of batch")

#: Each axis ends on a tick, with 4 to 6 ticks, on the smallest round step that encloses the data.
TICK_RANGE = (4, 6)
TICK_STEPS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 8.0)
TICK_HEADROOM = 1.005
FRACTION_STEPS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.50)
#: Fixed proportion axes, where the automatic one would look misleadingly full.
FRACTION_LIMITS: dict[str, float] = {"nCoV": 0.30}
LEADER_COLOUR = "#bdbdbd"
LEADER_WIDTH = 0.3


@dataclass(frozen=True)
class CompositionBlock:
    """One dataset's cell counts: ``counts[batch, type]`` in drawing order."""

    key: str
    name: str
    names: np.ndarray
    counts: np.ndarray

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    @property
    def shared(self) -> np.ndarray:
        return (self.counts > 0).all(axis=0)


def composition_block(key: str, name: str, cache: Mapping) -> CompositionBlock:
    """Counts per (batch, cell type) from a benchmark aggregate cache, ordered for drawing.

    Shared types first, then one-batch types; within each, largest first, ties by name.
    """
    names = np.asarray([str(value) for value in np.asarray(cache["label_names"])])
    counts = np.stack(
        [
            np.bincount(np.asarray(cache[part][0], dtype=int), minlength=len(names))
            for part in ("labels0", "labels1")
        ]
    )
    present = counts.sum(axis=0) > 0
    names, counts = names[present], counts[:, present]
    shared = (counts > 0).all(axis=0)
    order = sorted(
        range(len(names)), key=lambda i: (not shared[i], -int(counts[:, i].sum()), names[i])
    )
    return CompositionBlock(key, name, names[order], counts[:, order])


def split_by_size(blocks: Sequence[CompositionBlock]) -> list[list[CompositionBlock]]:
    """Blocks smallest first, in figures of at most :data:`SPLIT_AT`."""
    ordered = sorted(blocks, key=lambda block: block.total)
    return [list(ordered[i : i + SPLIT_AT]) for i in range(0, len(ordered), SPLIT_AT)]


def name_column_width(blocks: Sequence[CompositionBlock]) -> float:
    """Inches of the cell-type name column, measured over every name so paired figures align."""
    scratch = plt.figure(figsize=(layout.FULL_WIDTH, 1.0))
    try:
        probes = [
            scratch.text(0.0, 0.0, str(value), fontsize=TEXT_PT)
            for block in blocks
            for value in block.names
        ]
        scratch.canvas.draw()
        renderer = scratch.canvas.get_renderer()
        widest = max(probe.get_window_extent(renderer).width / scratch.dpi for probe in probes)
    finally:
        plt.close(scratch)
    return MARGIN_IN + widest + NAME_PAD_IN


def _axis(value: float, candidates) -> tuple[float, np.ndarray]:
    """``(limit, ticks)``: the tightest axis ending on a tick that encloses ``value``."""
    target = max(value, 1e-12) * TICK_HEADROOM
    low, high = TICK_RANGE
    best = None
    for step in candidates:
        limit = float(np.ceil(target / step - 1e-9) * step)
        count = int(round(limit / step)) + 1
        if low <= count <= high and (best is None or (limit, count) < best[:2]):
            best = (limit, count, step)
    if best is None:
        step = max(candidates)
        limit = float(np.ceil(target / step - 1e-9) * step)
        return limit, np.arange(0.0, limit + step / 2, step)
    limit, count, step = best
    return limit, np.arange(count) * step


def _count_axis(value: float) -> tuple[float, np.ndarray]:
    return _axis(value, [m * 10.0**p for p in range(-2, 9) for m in TICK_STEPS])


def _fraction_axis(value: float, dataset: str) -> tuple[float, np.ndarray]:
    limit = FRACTION_LIMITS.get(dataset)
    if limit is None:
        return _axis(value, FRACTION_STEPS)
    if value > limit + 1e-12:
        raise ValueError(f"{dataset}'s largest share is {value:.4f}; a {limit} axis would clip it")
    low, high = TICK_RANGE
    step = max(
        s
        for s in FRACTION_STEPS
        if abs(limit / s - round(limit / s)) < 1e-9 and low <= round(limit / s) + 1 <= high
    )
    return limit, np.arange(round(limit / step) + 1) * step


def _style(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 2.0))
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linewidth=0.4, color="#d9d9d9")


def batch_composition(blocks: Sequence[CompositionBlock], *, name_column_in: float) -> Figure:
    """One figure of up to :data:`SPLIT_AT` dataset blocks, sized from its number of types."""
    rc = {
        key: TEXT_PT
        for key in (
            "xtick.labelsize",
            "ytick.labelsize",
            "legend.fontsize",
            "axes.titlesize",
            "axes.labelsize",
        )
    }
    with plt.rc_context(rc | {"figure.constrained_layout.use": False}):
        bars = sum(len(block.names) for block in blocks)
        height = (
            2 * MARGIN_IN
            + HEADER_IN
            + FOOTER_IN
            + len(blocks) * (ROW_TITLE_IN + TICKS_IN)
            + (len(blocks) - 1) * ROW_GAP_IN
            + bars * BAR_IN
        )
        fig = plt.figure(figsize=(layout.FULL_WIDTH, height))
        width_in, height_in = fig.get_figwidth(), fig.get_figheight()
        panel_in = (layout.FULL_WIDTH - name_column_in - RIGHT_MARGIN_IN - PANEL_GAP_IN) / 2

        top = 1.0 - (MARGIN_IN + HEADER_IN) / height_in
        rows = []
        for index, block in enumerate(blocks):
            top -= ROW_TITLE_IN / height_in
            bottom = top - len(block.names) * BAR_IN / height_in
            axes = [
                fig.add_axes(
                    [
                        (name_column_in + column * (panel_in + PANEL_GAP_IN)) / width_in,
                        bottom,
                        panel_in / width_in,
                        top - bottom,
                    ]
                )
                for column in range(2)
            ]
            _draw_block(axes, block, first=index == 0, last=index == len(blocks) - 1)
            _leaders(fig, axes, len(block.names))
            rows.append(axes)
            top = bottom - (TICKS_IN + ROW_GAP_IN) / height_in
        _headings(fig, blocks, rows)
    return fig


def _draw_block(axes, block: CompositionBlock, *, first: bool, last: bool) -> None:
    names, counts = block.names, block.counts
    y = np.arange(len(names))[::-1]
    fractions = counts / counts.sum(axis=1, keepdims=True)

    left, right = axes
    base = np.zeros(len(names))
    for batch in (0, 1):
        left.barh(
            y,
            counts[batch],
            height=BAR_FRACTION,
            left=base,
            color=BATCH_COLOURS[batch],
            linewidth=0,
        )
        base = base + counts[batch]
    left.set_yticks(y, labels=[str(value) for value in names], fontsize=TEXT_PT)

    offset = BAR_FRACTION / 4
    for batch in (0, 1):
        right.barh(
            y + (offset if batch == 0 else -offset),
            fractions[batch],
            height=BAR_FRACTION / 2,
            color=BATCH_COLOURS[batch],
            linewidth=0,
        )
    right.set_yticks(y, labels=[""] * len(names))

    for column, ax in enumerate(axes):
        ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
        if column == 1:
            limit, ticks = _fraction_axis(float(fractions.max()), block.key)
            ax.set_xticks(ticks, labels=[f"{t:.2f}" for t in ticks])
        else:
            limit, ticks = _count_axis(float(counts.sum(axis=0).max()))
            ax.set_xticks(ticks)
        ax.set_xlim(0, limit)
        ax.tick_params(labelsize=TEXT_PT)
        _style(ax)
        if last:
            ax.set_xlabel(AXIS_NAMES[column], fontsize=TEXT_PT)
        if first:
            ax.set_title(COLUMN_TITLES[column], fontsize=TEXT_PT, pad=3)


def _leaders(fig, axes, count: int) -> None:
    """A hairline per cell type across the gap between the two panels."""
    fig.canvas.draw()
    inverse = fig.transFigure.inverted()
    left, right = (ax.get_window_extent().transformed(inverse) for ax in axes)
    for row in range(count):
        _, y = inverse.transform(axes[0].transData.transform((0.0, row)))
        fig.add_artist(
            Line2D(
                [left.x1, right.x0],
                [y, y],
                transform=fig.transFigure,
                color=LEADER_COLOUR,
                linewidth=LEADER_WIDTH,
                zorder=0,
            )
        )


def _headings(fig, blocks, rows) -> None:
    """A letter, name and size above each block, and one batch key under the last."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    for index, (block, axes) in enumerate(zip(blocks, rows)):
        box = axes[0].get_window_extent().transformed(inverse)
        letter = "abcdefghijklmnopqrstuvwxyz"[index]
        fig.text(
            MARGIN_IN / fig.get_figwidth(),
            box.y1 + 0.025 / fig.get_figheight(),
            rf"$\mathbf{{{letter}}}$  {block.name} ($n = {block.total:,}$)".replace(",", r"{,}"),
            ha="left",
            va="bottom",
            fontsize=TEXT_PT,
        )
    handles = [
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markersize=3.6,
            markerfacecolor=colour,
            markeredgecolor="none",
        )
        for colour in BATCH_COLOURS
    ]
    labels = [ax.xaxis.label.get_window_extent(renderer).transformed(inverse) for ax in rows[-1]]
    fig.legend(
        handles,
        list(BATCH_LABELS),
        loc="center",
        bbox_to_anchor=((labels[0].x1 + labels[1].x0) / 2, (labels[0].y0 + labels[0].y1) / 2),
        ncol=2,
        frameon=False,
        fontsize=TEXT_PT,
        handletextpad=0.3,
        columnspacing=1.4,
        borderpad=0.0,
        borderaxespad=0.0,
    )
