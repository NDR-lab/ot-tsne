"""Build the final conference figures for the three PCA-tail experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from sklearn.decomposition import PCA

from ne_aligned_utils.paper import layout as paper_layout, registry
from ne_aligned_utils.paper.inputs import synthetic
from ne_aligned_utils.paper.inputs.synthetic import reconstruct_corruptions
from ne_aligned_utils.paper.rendering import metrics as metric_drawing

REL_REG = 1.0
TOP_K = 15
DISPLAY_MAGNITUDE = 10.0
#: Digits shown in order, avoiding selection based on visual appearance.
DISPLAY_DIGITS = tuple(range(10))

#: Construction-panel geometry in inches; explicit placement preserves square image cells.
CONSTRUCTION_GAP = 0.012
#: Additional separation between the batch-0 result and the clean source for batch 1.
CONSTRUCTION_BATCH_GAP = 0.06

#: Minimum distance from panel ink to the figure edge, in inches.
FIGURE_MARGIN_IN = 0.04

#: Left edge aligned with the metric panels' visible ink rather than their axes spines.
CONSTRUCTION_LEFT_IN = FIGURE_MARGIN_IN

#: Extra width around construction-arrow columns, in inches.
CONSTRUCTION_ARROW_PAD_IN = 0.06

#: Number of digit rows in the construction block.
CONSTRUCTION_ROWS = 4

# --- panel A, strip layout -------------------------------------------------------------------
# Each strip pairs clean digits with the transformed batch below them.

#: Digits per construction strip.
STRIP_COLUMNS = 4
#: Gap between a clean row and its transformed row.
STRIP_ROW_GAP_IN = 0.022
#: Gap between construction blocks.
STRIP_BLOCK_GAP_IN = 0.13
#: Row name to the first image, and last image to the arrow that names the step.
STRIP_LABEL_PAD_IN = 0.055
STRIP_ARROW_PAD_IN = 0.07
#: The arrow itself, and the gap between it and the note beside it.
STRIP_ARROW_IN = 0.15
STRIP_NOTE_PAD_IN = 0.05
#: Maximum gap between gallery rows in the strip layout.
STRIP_GALLERY_ROW_GAP_IN = 0.10

#: Maximum digit side length in the strip layout, in inches.
STRIP_MAX_SIDE_IN = 0.26

#: Figure geometry for the strip layout.
STRIP_FIGURE_IN = 3.3
STRIP_HEIGHT_RATIOS = (1.91, 1.0)
STRIP_HSPACE = 0.125
#: Gallery margin and spacing for the strip layout.
STRIP_GALLERY_RIGHT_IN = 0.15
STRIP_GALLERY_WSPACE = 0.07

# --- panel A, inline flavour -----------------------------------------------------------------
# The inline layout places each transformation label between its paired rows.

#: Space for the arrow and label between paired rows.
STRIP_STEP_GAP_IN = 0.18
#: The arrow drawn in that gap, and its distance from the note beside it.
STRIP_INLINE_ARROW_IN = 0.13
STRIP_INLINE_NOTE_PAD_IN = 0.05
#: Figure geometry for the inline layout.
STRIP_INLINE_FIGURE_IN = 3.53
STRIP_INLINE_HEIGHT_RATIOS = (2.375, 1.0)
#: Vertical gap between the top and metric rows.
STRIP_INLINE_HSPACE = 0.164
#: Right margin of the inline gallery.
STRIP_INLINE_GALLERY_RIGHT_IN = FIGURE_MARGIN_IN
#: Wide enough for those names plus their label pad.
STRIP_INLINE_BLOCK_GAP_IN = 0.26
#: Arrow-head size for the inline layout.
STRIP_INLINE_ARROW_HEAD_PT = 5

#: Space above panel A's first row, in inches.
STRIP_INLINE_HEADER_IN = 0.03

#: Column width below which the widest header no longer fits on one line, in inches. Measured:
#: "+lambda alpha_c = 0" is 0.36 in at 7 pt.
CONSTRUCTION_NARROW_IN = 0.38
CONSTRUCTION_TITLE_IN = 0.30
#: Reserved above the first image row for one-line construction headers.
CONSTRUCTION_HEADER_IN = 0.16

#: Legend column counts. The narrow layout has no room for the two-column keys the wide one takes.
NARROW_LEGEND_NCOL = 1
WIDE_LEGEND_NCOL = 2

#: Shared type size for metric-panel text.
PANEL_TITLE_PT = 7.0

#: Common ticks and padding for bounded metric axes.
METRIC_TICKS: tuple[float, ...] = (0.0, 0.5, 1.0)
METRIC_PAD = 0.085

#: Between a panel's letter and a title set beside it, in inches. Only reached when the title is
#: too wide to centre on the panel without printing through the letter.
HEADING_LETTER_GAP_IN = 0.055

#: Smallest gap between two metric panels' ink, in inches.
PANEL_CLEARANCE_IN = 0.09

#: Marker radius plus edge and clearance, in points.
MARKER_CLEARANCE_PT = 5.0 / 2 + 0.4 + 0.5


def metric_axes(fig) -> list:
    """Return the metric-row axes managed by subplot parameters."""
    return [ax for ax in fig.axes if ax._left_title.get_text()]


def panel_ink_span(ax, renderer, inverse) -> tuple[float, float]:
    """Return the horizontal span of a metric panel, excluding its title and legend."""
    boxes = [ax.get_window_extent(renderer)]
    for artist in (ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()):
        if artist.get_text().strip() and artist.get_visible():
            boxes.append(artist.get_window_extent(renderer))
    span = Bbox.union(boxes).transformed(inverse)
    return span.x0, span.x1


def fit_figure_margins(fig, *, margin_in: float = FIGURE_MARGIN_IN, passes: int = 4) -> None:
    """Fit metric-row labels within the figure margin."""
    panels = metric_axes(fig)
    if not panels:
        return
    margin_x = margin_in / fig.get_figwidth()
    margin_y = margin_in / fig.get_figheight()
    for _ in range(passes):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inverse = fig.transFigure.inverted()
        spans = [panel_ink_span(ax, renderer, inverse) for ax in panels]
        bottom_ink = min(ax.get_tightbbox(renderer).transformed(inverse).y0 for ax in panels)
        slack_left = min(left for left, _ in spans) - margin_x
        slack_right = (1.0 - margin_x) - max(right for _, right in spans)
        slack_bottom = bottom_ink - margin_y
        if max(abs(slack_left), abs(slack_right), abs(slack_bottom)) < 5e-4:
            return
        pars = fig.subplotpars
        fig.subplots_adjust(
            left=max(0.0, pars.left - slack_left),
            right=min(1.0, pars.right + slack_right),
            bottom=max(0.0, pars.bottom - slack_bottom),
        )
    fig.canvas.draw()


def marker_lines(ax) -> list:
    """Return plotted curves without error-bar caps."""
    return [
        line
        for line in ax.lines
        if line.get_marker() not in ("", " ", "None", None)
        and line.get_linestyle() not in ("", " ", "None", None)
    ]


def clearance_pad(span_pt: float, data_span: float, clearance_pt: float) -> float:
    """Convert a fixed point clearance to padding in data units."""
    room = span_pt - 2 * clearance_pt
    return clearance_pt * data_span / room if room > 0 else 0.0


def clear_markers(fig, *, clearance_pt: float = MARKER_CLEARANCE_PT) -> None:
    """Expand metric limits so markers remain visible and y scales stay aligned."""
    panels = [(ax, marker_lines(ax)) for ax in metric_axes(fig)]
    panels = [(ax, lines) for ax, lines in panels if lines]
    if not panels:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    grow = 0.0
    for ax, lines in panels:
        box = ax.get_window_extent(renderer)
        width_pt, height_pt = (box.width * 72.0 / fig.dpi, box.height * 72.0 / fig.dpi)
        xs = np.concatenate([np.asarray(line.get_xdata(), float) for line in lines])
        ys = np.concatenate([np.asarray(line.get_ydata(), float) for line in lines])
        pad_x = clearance_pad(width_pt, xs.max() - xs.min(), clearance_pt)
        pad_y = clearance_pad(height_pt, ys.max() - ys.min(), clearance_pt)
        x_low, x_high = ax.get_xlim()
        ax.set_xlim(min(x_low, xs.min() - pad_x), max(x_high, xs.max() + pad_x))
        y_low, y_high = ax.get_ylim()
        grow = max(grow, y_low - (ys.min() - pad_y), (ys.max() + pad_y) - y_high)
    if grow > 0:
        for ax, _ in panels:
            y_low, y_high = ax.get_ylim()
            ax.set_ylim(y_low - grow, y_high + grow)


def lay_out_panels(fig) -> None:
    """Lay out metric panels and align their headings."""
    for _ in range(3):
        clear_markers(fig)
        fit_figure_margins(fig)
    align_panel_titles(fig)


def align_panel_titles(fig) -> None:
    """Align panel letters to ink edges and titles to plot centers."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    clear = HEADING_LETTER_GAP_IN / fig.get_figwidth()

    def targets(letter, title, span, rect):
        """Return non-overlapping positions for a panel letter and title."""
        after = span[0]
        if letter is not None:
            after += letter.get_window_extent(renderer).transformed(inverse).width + clear
        if title is None:
            return span[0], None
        width = title.get_window_extent(renderer).transformed(inverse).width
        return span[0], max((rect[0] + rect[1] - width) / 2, after)

    def shift(text, target: float, scale: float = 1.0) -> None:
        drawn = text.get_window_extent(renderer).transformed(inverse)
        text.set_x(text.get_position()[0] + (target - drawn.x0) / scale)

    # Panels A and B record their own spans because they do not fill their grid cells.
    spanned: dict[tuple[float, float], list] = {}
    for text in fig.texts:
        span = getattr(text, "_panel_span", None)
        if span is not None:
            spanned.setdefault(span, []).append(text)
    for span, texts in spanned.items():
        letter = next((t for t in texts if t.get_ha() == "left"), None)
        title = next((t for t in texts if t.get_ha() != "left"), None)
        rect = getattr(texts[0], "_panel_rect", span)
        letter_at, title_at = targets(letter, title, span, rect)
        if letter is not None:
            shift(letter, letter_at)
        if title is not None:
            shift(title, title_at)

    for ax in metric_axes(fig):
        box = ax.get_position()
        letter = ax._left_title if ax._left_title.get_text() else None
        title = ax.title if ax.title.get_text() else None
        letter_at, title_at = targets(
            letter, title, panel_ink_span(ax, renderer, inverse), (box.x0, box.x1)
        )
        width = box.width
        if letter is not None:
            shift(letter, letter_at, width)
        if title is not None:
            shift(title, title_at, width)


def set_panel_title(
    ax, text: str, *, fontsize: float, letter: str = "", centred: bool = False
) -> None:
    """Set a panel heading, optionally separating its letter and title."""
    if not centred:
        ax.set_title(text, loc="left", x=0.0, fontsize=fontsize)
        return
    ax.set_title(rf"$\mathbf{{{letter.lower()}}}$", loc="left", x=0.0, fontsize=fontsize)
    ax.set_title(text, loc="center", fontsize=fontsize)


#: Muted OT brown for paired curves, distinct from the orange diagonal-variant identity.
OT_LIGHT = "#836557"
PANEL_LEGEND_PT = 7.0

#: Construction column-header size in points.
CONSTRUCTION_HEADER_PT = 7.0

#: Column headings in the visualization gallery, in points.
GALLERY_TITLE_PT = 7.0

#: Gap between the gallery panels, as a fraction of a panel.
GALLERY_WSPACE = 0.12

#: Vertical gap between the two gallery rows, as a fraction of a panel. Larger than the horizontal
#: one because a column title sits in it.
GALLERY_HSPACE = 0.0

#: Height reserved above the gallery's first row for its column titles, in inches.
GALLERY_TITLE_IN = 0.12

#: Marker area in a gallery panel, in points squared, and the panel width it was chosen at.
GALLERY_POINT_PT2 = 0.7
GALLERY_POINT_REF_IN = 0.58

#: Between a gallery panel and its row name, in points.
GALLERY_LABEL_PAD_PT = 3

#: Islands named in the unintegrated panel: one per digit for this many digits, and one per batch.
GALLERY_DIGIT_NAMES = 10
#: How far outside an island its name sits, as a fraction of a line of type.
GALLERY_NAME_PADS = (0.6, 0.55, 0.5, 0.45)
#: How much nearer its own island a name has to be than to any other, before it is kept.
GALLERY_NAME_MARGIN = 1.15
#: Among directions with nearly as much room as the best one, how near the best a direction has to
#: be to still be considered.
GALLERY_NAME_ROOM_TOLERANCE = 0.9
#: How far off straight-outward a direction may point and still count as outward: 0.5 is 60 degrees,
#: so "out and a bit below" qualifies and "back into the cloud" does not.
GALLERY_NAME_OUTWARD_MIN = 0.5
#: The same test against due right, which is tried first.
GALLERY_NAME_RIGHT_MIN = 0.5

# Small optical shift that aligns the gallery's method-name baseline with the construction-stage
# headers beside it. The A/B panel headings remain on their shared grid-row coordinate.
GALLERY_DOWN_IN = 0.04

#: Gap between the construction block and the gallery beside it, in inches.
GALLERY_BLOCK_GAP = 0.14

#: Margin reserved to the right of the gallery for its Batch/Digit labels, in inches.
GALLERY_RIGHT_IN = 0.214

#: The two batch colors, as the cluster galleries draw them.
GALLERY_BATCH_COLOURS: tuple[str, str] = ("#377eb8", "#e66101")

DISPLAY_NAME = {
    "one_sided": "One-sided nuisance",
    "two_sided_correlated_scales": "Two-sided, shared scale schedule",
    "two_sided_independent_scales": "Two-sided, independent scale assignment",
}


def load_mnist_and_pca(
    manifest: dict,
    *,
    source_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PCA]:
    """Reproduce the notebooks' disjoint, exactly class-balanced split."""
    if manifest.get("batch_sampling") != "disjoint_stratified_exact_balance_v1":
        raise ValueError(
            "The selected run used paired clean images; compatible completed "
            "disjoint_stratified_exact_balance_v1 inputs are required."
        )
    from ne_aligned_utils.experiments.synthetic import sample_mnist

    x0, labels0, x1, labels1, _, _ = sample_mnist(
        synthetic.mnist_directory(source_root),
        int(manifest["n_per_digit_per_batch"]),
        int(manifest["sampling_seed"]),
    )
    pca = PCA(n_components=None, svd_solver="full").fit(np.vstack([x0, x1]))
    return x0, labels0, x1, labels1, pca


def mean_sd(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return frame.groupby("magnitude")[metric].agg(mean="mean", sd="std").reset_index()


def add_curve(ax, frame, metric, *, label, color, marker, linestyle="-", seen=None):
    table = mean_sd(frame, metric)
    ax.errorbar(
        table.magnitude,
        table["mean"],
        yerr=table.sd.fillna(0),
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        linewidth=0.9,
        markersize=5,
        capsize=1.5,
    )
    if seen is not None:
        seen.extend(span_values(table["mean"], table.sd))
    return table


def span_values(mean, sd) -> list[float]:
    """Every value an axis has to enclose: each point and both ends of its clipped error bar."""
    mean = np.asarray(mean, dtype=float)
    sd = np.nan_to_num(np.asarray(sd, dtype=float))
    low, high = 0.0, 1.0
    return [*mean, *np.clip(mean - sd, low, None), *np.clip(mean + sd, None, high)]


def fit_axes(ax, xs, ys, *, xlabel, ylabel) -> None:
    """Ticks rounded outwards to enclose the data, as the benchmark panels place theirs."""
    xs = np.asarray(xs, float)
    metric_drawing.round_three_ticks(
        ax, xs, np.asarray(ys, float), step_x=10.0, step_y=0.1, n_x=3, n_y=3
    )
    # Show the actual tested magnitudes.
    x_ticks = np.unique(xs)
    ax.set_xticks(x_ticks, labels=[f"{value:g}" for value in x_ticks])
    if len(x_ticks) > 1:
        pad = 0.025 * (x_ticks[-1] - x_ticks[0])
        ax.set_xlim(x_ticks[0] - pad, x_ticks[-1] + pad)
    # The full unit interval on every panel.
    ax.set_yticks(METRIC_TICKS, labels=[f"{value:.1f}" for value in METRIC_TICKS])
    ax.set_ylim(-METRIC_PAD, 1.0 + METRIC_PAD)
    ax.set_xlabel(xlabel, fontsize=PANEL_TITLE_PT)
    ax.set_ylabel(ylabel, fontsize=PANEL_TITLE_PT)
    paper_layout.style_axes(ax)


def clean_legend(ax, *, loc="best", ncol=1, fontsize=None, bbox_to_anchor=None):
    handles, labels = ax.get_legend_handles_labels()
    simple = [
        Line2D(
            [],
            [],
            color=h.lines[0].get_color(),
            marker=h.lines[0].get_marker(),
            linestyle=h.lines[0].get_linestyle(),
            linewidth=0.9,
            markersize=5,
        )
        if hasattr(h, "lines")
        else h
        for h in handles
    ]
    # A backing panel rather than a transparent key.
    return ax.legend(
        simple,
        labels,
        loc=loc,
        ncol=ncol,
        bbox_to_anchor=bbox_to_anchor,
        fontsize=PANEL_LEGEND_PT if fontsize is None else fontsize,
        handletextpad=0.4,
        columnspacing=0.8,
        labelspacing=0.25,
        borderaxespad=0.2,
        frameon=True,
        framealpha=0.82,
        edgecolor="none",
        borderpad=0.2,
    )


def construction_stages(variant: str, narrow: bool = False) -> tuple[tuple[str, str], ...]:
    """``(kind, header)`` per column, where ``kind`` names what the cell draws."""
    return (
        ("clean0", "Clean"),
        ("alpha", ""),
        ("batch0", "Batch 0"),
        ("clean1", "Clean"),
        ("beta", ""),
        ("batch1", "Batch 1"),
    )


def construction_rect(fig, cell, n_cols: int, want_rows=None, n_arrows: int = 0):
    """``(x0, y1, width, height, rows, side)`` for a construction block in ``cell``."""
    box = cell.get_position(fig)
    x0 = CONSTRUCTION_LEFT_IN / fig.get_figwidth()
    width_in = (box.x1 - x0) * fig.get_figwidth()
    arrow_in = 2 * n_arrows * CONSTRUCTION_ARROW_PAD_IN
    # Bounded by the cell in *both* directions.
    available = box.height * fig.get_figheight() - CONSTRUCTION_HEADER_IN
    # Bounded by the height as well as the width, so widening the block cannot silently cost it a
    # row: at three rows the tallest square the cell allows is `available / 3`.
    rows = min(len(DISPLAY_DIGITS), CONSTRUCTION_ROWS if want_rows is None else want_rows)
    side_in = min(
        (width_in - (n_cols - 1) * CONSTRUCTION_GAP - CONSTRUCTION_BATCH_GAP - arrow_in) / n_cols,
        (available - (rows - 1) * CONSTRUCTION_GAP) / rows,
    )
    block_in = (
        n_cols * side_in + (n_cols - 1) * CONSTRUCTION_GAP + CONSTRUCTION_BATCH_GAP + arrow_in
    )
    height_in = rows * side_in + (rows - 1) * CONSTRUCTION_GAP + CONSTRUCTION_HEADER_IN
    return (
        x0,
        box.y1,
        block_in / fig.get_figwidth(),
        height_in / fig.get_figheight(),
        rows,
        side_in,
    )


def title_lift(fig) -> float:
    """Where an axes title sits above its cell, in figure fractions."""
    return 0.5 * float(plt.rcParams["axes.titlepad"]) / 72.0 / fig.get_figheight()


def gallery_title_lift(fig) -> float:
    """How far the gallery heading sits above its cell, in figure fractions."""
    # Reserve the measured height of one column-title line.
    return title_lift(fig)


def construction_panel(
    fig,
    cell,
    clean0,
    labels0,
    clean1,
    labels1,
    alpha,
    beta,
    *,
    variant,
    letter="A",
    lift=None,
    rows=None,
    centred=False,
) -> tuple[float, float, float]:
    """Panel A: separate clean sources and corruptions for both disjoint batches."""
    stages = construction_stages(variant)
    arrow_columns = sum(1 for kind, _ in stages if kind in ("alpha", "beta"))
    x0, y1, width, height, rows, side_in = construction_rect(
        fig, cell, len(stages), rows, n_arrows=arrow_columns
    )
    # A column is the block's width over all construction stages. More digit rows make it narrower
    # than the widest header, so that header wraps rather than running into its neighbor.
    column_in = width * fig.get_figwidth() / len(stages)
    stages = construction_stages(variant, narrow=column_in < CONSTRUCTION_NARROW_IN)
    digits = DISPLAY_DIGITS[:rows]
    gap_x = CONSTRUCTION_GAP / fig.get_figwidth()
    gap_y = CONSTRUCTION_GAP / fig.get_figheight()
    side_x = side_in / fig.get_figwidth()
    side_y = side_in / fig.get_figheight()
    batch_gap_x = CONSTRUCTION_BATCH_GAP / fig.get_figwidth()
    arrow_x = side_x + 2 * CONSTRUCTION_ARROW_PAD_IN / fig.get_figwidth()
    top = y1 - CONSTRUCTION_HEADER_IN / fig.get_figheight()

    def column_width(col: int) -> float:
        return arrow_x if stages[col][0] in ("alpha", "beta") else side_x

    def column_left(col: int) -> float:
        # Walked rather than multiplied out: the arrow columns are wider than the image ones, so
        # the nth column no longer starts at n times anything.
        left = x0 + sum(column_width(index) + gap_x for index in range(col))
        return left + (batch_gap_x if col >= 3 else 0.0)

    representatives0 = [np.flatnonzero(labels0 == digit)[0] for digit in digits]
    representatives1 = [np.flatnonzero(labels1 == digit)[0] for digit in digits]
    for row, (digit, index0, index1) in enumerate(zip(digits, representatives0, representatives1)):
        drawn = {
            "clean0": clean0[index0],
            "alpha": DISPLAY_MAGNITUDE * alpha[digit],
            "batch0": clean0[index0] + DISPLAY_MAGNITUDE * alpha[digit],
            "clean1": clean1[index1],
            "beta": DISPLAY_MAGNITUDE * beta[digit],
            "batch1": clean1[index1] + DISPLAY_MAGNITUDE * beta[digit],
        }
        for col, (kind, header) in enumerate(stages):
            if kind in ("alpha", "beta"):
                continue
            value = drawn[kind]
            ax = fig.add_axes(
                [
                    column_left(col),
                    top - (row + 1) * side_y - row * gap_y,
                    side_x,
                    side_y,
                ]
            )
            ax.imshow(np.clip(value, 0, 1).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                # A left-anchored title is measured from the axes' left spine, not the cell's edge,
                # so what a title has to fit in is the panel minus its y labels -- about 1.02 in of
                # the 1.47 in cell here.
                ax.set_title(
                    header, pad=plt.rcParams["axes.titlepad"], fontsize=CONSTRUCTION_HEADER_PT
                )
            # Digit images identify their own rows.

    # Each batch construction is a single transformation, so one labelled arrow spans the digit
    # examples rather than repeating the same arrow in every row.
    image_block_height = rows * side_y + (rows - 1) * gap_y
    image_block_bottom = top - image_block_height
    for col, (kind, _) in enumerate(stages):
        if kind not in ("alpha", "beta"):
            continue
        batch = 0 if kind == "alpha" else 1
        suffix = "=0" if variant == "one_sided" and batch == 0 else ""
        arrow_ax = fig.add_axes(
            [
                column_left(col),
                image_block_bottom,
                arrow_x,
                image_block_height,
            ]
        )
        arrow_ax.set_xlim(0, 1)
        arrow_ax.set_ylim(0, 1)
        arrow_ax.axis("off")
        arrow_ax.annotate(
            "",
            xy=(0.90, 0.50),
            xytext=(0.10, 0.50),
            arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.0, "mutation_scale": 8},
        )
        arrow_ax.text(
            0.5,
            0.535,
            "$10\\times$\n"
            rf"$\mathrm{{noise}}_{{\mathrm{{class}},{batch}}}{suffix}$",
            ha="center",
            va="bottom",
            fontsize=PANEL_TITLE_PT,
            linespacing=0.82,
        )
    at = y1 + (title_lift(fig) if lift is None else lift)
    # Centred over the block, with the letter left on its edge.
    span = (x0, x0 + width)
    if centred:
        title = fig.text(
            x0 + width / 2,
            at,
            f"Construction of batches ($\\lambda={DISPLAY_MAGNITUDE:g}$)",
            ha="center",
            va="bottom",
            fontsize=PANEL_TITLE_PT,
        )
        title._panel_span = span
    heading = fig.text(
        x0,
        at,
        # State the displayed magnitude explicitly alongside lambda-labelled columns.
        rf"$\mathbf{{{letter.lower()}}}$"
        if centred
        else rf"$\mathbf{{{letter.lower()}}}$  Construction of batches "
        rf"($\lambda={DISPLAY_MAGNITUDE:g}$)",
        ha="left",
        va="bottom",
        fontsize=PANEL_TITLE_PT,
    )
    heading._panel_span = span
    # ``(right edge, block bottom, block top)`` so the gallery can sit on exactly the same lines.
    return x0 + width, top - rows * side_y - (rows - 1) * gap_y, top


def text_height_in(fig, string: str, fontsize: float) -> float:
    """Inches ``string`` occupies across, once turned on its side. See :func:`text_width_in`."""
    probe = fig.text(0.0, 0.0, string, fontsize=fontsize)
    fig.canvas.draw()
    height = probe.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
    probe.remove()
    return height


def text_width_in(fig, string: str, fontsize: float) -> float:
    """Inches ``string`` occupies at ``fontsize``, measured on a throwaway artist."""
    probe = fig.text(0.0, 0.0, string, fontsize=fontsize)
    fig.canvas.draw()
    width = probe.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    probe.remove()
    return width


def construction_panel_strip(
    fig,
    cell,
    clean0,
    labels0,
    clean1,
    labels1,
    alpha,
    beta,
    *,
    variant,
    letter="A",
    lift=None,
    columns=STRIP_COLUMNS,
    centred=False,
    inline=False,
) -> tuple[float, float, float]:
    """Panel A: clean digits above their transformed batch, once per batch."""
    row_labels = ("Clean", "Batch 0", "Clean", "Batch 1")
    # One line inline, two beside: in the gap between two image rows the note has a row's width to
    # run in and only a line's height; beside the block it has the opposite.
    notes = tuple(
        rf"$+{DISPLAY_MAGNITUDE:g}\times\,"
        rf"\mathrm{{noise}}_{{\mathrm{{class}},{batch}}}$"
        if inline
        else f"$+{DISPLAY_MAGNITUDE:g}\\times$\n"
        rf"$\mathrm{{noise}}_{{\mathrm{{class}},{batch}}}$"
        for batch in (0, 1)
    )
    box = cell.get_position(fig)
    x0 = CONSTRUCTION_LEFT_IN / fig.get_figwidth()
    # Turned on its side a name is a line tall, not a word wide. That is the whole saving, and it
    # is why panel B's own row names are set the same way.
    label_in = max(
        (text_height_in if inline else text_width_in)(fig, text, CONSTRUCTION_HEADER_PT)
        for text in row_labels
    )
    note_in = max(
        max(text_width_in(fig, line, PANEL_TITLE_PT) for line in note.split("\n")) for note in notes
    )
    left_in = label_in + STRIP_LABEL_PAD_IN
    right_in = (
        0.0 if inline else (STRIP_ARROW_PAD_IN + STRIP_ARROW_IN + STRIP_NOTE_PAD_IN + note_in)
    )
    step_gap_in = STRIP_STEP_GAP_IN if inline else STRIP_ROW_GAP_IN
    # Beside flavour: the same header strip the column layout reserves, even though nothing of panel
    # A sits in it.
    header_in = STRIP_INLINE_HEADER_IN if inline else CONSTRUCTION_HEADER_IN
    available_in = box.height * fig.get_figheight() - header_in
    strip_in = (
        (box.x1 - x0) * fig.get_figwidth() - left_in - right_in - (columns - 1) * CONSTRUCTION_GAP
    )
    stack_in = available_in - 2 * step_gap_in - STRIP_BLOCK_GAP_IN
    side_in = min(strip_in / columns, stack_in / 4)
    if not inline:
        side_in = min(side_in, STRIP_MAX_SIDE_IN)
    side_x, side_y = side_in / fig.get_figwidth(), side_in / fig.get_figheight()
    gap_x = CONSTRUCTION_GAP / fig.get_figwidth()
    row_gap_y = step_gap_in / fig.get_figheight()
    block_gap_y = STRIP_BLOCK_GAP_IN / fig.get_figheight()
    images_x = x0 + left_in / fig.get_figwidth()
    strip_x = columns * side_x + (columns - 1) * gap_x
    top = box.y1 - header_in / fig.get_figheight()

    digits = DISPLAY_DIGITS[:columns]
    index0 = [int(np.flatnonzero(labels0 == digit)[0]) for digit in digits]
    index1 = [int(np.flatnonzero(labels1 == digit)[0]) for digit in digits]
    rows = (
        [clean0[i] for i in index0],
        [clean0[i] + DISPLAY_MAGNITUDE * alpha[d] for i, d in zip(index0, digits)],
        [clean1[i] for i in index1],
        [clean1[i] + DISPLAY_MAGNITUDE * beta[d] for i, d in zip(index1, digits)],
    )
    row_top = []
    y = top
    for row in range(4):
        row_top.append(y)
        y -= side_y + (block_gap_y if row == 1 else row_gap_y)

    for row, (images, name) in enumerate(zip(rows, row_labels)):
        for column, value in enumerate(images):
            ax = fig.add_axes(
                [
                    images_x + column * (side_x + gap_x),
                    row_top[row] - side_y,
                    side_x,
                    side_y,
                ]
            )
            ax.imshow(np.clip(value, 0, 1).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        if inline:
            fig.text(
                x0 + label_in / fig.get_figwidth() / 2,
                row_top[row] - side_y / 2,
                name,
                ha="center",
                va="center",
                rotation=90,
                fontsize=CONSTRUCTION_HEADER_PT,
            )
        else:
            fig.text(
                images_x - STRIP_LABEL_PAD_IN / fig.get_figwidth(),
                row_top[row] - side_y / 2,
                name,
                ha="right",
                va="center",
                fontsize=CONSTRUCTION_HEADER_PT,
            )

    # One arrow per block, pointing from the clean row into the batch row, with the step named
    # beside it.
    for block, note in enumerate(notes):
        block_top, block_bottom = row_top[2 * block], row_top[2 * block + 1] - side_y
        if inline:
            # In the gap the two rows leave, and centered on the strip: the step applies to the
            # whole row, so it is named over the whole row rather than off one end of it.
            group_in = STRIP_INLINE_ARROW_IN + STRIP_INLINE_NOTE_PAD_IN + note_in
            group_x = images_x + (strip_x - group_in / fig.get_figwidth()) / 2
            middle = row_top[2 * block] - side_y - (step_gap_in / fig.get_figheight()) / 2
            arrow_ax = fig.add_axes(
                [
                    group_x,
                    middle - (step_gap_in / 2) / fig.get_figheight(),
                    STRIP_INLINE_ARROW_IN / fig.get_figwidth(),
                    step_gap_in / fig.get_figheight(),
                ]
            )
            note_x = group_x + (
                (STRIP_INLINE_ARROW_IN + STRIP_INLINE_NOTE_PAD_IN) / fig.get_figwidth()
            )
            note_y = middle
        else:
            arrow_ax = fig.add_axes(
                [
                    images_x + strip_x + STRIP_ARROW_PAD_IN / fig.get_figwidth(),
                    block_bottom,
                    STRIP_ARROW_IN / fig.get_figwidth(),
                    block_top - block_bottom,
                ]
            )
            note_x = (
                images_x
                + strip_x
                + (STRIP_ARROW_PAD_IN + STRIP_ARROW_IN + STRIP_NOTE_PAD_IN) / fig.get_figwidth()
            )
            note_y = (block_top + block_bottom) / 2
        arrow_ax.set_xlim(0, 1)
        arrow_ax.set_ylim(0, 1)
        arrow_ax.axis("off")
        arrow_ax.annotate(
            "",
            xy=(0.5, 0.10),
            xytext=(0.5, 0.90),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "black",
                "lw": 1.0,
                "shrinkA": 0,
                "shrinkB": 0,
                "mutation_scale": STRIP_INLINE_ARROW_HEAD_PT if inline else 8,
            },
        )
        fig.text(
            note_x, note_y, note, ha="left", va="center", fontsize=PANEL_TITLE_PT, linespacing=0.9
        )

    width = (left_in + right_in) / fig.get_figwidth() + strip_x
    at = box.y1 + (title_lift(fig) if lift is None else lift)
    span = (x0, x0 + width)
    rect = (images_x, images_x + strip_x)
    if centred:
        title = fig.text(
            (rect[0] + rect[1]) / 2,
            at,
            f"Construction of batches ($\\lambda={DISPLAY_MAGNITUDE:g}$)",
            ha="center",
            va="bottom",
            fontsize=PANEL_TITLE_PT,
        )
        title._panel_span, title._panel_rect = span, rect
    heading = fig.text(
        x0,
        at,
        rf"$\mathbf{{{letter.lower()}}}$"
        if centred
        else rf"$\mathbf{{{letter.lower()}}}$  Construction of batches "
        rf"($\lambda={DISPLAY_MAGNITUDE:g}$)",
        ha="left",
        va="bottom",
        fontsize=PANEL_TITLE_PT,
    )
    heading._panel_span, heading._panel_rect = span, rect
    return x0 + width, row_top[-1] - side_y, top


def select_downstream(summary: pd.DataFrame) -> pd.DataFrame:
    ot = summary[
        summary.display_method.str.startswith("OT-tSNE")
        & (summary.epsilon_policy == "adaptive_per_magnitude")
        & np.isclose(summary.rel_reg, REL_REG)
    ]
    scanorama = summary[summary.display_method.str.startswith("Scanorama")]
    if "method_id" in summary:
        native = summary.method_id.str.startswith("multimap__native-defaults__")
        multimap = summary[
            (summary.method == "multimap")
            & (native | ((summary.effective_k == 30) & (summary.preprocessing == "default")))
        ]
    else:
        multimap = summary[summary.display_method.eq("MultiMAP (k=30, default)")]
    selected = pd.concat([ot, multimap, scanorama], ignore_index=True)
    if selected.display_method.nunique() != 3:
        raise ValueError(
            f"expected adaptive OT-tSNE at rel_reg=1, the selected MultiMAP profile, "
            f"and Scanorama; got "
            f"{selected.display_method.unique().tolist()}"
        )
    return selected


def mechanism_panels(
    fig,
    specs,
    variant,
    runs,
    directed,
    *,
    letters=("B", "C"),
    ncol=1,
    titles=("Directed graph purity", "Symmetrized graph purity"),
    title_pt=None,
    directed_panel=True,
    legend_outside=False,
    ylabel="Neighbor purity",
    centred=False,
):
    # Two shades per family, one marker shape per family.
    ot = registry.resolve("ot")
    dark, light = registry.resolve("joint-pca").color, registry.resolve("joint").color
    display_magnitudes = {float(value) for value in runs.configs["original"]["magnitudes"]}
    directed = directed[directed.magnitude.astype(float).isin(display_magnitudes)]
    knn = runs.knn.query("mode == 'original'")
    knn = knn[knn.magnitude.astype(float).isin(display_magnitudes)]
    purity_fields = (
        ("forward_correct_class_neighbor_fraction", "backward_correct_class_neighbor_fraction")
        if variant == "one_sided"
        else (
            "batch0_to_batch1_correct_class_neighbor_fraction",
            "batch1_to_batch0_correct_class_neighbor_fraction",
        )
    )

    if not directed_panel:
        # Only the symmetrized panel, and it takes the first spec and the first letter.
        specs, letters, titles = (specs[0],), (letters[0],) * 2, (titles[1],) * 2
    else:
        ax_b = fig.add_subplot(specs[0])
        seen = []
        table = add_curve(
            ax_b,
            directed,
            "eot_row_digit_purity",
            label=r"EOT $X_0\!\to X_1$",
            color=ot.color,
            marker=ot.marker,
            seen=seen,
        )
        add_curve(
            ax_b,
            directed,
            "eot_column_digit_purity",
            label=r"EOT $X_1\!\to X_0$",
            color=OT_LIGHT,
            marker=ot.marker,
            seen=seen,
        )
        add_curve(
            ax_b,
            knn,
            purity_fields[0],
            label=r"kNN $X_0\!\to X_1$",
            color=dark,
            marker="o",
            seen=seen,
        )
        add_curve(
            ax_b,
            knn,
            purity_fields[1],
            label=r"kNN $X_1\!\to X_0$",
            color=light,
            marker="o",
            seen=seen,
        )
        fit_axes(
            ax_b, table.magnitude, seen, xlabel=r"Batch effect strength, $\lambda$", ylabel=ylabel
        )
        set_panel_title(
            ax_b,
            titles[0] if centred else rf"$\mathbf{{{letters[0].lower()}}}$  {titles[0]}",
            fontsize=title_pt or PANEL_TITLE_PT,
            letter=letters[0],
            centred=centred,
        )
        clean_legend(ax_b, loc="best", ncol=ncol)

    ax_c = fig.add_subplot(specs[-1])
    seen = []
    # One OT curve, the shared (union) graph.
    table = add_curve(
        ax_c,
        directed,
        "eot_shared_or_digit_purity",
        label="Sparse OT",
        color=ot.color,
        marker=ot.marker,
        seen=seen,
    )
    add_curve(
        ax_c,
        knn,
        "shared_correct_class_neighbor_fraction",
        label="Shared kNN",
        color=dark,
        marker="o",
        seen=seen,
    )
    add_curve(
        ax_c,
        knn,
        "mutual_correct_class_neighbor_fraction",
        label="Mutual kNN",
        color=light,
        marker="o",
        seen=seen,
    )
    # Named once for the pair.
    fit_axes(
        ax_c,
        table.magnitude,
        seen,
        xlabel=r"Batch effect strength, $\lambda$",
        ylabel="" if directed_panel else ylabel,
    )
    set_panel_title(
        ax_c,
        titles[-1] if centred else rf"$\mathbf{{{letters[-1].lower()}}}$  {titles[-1]}",
        fontsize=title_pt or PANEL_TITLE_PT,
        letter=letters[-1],
        centred=centred,
    )
    if legend_outside:
        # Beside the panel rather than over it. With the directed panel gone this key is the only
        # four-entry one left, and the column beside it is free.
        clean_legend(ax_c, loc="center left", ncol=1, bbox_to_anchor=(1.04, 0.5))
    else:
        clean_legend(ax_c, loc="best", ncol=ncol)


def downstream_panels(
    fig,
    specs,
    runs,
    *,
    letters=("D", "E"),
    titles=("Embedding purity", "Batch mixing"),
    title_pt=None,
    purity_ylabel="",
    centred=False,
):
    downstream = select_downstream(runs.summary.query("mode == 'original'"))
    method_specs = {
        "ot": (registry.resolve("ot"), "OT-tSNE"),
        "multimap": (registry.resolve("multimap"), "MultiMAP"),
        "scanorama": (registry.resolve("scanorama-50"), "Scanorama→t-SNE"),
    }
    # No y label on the purity panel. Its heading already names the quantity, and "Neighbor
    # purity" is written once on the first graph panel for all three that measure it.
    settings = (
        ("purity", titles[0], purity_ylabel, letters[0]),
        ("kbet", titles[1], "kBET", letters[1]),
    )
    for spec, (metric, title, ylabel, letter) in zip(specs, settings):
        ax = fig.add_subplot(spec)
        seen, magnitudes = [], []
        for _, group in downstream.groupby("display_method", sort=False):
            name = str(group.display_method.iloc[0])
            key = (
                "ot"
                if name.startswith("OT")
                else "multimap"
                if name.startswith("MultiMAP")
                else "scanorama"
            )
            style, method_label = method_specs[key]
            if (
                key == "multimap"
                and "effective_k" in group
                and set(group.effective_k.dropna()) == {15}
            ):
                method_label = "MultiMAP k=15"
            group = group.sort_values("magnitude")
            ax.errorbar(
                group.magnitude,
                group[metric],
                yerr=group[f"{metric}_std"].fillna(0),
                color=style.color,
                marker=style.marker,
                label=method_label,
                linewidth=0.9,
                markersize=5,
                capsize=1.5,
            )
            seen.extend(span_values(group[metric], group[f"{metric}_std"]))
            magnitudes.extend(group.magnitude)
        fit_axes(ax, magnitudes, seen, xlabel=r"Batch effect strength, $\lambda$", ylabel=ylabel)
        set_panel_title(
            ax,
            title if centred else rf"$\mathbf{{{letter.lower()}}}$  {title}",
            fontsize=title_pt or PANEL_TITLE_PT,
            letter=letter,
            centred=centred,
        )
        if metric == "purity":
            clean_legend(ax, loc="best", ncol=1)


def island_geometry(embedding, ids):
    """Centroid, radius and clearance for each island, in the embedding's own units."""
    unique = np.unique(ids)
    centres = np.stack([embedding[ids == value].mean(axis=0) for value in unique])
    radii = np.array(
        [
            np.quantile(np.linalg.norm(embedding[ids == value] - centre, axis=1), 0.9)
            for value, centre in zip(unique, centres)
        ]
    )
    gaps = np.linalg.norm(centres[:, None] - centres[None], axis=2) - radii[:, None] - radii[None]
    np.fill_diagonal(gaps, np.inf)
    return unique, centres, radii, gaps.min(axis=1)


def batch_pair(unique, centres, clearance):
    """The two islands to name in the batch row: one per batch, stacked, batch 0 above batch 1."""
    sides = unique % 2
    anchors = [i for i in np.flatnonzero(sides == 1) if clearance[i] > 0]
    if not anchors:
        return []
    anchor = max(anchors, key=lambda i: clearance[i])
    above = [
        i
        for i in np.flatnonzero(sides == 0)
        if clearance[i] > 0 and centres[i][1] > centres[anchor][1]
    ]
    if not above:
        return [(0.0, 1, [int(anchor)])]
    partner = min(above, key=lambda i: float(np.linalg.norm(centres[i] - centres[anchor])))
    return [(0.0, 1, [int(anchor)]), (0.0, 0, [int(partner)])]


def island_names(embedding, digits, batches, *, by, count, limits, panel_in, directions=24):
    """``(x, y, text)`` naming the best-separated island of each digit, or of each batch."""
    embedding = np.asarray(embedding, float)
    ids = np.asarray(digits) * 2 + np.asarray(batches)
    unique, centres, radii, clearance = island_geometry(embedding, ids)
    bucket = unique // 2 if by == "digit" else unique % 2
    (x_low, x_high), (y_low, y_high) = limits
    line = PANEL_TITLE_PT / 72.0 / panel_in * (x_high - x_low)

    middle = embedding.mean(axis=0)

    def place(index, taken):
        """Return ``(point, outward)`` for a name, or ``(None, False)`` when none fits."""
        own = embedding[ids == unique[index]]
        others = np.vstack([embedding[ids != unique[index]], taken])
        centre, radius = centres[index], radii[index]
        outward = centre - middle
        norm = float(np.linalg.norm(outward))
        outward = outward / norm if norm else np.zeros(2)
        accepted = []
        for step in GALLERY_NAME_PADS:
            pad = step * line
            for angle in np.linspace(0, 2 * np.pi, directions, endpoint=False):
                heading = np.array([np.cos(angle), np.sin(angle)])
                point = centre + heading * (radius + pad)
                if not (
                    x_low + pad <= point[0] <= x_high - pad
                    and y_low + pad <= point[1] <= y_high - pad
                ):
                    continue
                here = float(np.linalg.norm(own - point, axis=1).min())
                room = float(np.linalg.norm(others - point, axis=1).min())
                if room > here * GALLERY_NAME_MARGIN:
                    accepted.append(
                        (room, float(heading @ outward), step, point, float(heading[0]))
                    )
        if not accepted:
            return None, False
        # To the island's right if anywhere on its right will take it, however tight the fit.
        right = [a for a in accepted if a[4] >= GALLERY_NAME_RIGHT_MIN]
        if right:
            return max(right, key=lambda a: (a[0], a[2]))[3], True
        # Otherwise the outside of the cloud, where the panel's own margin is behind the name
        # rather than another island.
        away = [a for a in accepted if a[1] >= GALLERY_NAME_OUTWARD_MIN]
        if away:
            return max(away, key=lambda a: (a[0], a[2]))[3], True
        best = max(room for room, _, _, _, _ in accepted)
        near = [a for a in accepted if a[0] >= best * GALLERY_NAME_ROOM_TOLERANCE]
        return max(near, key=lambda a: a[1])[3], False

    if by == "batch":
        order = batch_pair(unique, centres, clearance)
    else:
        # Buckets in order of how much room their best island has, so the ones with a clear answer
        # are placed first and the crowded ones work around them rather than the other way about.
        order = []
        for value in np.unique(bucket):
            members = sorted(np.flatnonzero(bucket == value), key=lambda m: -clearance[m])
            members = [m for m in members if clearance[m] > 0]
            if members:
                order.append((float(clearance[members[0]]), int(value), members))
        order.sort(reverse=True)

    taken = np.empty((0, 2))
    placed = []
    for _, value, members in order:
        if len(placed) == count:
            break
        # Both of a digit's islands are tried, and one that can be named on the outside of the cloud
        # wins over one that can only be named into it -- even when the second has more room around
        # it.
        chosen, fallback = None, None
        for index in members:
            spot, away = place(index, taken)
            if spot is None:
                continue
            if away:
                chosen = spot
                break
            fallback = spot if fallback is None else fallback
        chosen = chosen if chosen is not None else fallback
        if chosen is not None:
            placed.append((float(chosen[0]), float(chosen[1]), str(value)))
            taken = np.vstack([taken, chosen[None]])
    return placed


def representative_embeddings(
    fig,
    spec,
    run_dir,
    labels0,
    labels1,
    *,
    unintegrated,
    letter="B",
    left=None,
    band=None,
    centred=False,
    row_gap_in=None,
    right_in=GALLERY_RIGHT_IN,
    wspace=GALLERY_WSPACE,
    label_side="right",
):
    """Panel B: one visualization per method at the reported magnitude, colored by digit."""
    ot_name = (
        "mag_10__ot__tsne__corruption_seed_0__embedding_seed_0"
        "__epsilon_policy_adaptive_per_magnitude__eps_1__k_15.npz"
    )
    patterns = (
        ("Joint t-SNE", None),
        ("OT-tSNE", ot_name),
        (
            "MultiMAP",
            "mag_10__multimap__default-guarded-scaling__corruption_seed_0__embedding_seed_0__k_30.npz",
        ),
        # Named as panel D's key names it, and on one line. It is the widest column heading by
        # some way, so whether it fits is a property of the layout, not of the string.
        (
            "Scanorama\u2192t-SNE",
            "mag_10__scanorama-50-tsne__tsne__corruption_seed_0__embedding_seed_0.npz",
        ),
    )
    # Two rows of the same four embeddings: batch above, digit below.
    cell = spec.get_position(fig)
    # The block may start left of its own cell, where the construction block leaves width unused.
    x0 = cell.x0 if left is None else left
    right = 1.0 - right_in / fig.get_figwidth()
    # Align gallery and construction blocks on the same vertical band.
    bottom, top_y = (cell.y0, cell.y1) if band is None else band
    band_in = (top_y - bottom) * fig.get_figheight()
    # ``wspace`` is a fraction of a panel, so n panels plus n-1 gaps span n + (n-1)*wspace of one.
    span = len(patterns) + (len(patterns) - 1) * wspace
    # The band holds the two panel rows and nothing else.
    panel_in = min((right - x0) * fig.get_figwidth() / span, band_in / 2)
    spare = max(0.0, band_in - 2 * panel_in)
    if row_gap_in is not None and spare > row_gap_in:
        # Keep excess row height below the paired gallery rows.
        spare = row_gap_in
        bottom = top_y - (2 * panel_in + spare) / fig.get_figheight()
    block = span * panel_in / fig.get_figwidth()
    # Height fixes the square-panel size, leaving a little unused width. Right-align the block so
    # that slack sits between A and B rather than at the outer edge of the figure.
    x0 = right - block
    heading_left = x0
    gallery = spec.get_gridspec().figure.add_gridspec(
        3,
        len(patterns),
        left=x0,
        right=x0 + block,
        bottom=bottom,
        top=top_y,
        wspace=wspace,
        hspace=GALLERY_HSPACE,
        height_ratios=(panel_in, max(spare, 1e-6), panel_in),
    )
    joint_labels = np.concatenate([labels0, labels1])
    batch_ids = np.concatenate(
        [np.zeros(len(labels0), dtype=int), np.ones(len(labels1), dtype=int)]
    )
    batch_colors = np.asarray(GALLERY_BATCH_COLOURS)[batch_ids]
    axes = []
    for index, (title, filename) in enumerate(patterns):
        if filename is None:
            embedding = unintegrated
        else:
            path = synthetic.gallery_path(run_dir, filename)
            if not path.exists():
                raise FileNotFoundError(f"required cached embedding is missing: {path}")
            embedding = np.load(path)["embedding"]
        for row, (colors, cmap, row_label) in enumerate(
            ((batch_colors, None, "Batch"), (joint_labels, "tab10", "Digit"))
        ):
            ax = fig.add_subplot(gallery[row * 2, index])
            if row == 0:
                axes.append(ax)
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=colors,
                cmap=cmap,
                s=GALLERY_POINT_PT2 * (panel_in / GALLERY_POINT_REF_IN) ** 2,
                linewidths=0,
                rasterized=True,
            )
            ax.set(xticks=[], yticks=[])
            metric_drawing.frame_embedding_panel(ax, embedding)
            if filename is None:
                # Label digit islands only in the unintegrated panel, where batches form separate
                # islands.
                for x, y, name in island_names(
                    embedding,
                    joint_labels,
                    batch_ids,
                    by="digit" if row else "batch",
                    count=GALLERY_DIGIT_NAMES if row else 2,
                    limits=(ax.get_xlim(), ax.get_ylim()),
                    panel_in=panel_in,
                ):
                    ax.text(x, y, name, fontsize=PANEL_TITLE_PT, ha="center", va="center")
            for spine in ax.spines.values():
                spine.set_visible(False)
            if index == (len(patterns) - 1 if label_side == "right" else 0):
                # No ticks and no frame, so nothing stands between the label and the points.
                ax.set_ylabel(row_label, fontsize=GALLERY_TITLE_PT, labelpad=GALLERY_LABEL_PAD_PT)
                ax.yaxis.set_label_position(label_side)
            if row == 0:
                ax.set_title(title, loc="center", fontsize=GALLERY_TITLE_PT)
    # The heading clears the column titles rather than sharing their line.
    lift = gallery_title_lift(fig)
    at = spec.get_position(fig).y1 + lift
    # The Batch/Digit names are part of the panel, so the span they hang off includes them: the
    # letter belongs above them, not above the first scatter, and the heading is centered on what
    # the
    # reader sees rather than on the scatters alone.
    names_in = (
        max(text_height_in(fig, name, GALLERY_TITLE_PT) for name in ("Batch", "Digit"))
        + GALLERY_LABEL_PAD_PT / 72.0
    ) / fig.get_figwidth()
    span = (
        (heading_left - names_in, heading_left + block)
        if label_side == "left"
        else (heading_left, heading_left + block + names_in)
    )
    # The letter hangs off the span, the heading off the block: the names say which row is which
    # and the block is what is plotted.
    rect = (heading_left, heading_left + block)
    if centred:
        title = fig.text(
            heading_left + block / 2,
            at,
            f"Embedding visualization ($\\lambda={DISPLAY_MAGNITUDE:g}$)",
            ha="center",
            va="bottom",
            fontsize=PANEL_TITLE_PT,
        )
        title._panel_span, title._panel_rect = span, rect
    heading = fig.text(
        span[0],
        at,
        # Display cached seed-0 embeddings at the same magnitude as panel A.
        rf"$\mathbf{{{letter.lower()}}}$"
        if centred
        else rf"$\mathbf{{{letter.lower()}}}$  Embedding visualization "
        rf"($\lambda={DISPLAY_MAGNITUDE:g}$)",
        ha="left",
        va="bottom",
        fontsize=PANEL_TITLE_PT,
    )
    heading._panel_span, heading._panel_rect = span, rect
    return span


def trim_panel_right(fig, ax, limit: float) -> None:
    """Narrow ``ax`` until nothing it draws reaches past ``limit``."""
    fig.canvas.draw()
    _, right = panel_ink_span(ax, fig.canvas.get_renderer(), fig.transFigure.inverted())
    if right <= limit:
        return
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, max(box.width - (right - limit), 0.05), box.height])


def shift_panel_to(fig, ax, target_left: float, *, clear_of=None) -> None:
    """Slide one metric panel sideways until its left ink edge lands on ``target_left``."""
    fig.canvas.draw()
    renderer, inverse = fig.canvas.get_renderer(), fig.transFigure.inverted()
    left, _ = panel_ink_span(ax, renderer, inverse)
    if clear_of is not None:
        # Never back into the panel to its left. Which panel that is decides how far this can go,
        # and with the graph panel's key beside it rather than inside it there is no room at all.
        floor = (
            panel_ink_span(clear_of, renderer, inverse)[1] + PANEL_CLEARANCE_IN / fig.get_figwidth()
        )
        target_left = max(target_left, floor)
    box = ax.get_position()
    ax.set_position([box.x0 + (target_left - left), box.y0, box.width, box.height])


def save(fig, stem, *, output_dir: Path):
    """Write the figure once, as ``layout.save_figure`` writes every other paper figure."""
    path = paper_layout.save_figure(fig, stem, preview=False, close=False, root=output_dir)
    plt.close(fig)
    return path


def build_variant(
    variant: str,
    *,
    source_root: Path,
    output_dir: Path,
    prepared: synthetic.SyntheticRuns | None = None,
) -> list[Path]:
    runs = prepared or synthetic.preflight(variant, source_root=source_root, output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean0, labels0, clean1, labels1, pca = load_mnist_and_pca(
        runs.configs["original"], source_root=source_root
    )
    audit = pd.read_csv(synthetic.audit_path(runs.run_dirs["original"]))
    alpha, beta = reconstruct_corruptions(
        variant, audit, pca, int(runs.configs["original"]["corruption_seeds"][0])
    )
    cache_root = synthetic.cache_directory(source_root)
    directed = synthetic.load_directed(runs, cache_root)
    unintegrated = synthetic.load_joint_embedding(runs, cache_root)

    # With embeddings: A and the visualizations on top, then C/D, then E/F -- each pair a row, so
    # the two graph panels are read against each other and the two visualization panels likewise.
    with plt.rc_context({"figure.constrained_layout.use": False}):
        alt = plt.figure(figsize=paper_layout.full_width(3.5))
    alt.subplots_adjust(left=0.062, right=0.985, bottom=0.085, top=0.925)
    # The top row is sized to what A and B actually need, rather than to a round number they then
    # float in.
    grid = alt.add_gridspec(2, 4, height_ratios=(1.70, 1.0), hspace=0.14, wspace=0.48)
    # The top row gets its own split rather than reusing the four equal columns below.
    a_right, a_bottom, a_top = construction_panel(
        alt,
        grid[0, 0:2],
        clean0,
        labels0,
        clean1,
        labels1,
        alpha,
        beta,
        variant=variant,
        lift=gallery_title_lift(alt),
    )
    representative_embeddings(
        alt,
        grid[0, 2:4],
        runs.run_dirs["original"],
        labels0,
        labels1,
        unintegrated=unintegrated,
        left=a_right + GALLERY_BLOCK_GAP / alt.get_figwidth(),
        band=(a_bottom, a_top),
    )
    mechanism_panels(
        alt,
        (grid[1, 0], grid[1, 1]),
        variant,
        runs,
        directed,
        letters=("C", "D"),
        ncol=NARROW_LEGEND_NCOL,
        titles=("Directed graph purity", "Symmetrized graph purity"),
    )
    downstream_panels(alt, (grid[1, 2], grid[1, 3]), runs, letters=("E", "F"))
    # Use compact magnitude-axis labels; define lambda in the caption.
    for ax in alt.axes[-4:]:
        ax.set_xlabel(r"$\lambda$")
    lay_out_panels(alt)
    alt_figure = save(alt, f"{variant}_final_with_embeddings", output_dir=output_dir)
    # A second pair without the directed-graph panel.
    trimmed = build_without_directed(
        variant,
        runs,
        clean0,
        labels0,
        clean1,
        labels1,
        alpha,
        beta,
        directed,
        output_dir=output_dir,
        unintegrated=unintegrated,
    )
    centred = build_without_directed(
        variant,
        runs,
        clean0,
        labels0,
        clean1,
        labels1,
        alpha,
        beta,
        directed,
        output_dir=output_dir,
        unintegrated=unintegrated,
        centred=True,
        suffix="_centred",
    )
    # A third copy of the centered figure, differing only in how panel A is drawn: strips rather
    # than
    # columns.
    strip = build_without_directed(
        variant,
        runs,
        clean0,
        labels0,
        clean1,
        labels1,
        alpha,
        beta,
        directed,
        output_dir=output_dir,
        unintegrated=unintegrated,
        centred=True,
        layout="strip",
        suffix="_centred_strip",
    )
    # And a fourth, the strip drawn as tightly as it goes: the step named between the rows, the row
    # names along them, and the gallery's own names moved into the band that opens up between the
    # two panels.
    inline = build_without_directed(
        variant,
        runs,
        clean0,
        labels0,
        clean1,
        labels1,
        alpha,
        beta,
        directed,
        output_dir=output_dir,
        unintegrated=unintegrated,
        centred=True,
        layout="inline",
        suffix="_centred_strip_inline",
    )
    return [alt_figure, *trimmed, *centred, *strip, *inline]


def build_without_directed(
    variant,
    runs,
    clean0,
    labels0,
    clean1,
    labels1,
    alpha,
    beta,
    directed,
    *,
    output_dir: Path,
    unintegrated: np.ndarray,
    centred=False,
    rows=None,
    suffix="",
    layout="columns",
) -> list[Path]:
    """The same figure with panel c dropped, so the metrics run c/d/e."""
    strip, inline = layout in ("strip", "inline"), layout == "inline"
    height_in = {"columns": 3.2, "strip": STRIP_FIGURE_IN, "inline": STRIP_INLINE_FIGURE_IN}
    with plt.rc_context({"figure.constrained_layout.use": False}):
        alt = plt.figure(figsize=paper_layout.full_width(height_in[layout]))
    alt.subplots_adjust(left=0.062, right=0.985, bottom=0.095, top=0.925)
    # 2.22 rather than 1.90, with 0.1 in off the figure: the metric row loses that 0.1 in and the
    # top row keeps its height.
    ratios = {
        "columns": (2.22, 1.0),
        "strip": STRIP_HEIGHT_RATIOS,
        "inline": STRIP_INLINE_HEIGHT_RATIOS,
    }
    hspaces = {"columns": 0.16, "strip": STRIP_HSPACE, "inline": STRIP_INLINE_HSPACE}
    grid = alt.add_gridspec(2, 1, height_ratios=ratios[layout], hspace=hspaces[layout])
    top = grid[0].subgridspec(1, 4)
    # Three panels, not four columns with one left over.
    bottom = grid[1].subgridspec(
        1,
        4,
        width_ratios=(1.0, 0.43 if inline else 0.52, 1.0, 1.0),
        wspace=0.34,
    )
    if strip:
        a_right, a_bottom, a_top = construction_panel_strip(
            alt,
            top[0:2],
            clean0,
            labels0,
            clean1,
            labels1,
            alpha,
            beta,
            variant=variant,
            lift=gallery_title_lift(alt),
            centred=centred,
            inline=inline,
        )
        # The gallery takes the row's height, not panel A's block.
        cell = top[0:2].get_position(alt)
        band = (cell.y0, cell.y1 - CONSTRUCTION_HEADER_IN / alt.get_figheight())
    else:
        a_right, a_bottom, a_top = construction_panel(
            alt,
            top[0:2],
            clean0,
            labels0,
            clean1,
            labels1,
            alpha,
            beta,
            variant=variant,
            lift=gallery_title_lift(alt),
            rows=rows,
            centred=centred,
        )
        band = (a_bottom, a_top)
    block_gap = STRIP_INLINE_BLOCK_GAP_IN if inline else GALLERY_BLOCK_GAP
    right_in = {
        "columns": GALLERY_RIGHT_IN,
        "strip": STRIP_GALLERY_RIGHT_IN,
        "inline": STRIP_INLINE_GALLERY_RIGHT_IN,
    }
    representative_embeddings(
        alt,
        top[2:4],
        runs.run_dirs["original"],
        labels0,
        labels1,
        unintegrated=unintegrated,
        left=a_right + block_gap / alt.get_figwidth(),
        band=band,
        centred=centred,
        letter="C",
        row_gap_in=STRIP_GALLERY_ROW_GAP_IN if strip else None,
        right_in=right_in[layout],
        wspace=STRIP_GALLERY_WSPACE if strip else GALLERY_WSPACE,
        label_side="left" if inline else "right",
    )
    # B is the graph panel and C the gallery, though the gallery is the one drawn beside A.
    mechanism_panels(
        alt,
        (bottom[0],),
        variant,
        runs,
        directed,
        letters=("B",),
        directed_panel=False,
        legend_outside=True,
        centred=centred,
    )
    downstream_panels(
        alt,
        (bottom[2], bottom[3]),
        runs,
        letters=("D", "E"),
        purity_ylabel="Neighbor purity",
        centred=centred,
    )
    lay_out_panels(alt)
    return [
        save(alt, f"{variant}_final_with_embeddings_no_directed{suffix}", output_dir=output_dir)
    ]
