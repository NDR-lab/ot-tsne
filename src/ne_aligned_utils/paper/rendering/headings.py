"""Measured family headings and panel tags, independent of registries."""

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt

FAMILY_TAGS = ("a", "b", "c", "d")
FAMILY_TAG_PLACEMENTS = ("top", "centre")
FAMILY_TAG_GAP_EM = 0.45


def family_title(
    family: str,
    *,
    family_labels: Mapping[str, str],
    family_order: Sequence[str],
    placement: str = "top",
) -> str:
    """The heading as it is set on the axes, padded so the four blocks line up.

    ``"top"`` pads a short heading with a trailing blank line, so its text sits on the first row
    beside the wrapped ones. ``"centre"`` leaves it alone -- the block is raised by half a line
    through the title ``pad`` instead, which is what centres it against a two-line neighbor.
    """
    if placement not in FAMILY_TAG_PLACEMENTS:
        raise ValueError(f"placement must be one of {FAMILY_TAG_PLACEMENTS}, got {placement!r}")
    label = family_labels[family]
    if placement != "top":
        return label
    tallest = max(len(family_labels[f].split("\n")) for f in family_order)
    return label + "\n" * (tallest - len(label.split("\n")))


def set_family_title(
    ax,
    family: str,
    *,
    family_labels: Mapping[str, str],
    family_order: Sequence[str],
    placement: str = "top",
) -> None:
    """Set one column heading, padded or raised so all four sit on one line."""
    pad = float(plt.rcParams["axes.titlepad"])
    if placement == "centre":
        # Half a blank line's worth, which is what lifts a one-line heading to the middle of a
        # band sized for two.
        missing = max(len(family_labels[f].split("\n")) for f in family_order) - len(
            family_labels[family].split("\n")
        )
        pad += missing / 2 * float(plt.rcParams["axes.titlesize"]) * 1.2
    # Center headings on panels; position letters from panel ink separately.
    ax.set_title(
        family_title(
            family, family_labels=family_labels, family_order=family_order, placement=placement
        ),
        pad=pad,
    )


def _column_ink_span(fig, ax, inv, renderer) -> tuple[float, float]:
    """The horizontal extent of everything drawn in ``ax``'s grid column, in figure coordinates.

    The column, not the one panel: the first column's y label and tick numbers belong to it as
    much as its axes box does, and on a multi-row grid they are carried by whichever row happens
    to hold them. The x tick numbers count at both ends -- the outer ones overhang their spines by
    half their width, 0.046 in on the pooled figure.
    """
    spec = ax.get_subplotspec()
    peers = [ax]
    if spec is not None:
        column = spec.colspan.start
        peers = [
            other
            for other in fig.axes
            if (other_spec := other.get_subplotspec()) is not None
            and len(list(other_spec.colspan)) == 1
            and other_spec.colspan.start == column
        ]
    low, high = [], []
    for peer in peers:
        box = peer.get_window_extent().transformed(inv)
        low.append(box.x0)
        high.append(box.x1)
        labels = (
            *peer.get_yticklabels(),
            *peer.get_xticklabels(),
            peer.yaxis.label,
            peer.xaxis.label,
        )
        for artist in labels:
            if artist.get_text():
                label = artist.get_window_extent(renderer).transformed(inv)
                low.append(label.x0)
                high.append(label.x1)
    return min(low), max(high)


def _column_edges(fig, title_axes: Sequence, inv, renderer) -> list[float]:
    """Return letter anchors at boundaries between columns' rendered ink spans.

    Interior boundaries bisect gaps; the first starts at its own leftmost ink.
    Measured edges account for unequal label widths across columns.
    """
    spans = [_column_ink_span(fig, ax, inv, renderer) for ax in title_axes]
    edges = [spans[0][0]]
    for before, span in zip(spans, spans[1:]):
        edges.append((before[1] + span[0]) / 2.0)
    return edges


def _clear_headings(fig, title_axes: Sequence, tags: Sequence, gap: float) -> None:
    """Shift oversized headings to clear fixed panel letters.

    Constrain each heading between neighboring letters, sharing unavoidable shortfall
    evenly. Headings already fitting their spans remain centered.
    """
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    marks = [tag.get_window_extent(renderer).transformed(inv) for tag in tags]
    for index, ax in enumerate(title_axes):
        if not ax.title.get_text().strip():
            continue
        title = ax.title.get_window_extent(renderer).transformed(inv)
        low = marks[index].x1 + gap
        high = marks[index + 1].x0 - gap if index + 1 < len(marks) else 1.0
        span = title.x1 - title.x0
        if span >= high - low:
            target = (low + high - span) / 2.0
        else:
            target = min(max(title.x0, low), high - span)
        if abs(target - title.x0) < 1e-6:
            continue
        width = ax.get_window_extent().transformed(inv).width
        ax.title.set_x(ax.title.get_position()[0] + (target - title.x0) / width)


def _first_line_baseline(title, renderer, inv) -> float:
    """Figure-space y of the baseline of a title's first line.

    Matplotlib anchors a ``va="baseline"`` text on the baseline of its *last* line, so a heading's
    own position names the bottom row rather than the top one. The rest is the drop from the first
    baseline to the last, which is measured off the title instead of assumed to be
    ``1.2 * titlesize``: the layout takes a line's advance from the font's metrics, which run about
    4% wider than that.
    """
    full = title.get_text()
    anchor = inv.transform(title.get_transform().transform(title.get_position()))[1]
    lines = full.split("\n")
    if len(lines) == 1:
        return anchor
    stacked = title.get_window_extent(renderer).transformed(inv).height
    title.set_text(lines[0])
    single = title.get_window_extent(renderer).transformed(inv).height
    title.set_text(full)
    return anchor + (stacked - single)


def place_family_tags(fig, title_axes: Sequence, *, placement: str = "top") -> list:
    """Align panel letters at a shared height beside headings after layout.

    Use the tallest heading to set the row height. Top alignment matches first-line
    baselines so letter ascenders do not produce inconsistent vertical placement.
    """
    if placement not in FAMILY_TAG_PLACEMENTS:
        raise ValueError(f"placement must be one of {FAMILY_TAG_PLACEMENTS}, got {placement!r}")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [ax.title.get_window_extent(renderer).transformed(inv) for ax in title_axes]
    # Anchor letters to each column's ink edge so heading width cannot move them.
    gap = FAMILY_TAG_GAP_EM * float(plt.rcParams["axes.titlesize"]) / 72.0 / fig.get_figwidth()
    edges = _column_edges(fig, title_axes, inv, renderer)
    size = float(plt.rcParams["axes.titlesize"])
    if placement == "top":
        y = max(_first_line_baseline(ax.title, renderer, inv) for ax in title_axes)
        valign = "baseline"
    else:
        y = (max(box.y1 for box in boxes) + min(box.y0 for box in boxes)) / 2
        valign = "center"
    drawn = []
    for tag, edge in zip(FAMILY_TAGS, edges):
        drawn.append(
            fig.text(
                edge,
                y,
                tag,
                ha="left",
                va=valign,
                fontsize=size,
                fontweight="bold",
            )
        )
    _clear_headings(fig, title_axes, drawn, gap)
    return drawn
