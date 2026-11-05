"""Legend glyphs, pairing, measurement and placement, with explicit presentation inputs.

No registry, figure-builder module or scientific selection is consulted here.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from ne_aligned_utils.paper.rendering.context import MethodPresentation


@dataclass(frozen=True)
class LegendPresentation:
    methods: Mapping[str, MethodPresentation]
    family_order: tuple[str, ...]
    integration_tool_order: tuple[str, ...] = ()
    omitted: tuple[str, ...] = ()
    dimension_size: float = 34.0

    def __post_init__(self):
        object.__setattr__(self, "methods", MappingProxyType(dict(self.methods)))


def handle_kwargs(*, marker, color, size, label, edge, alpha, unfilled):
    """Keyword arguments for a legend proxy glyph, separate from the scatter encoding."""
    common = dict(linestyle="none", marker=marker, markersize=size**0.5, color="none", label=label)
    if unfilled:
        common.update(markeredgecolor=color, markeredgewidth=0.9)
    else:
        common.update(
            markerfacecolor=color, markeredgecolor=edge[0], markeredgewidth=edge[1], alpha=alpha
        )
    return common


def method_handles(keys, *, presentation):
    return [Line2D([], [], **presentation.methods[key].legend_glyph) for key in keys]


INTEGRATION_NCOL = 2
DIMENSION_KEY: tuple[tuple[str, str], ...] = (
    ("o", r"50D$\rightarrow$t-SNE"),
    ("s", "2D"),
)
DIMENSION_KEY_COLOUR = "#9a9a9a"
COLUMN_SPACING = 1.0
KEY_COLUMN_SPACING = 0.5
KEY_HANDLELENGTH = 0.8
KEY_HANDLETEXTPAD = 0.15
PAIR_PAD = 0.25
PAIR_HANDLELENGTH = 1.4
REFERENCE_PANEL_INSET = 0.0103
LEGEND_SHIFT: dict[str, float] = {"ot": 0.016, "dedicated": 0.004, "integration": -0.020}
DIMENSION_KEY_PLACEMENT = "centre"


def _merge_paired_label(first: str, second: str, *, whole_word_tail: bool = True) -> str:
    """``"Seurat 50D"`` + ``"Seurat 2D"`` -> ``"Seurat 50D/2D"``, keeping any shared trailing word.

    With ``whole_word_tail`` a shared tail is only honoured when it is a *whole* trailing word, i.e.
    it begins at a space in both labels, so ``"Seurat 50D"`` and ``"Seurat 2D"`` keep both letters.
    Without it the match is greedy and the shared ``D`` is absorbed, giving ``"Seurat 50/2D"`` --
    narrower, and unambiguous once the reader has seen one such label. Either way the EOT pair,
    ``"EOT 50D t=0"`` and ``"EOT 2D t=0"``, keeps its trailing word.
    """
    limit = min(len(first), len(second))
    prefix = 0
    while prefix < limit and first[prefix] == second[prefix]:
        prefix += 1
    greedy = 0
    while greedy < limit - prefix and first[-1 - greedy] == second[-1 - greedy]:
        greedy += 1
    if not whole_word_tail:
        # Merge dimension labels greedily (50D/2D -> 50/2D), preserving trailing qualifiers.
        suffix = greedy
    else:
        suffix = 0
        for length in range(greedy, 0, -1):
            if first[-length] == " " and second[-length] == " ":
                suffix = length
                break
    head = first[:prefix]
    tail = first[len(first) - suffix :] if suffix else ""
    left = first[prefix : len(first) - suffix]
    right = second[prefix : len(second) - suffix]
    return f"{head}{left}/{right}{tail}"


def _place_legend(
    legend_ax,
    handles,
    labels,
    *,
    x: float,
    y: float,
    ncol: int,
    folded: bool,
    loc: str = "upper left",
    columnspacing: float = COLUMN_SPACING,
    handlelength: float | None = None,
    handletextpad: float = 0.4,
    pair_pad: float = PAIR_PAD,
    pair_handlelength: float = PAIR_HANDLELENGTH,
):
    """One legend block anchored by ``loc`` at ``(x, y)`` in figure coordinates.

    Built as a ``Legend`` and added as an artist rather than through ``Axes.legend``, which keeps
    only the most recent legend: with five blocks on one axes the fifth silently evicted the
    fourth, and the figure lost a key with nothing raised.
    """
    legend = Legend(
        legend_ax,
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=pair_pad)},
        handlelength=(pair_handlelength if folded else 1.0)
        if handlelength is None
        else handlelength,
        loc=loc,
        bbox_to_anchor=(x, y),
        bbox_transform=legend_ax.figure.transFigure,
        ncol=ncol,
        frameon=False,
        borderaxespad=0.0,
        borderpad=0.0,
        handletextpad=handletextpad,
        labelspacing=0.35,
        columnspacing=columnspacing,
    )
    legend.set_in_layout(False)
    legend_ax.add_artist(legend)
    return legend


def _legend_glyph_centres(legend, renderer, inverse) -> list[tuple[float, float]]:
    """Centre of every glyph a legend draws in its handle boxes, in figure fractions.

    Measured from the drawn artists rather than derived from ``handlelength`` and
    ``handletextpad``: the block's entries are paired handles laid out by ``HandlerTuple`` and the
    key's are single ones with their own handle length, so the two have no arithmetic in common.
    """
    from matplotlib.lines import Line2D

    centres: list[tuple[float, float]] = []
    stack = [legend]
    while stack:
        artist = stack.pop()
        stack.extend(getattr(artist, "get_children", list)())
        if isinstance(artist, Line2D):
            box = artist.get_window_extent(renderer).transformed(inverse)
            centres.append(((box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0))
    return centres


def _align_dimension_key(fig, legend, key, block, *, placement=DIMENSION_KEY_PLACEMENT) -> None:
    """Align a separate dimension key to the legend block's measured columns and row pitch.

    Shift the existing anchor to respect either upper- or lower-left placement.
    Leave keys sharing the last tool's row unchanged.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    block_glyphs = _legend_glyph_centres(legend, renderer, inverse)
    key_glyphs = _legend_glyph_centres(key, renderer, inverse)
    if not block_glyphs or not key_glyphs:
        return
    rows = sorted({round(y, 6) for _x, y in block_glyphs}, reverse=True)
    if len(rows) < 2 or min(y for _x, y in key_glyphs) >= rows[-1] - 1e-9:
        return
    pitch = rows[0] - rows[1]
    box = key.get_window_extent(renderer).transformed(inverse)
    if placement == "centre":
        target_x0 = (block.x0 + block.x1) / 2.0 - box.width / 2.0
    else:
        target_x0 = box.x0 + (min(x for x, _y in block_glyphs) - min(x for x, _y in key_glyphs))
    shift_x = target_x0 - box.x0
    shift_y = (rows[-1] - pitch) - max(y for _x, y in key_glyphs)
    anchor = Bbox(key.get_bbox_to_anchor().get_points()).transformed(inverse)
    key.set_bbox_to_anchor((anchor.x0 + shift_x, anchor.y0 + shift_y), fig.transFigure)


def _row_major_legend(handles: Sequence, labels: Sequence[str], ncol: int):
    """Permute entries so matplotlib's column-major legend reads left to right, then down.

    Matplotlib has no row-major option, so the fix is to hand it the entries in the order that
    *places* them row-major. With ``n`` entries in ``ncol`` columns it fills ceil(n/ncol) per
    column, short-changing the last columns, and the permutation below reproduces exactly that
    distribution -- taking column c's rows in turn and skipping indices past the end -- so no entry
    is dropped or duplicated however uneven the grid is.
    """
    n = len(labels)
    if ncol <= 1 or n <= ncol:
        return list(handles), list(labels)
    nrows = -(-n // ncol)
    permuted = [r * ncol + c for c in range(ncol) for r in range(nrows) if r * ncol + c < n]
    assert sorted(permuted) == list(range(n)), permuted
    return [handles[i] for i in permuted], [labels[i] for i in permuted]


def _merged_legend_entries(
    ordered: Sequence[str],
    *,
    presentation: LegendPresentation,
    whole_word_tail: bool = True,
) -> tuple[list, list[str]]:
    """Fold paired 2D and 50D methods into shared legend entries.

    Merge only matching tool, family, and color with distinct markers. MultiMAP settings
    remain separate because their colors differ. Apply presentation omissions first.
    """
    ordered = [k for k in ordered if k not in presentation.omitted]
    by_tool: dict[str, list[str]] = {}
    for key in ordered:
        by_tool.setdefault(presentation.methods[key].tool, []).append(key)

    handles: list = []
    labels: list[str] = []
    done: set[str] = set()
    for key in ordered:
        if key in done:
            continue
        style = presentation.methods[key]
        group = by_tool[style.tool]
        mate = None
        if len(group) == 2:
            other = presentation.methods[group[1] if group[0] == key else group[0]]
            mine = style.scatter
            theirs = other.scatter
            same_color = mine["c"] == theirs["c"]
            diff_marker = mine["marker"] != theirs["marker"]
            if style.family == other.family and same_color and diff_marker:
                mate = other
        if mate is None:
            handles.extend(method_handles([key], presentation=presentation))
            labels.append(presentation.methods[key].display_label)
            done.add(key)
            continue
        handles.append(tuple(method_handles([key, mate.key], presentation=presentation)))
        labels.append(
            _merge_paired_label(
                presentation.methods[key].display_label,
                presentation.methods[mate.key].display_label,
                whole_word_tail=whole_word_tail,
            )
        )
        done.update({key, mate.key})
    return handles, labels


def _integration_entries(
    keys: Sequence[str],
    *,
    presentation: LegendPresentation,
) -> tuple[list, list[str]]:
    """Paired handles and *tool-only* labels for the integration family.

    :func:`_merged_legend_entries` folds each tool's 50D and 2D forms onto one handle and names the
    result "Scanorama 50D/2D". The suffix is identical on all seven and is what
    :data:`DIMENSION_KEY` says once, so it comes off here.
    """
    handles, labels = _merged_legend_entries(keys, presentation=presentation)
    suffix = " 50D/2D"
    unfolded = [label for label in labels if not label.endswith(suffix)]
    if unfolded:
        raise ValueError(
            f"integration entries that are not a folded 50D/2D pair: {unfolded}. The dimension key "
            "claims every glyph pair in this block is 50D and 2D, so an unpaired entry here would "
            "be labeled by a key that does not describe it."
        )
    return handles, [label.removesuffix(suffix) for label in labels]


def _dimension_key_entries(
    *,
    size=34.0,
    color=DIMENSION_KEY_COLOUR,
    entries=DIMENSION_KEY,
) -> tuple[list, list[str]]:
    """The two key entries: one grey glyph each, its label immediately after it."""
    return (
        [
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="none",
                markersize=float(size) ** 0.5,
                markerfacecolor=color,
                markeredgecolor="none",
            )
            for marker, _label in entries
        ],
        [label for _marker, label in entries],
    )


def legend_rows(
    ordered: Sequence[str],
    *,
    presentation: LegendPresentation,
    integration_ncol: int = INTEGRATION_NCOL,
    extra_rows: int = 0,
) -> int:
    """Text rows the tallest legend block needs, with the integration family at two columns.

    The legend row is reserved from this, so it is what turns a two-column block into a shorter
    figure rather than a taller gap.
    """
    by_family: dict[str, list[str]] = {}
    for key in ordered:
        by_family.setdefault(presentation.methods[key].family, []).append(key)
    rows = []
    for family, keys in by_family.items():
        entries = len(_merged_legend_entries(keys, presentation=presentation)[1])
        ncol = integration_ncol if family == "integration" else 1
        rows.append(-(-entries // ncol))
    return (max(rows) if rows else 1) + extra_rows


def legend_order(ordered: Sequence[str], *, presentation: LegendPresentation) -> list[str]:
    """``ordered`` with the integration family sorted by the configured tool order.

    The sort is stable, so within a tool the 50D form stays before the 2D form and the folded
    label reads "Seurat 50D/2D".
    """
    rank = {tool: i for i, tool in enumerate(presentation.integration_tool_order)}
    integration = sorted(
        (k for k in ordered if presentation.methods[k].family == "integration"),
        key=lambda key: rank.get(presentation.methods[key].tool, len(rank)),
    )
    out, feed = [], iter(integration)
    for key in ordered:
        out.append(next(feed) if presentation.methods[key].family == "integration" else key)
    return out


def family_legends(
    fig,
    axes,
    legend_ax,
    ordered: Sequence[str],
    *,
    lift: float = 0.014,
    ot_anchor_x: float = 0.01,
    reference_anchor_x: float | None = 0.99,
    presentation: LegendPresentation,
    shifts: Mapping[str, float] | None = None,
    integration_ncol: int = INTEGRATION_NCOL,
) -> None:
    """One legend block per family under the panels, with the 50D/2D key on its own row."""
    fig.canvas.draw()
    band = legend_ax.get_position()
    legend_ax.axis("off")
    shifts = LEGEND_SHIFT if shifts is None else shifts
    by_family: dict[str, list[str]] = {}
    for key in ordered:
        by_family.setdefault(presentation.methods[key].family, []).append(key)
    gap, nudge, floor_x = 0.012, 0.02, 0.0
    for column, family in enumerate(presentation.family_order):
        keys = by_family.get(family, [])
        if not keys:
            continue
        if family == "ot":
            loc, anchor_x = "upper left", ot_anchor_x
        elif family == "reference":
            loc = "upper right"
            anchor_x = (
                axes[-1, column].get_position().x1 - REFERENCE_PANEL_INSET
                if reference_anchor_x is None
                else reference_anchor_x
            )
        else:
            loc = "upper left"
            anchor_x = max(axes[-1, column].get_position().x0 - nudge, floor_x)
        anchor_x += shifts.get(family, 0.0)
        if loc == "upper left":
            anchor_x = max(anchor_x, floor_x)

        two_column = family == "integration" and integration_ncol > 1
        if two_column:
            handles, labels = _integration_entries(keys, presentation=presentation)
            handles, labels = _row_major_legend(handles, labels, integration_ncol)
            ncol, folded = integration_ncol, True
        else:
            handles, labels = _merged_legend_entries(keys, presentation=presentation)
            ncol, folded = 1, any(isinstance(handle, tuple) for handle in handles)
        legend = _place_legend(
            legend_ax,
            handles,
            labels,
            x=anchor_x,
            y=band.y1 + lift,
            ncol=ncol,
            folded=folded,
            loc=loc,
        )
        fig.canvas.draw()
        extent = legend.get_window_extent().transformed(fig.transFigure.inverted())
        if two_column:
            # Reserve a full row for the dimension key, then align it to the drawn glyphs
            # and measured row pitch.
            dimension_key = _place_legend(
                legend_ax,
                *_dimension_key_entries(size=presentation.dimension_size),
                x=anchor_x,
                y=extent.y0 - 0.002,
                ncol=len(DIMENSION_KEY),
                folded=False,
                loc="upper left",
                columnspacing=KEY_COLUMN_SPACING,
                handlelength=KEY_HANDLELENGTH,
                handletextpad=KEY_HANDLETEXTPAD,
            )
            _align_dimension_key(fig, legend, dimension_key, extent)
            fig.canvas.draw()
            extent = Bbox.union(
                [extent, dimension_key.get_window_extent().transformed(fig.transFigure.inverted())]
            )
        floor_x = extent.x1 + gap
