"""Registry- and scanner-independent runtime geometry, legends and device glyphs."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ne_aligned_utils.paper import layout
from ne_aligned_utils.paper.rendering.runtime import RuntimeFigureStyle


def order_by_size(datasets: tuple[str, ...], sizes: dict[str, int]) -> tuple[str, ...]:
    """Sort datasets by increasing cell count, breaking ties and unknown sizes by name."""
    return tuple(sorted(datasets, key=lambda d: (sizes.get(d, 0), d)))


def dataset_offsets(datasets: tuple[str, ...], *, row_spread: float) -> dict[str, float]:
    """Vertical offset per dataset, **smallest at the top of the row**.

    ``datasets`` is expected in increasing-size order, so reading a row downwards goes from the
    smallest dataset to the largest -- the same direction as reading the legend, which lists them
    in that order too. Hence the reversed endpoints: matplotlib's y grows upward.
    """
    if len(datasets) == 1:
        return {datasets[0]: 0.0}
    steps = np.linspace(row_spread / 2.0, -row_spread / 2.0, len(datasets))
    return {dataset: float(step) for dataset, step in zip(datasets, steps)}


def axis_limits(medians: pd.DataFrame, *, phase_families: Sequence[str]) -> tuple[float, float]:
    """Return shared log-axis limits enclosing the markers actually drawn.

    Include pre-embedding values only for phase-pair families; other stored stages
    must not widen axes for markers absent from the figure.
    """
    drawn = [medians["time_total"].to_numpy(dtype=float)]
    phase = medians.loc[medians["family"].isin(phase_families), "time_pre_embedding"]
    drawn.append(phase.to_numpy(dtype=float))
    values = np.concatenate(drawn)
    values = values[np.isfinite(values) & (values > 0.0)]
    low = float(values.min())
    high = float(values.max())
    return low / 2.5, high * 2.5


def legend_geometry(
    labels: list[str], figsize: tuple[float, float], *, style: RuntimeFigureStyle
) -> tuple[int, float, float]:
    """Return legend columns and measured dataset/combined band fractions from actual labels."""
    ncol = layout.legend_columns(labels, width=figsize[0], fontsize=style.dataset_legend_pt)
    datasets_in = layout.legend_height(len(labels), ncol, fontsize=style.dataset_legend_pt)
    phase_in = layout.legend_height(2, 2, fontsize=style.phase_legend_pt)
    height = figsize[1]
    return ncol, datasets_in / height, (datasets_in + phase_in + style.legend_pad_in) / height


def family_axes(counts: list[int], legend_fraction: float, figsize: tuple[float, float]):
    """One axis per family, heights proportional to the number of method rows.

    ``constrained_layout`` is off for this figure: it distributes space by content, which would give
    the two-row ``No integration`` panel the same height as the seven-row ones and break the shared
    row pitch. The rows have to be the same height in every panel or a reader comparing across
    panels is comparing different things.
    """
    import matplotlib.pyplot as plt

    # None inherits the application's constrained-layout setting; this manual
    # axes layout must disable it explicitly.
    fig = plt.figure(figsize=figsize, layout="none")
    left, right = 0.255, 0.995
    bottom, top = 0.052, 1.0 - legend_fraction
    gap = 0.016
    total_rows = sum(counts)
    usable = (top - bottom) - gap * (len(counts) - 1)

    axes = []
    cursor = top
    for count in counts:
        height = usable * count / total_rows
        axes.append(fig.add_axes((left, cursor - height, right - left, height)))
        cursor -= height + gap
    return fig, axes


def style_family_axis(
    ax,
    keys: list[str],
    xlim: tuple[float, float],
    header: str,
    *,
    bottom: bool,
    method_labels: Mapping[str, str],
    style: RuntimeFigureStyle,
) -> None:
    """Log axis, method names on y, and an unobtrusive family header.

    Only the bottom panel carries x tick labels. The panels share one scale, so repeating the
    decades five times would add no information and would collide with the header of the panel
    below -- the gridlines already carry the position across.
    """
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.75, len(keys) - 0.25)
    # Alternating bands group dataset markers belonging to the same method.
    for row in range(len(keys)):
        if row % 2 == 0:
            ax.axhspan(row - 0.5, row + 0.5, color=style.row_band, linewidth=0.0, zorder=0)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([method_labels[key] for key in reversed(keys)], fontsize=7.5)
    ax.tick_params(axis="y", length=0.0, pad=2.0)
    ax.tick_params(axis="x", which="both", labelbottom=bottom, length=2.0 if bottom else 0.0)
    ax.grid(axis="x", which="major", color="#d9d9d9", linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["bottom"].set_linewidth(0.5)
    # Place headers above the data area, distinguished from method labels by position and color.
    ax.text(
        0.0,
        1.005,
        header,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#595959",
    )


def dataset_label(dataset: str, n_obs: int | None = None, *, labels: Mapping[str, str]) -> str:
    """The dataset's name, with its cell count in brackets where one is known.

    The count is on the legend rather than in a separate table because it is the single strongest
    predictor of every runtime in the figure -- the spread across datasets within one method row is
    mostly this number -- and a reader who cannot see it has to take the ordering on trust.
    """
    name = labels.get(dataset, dataset)
    if n_obs is None:
        return name
    return f"{name} ({int(n_obs):,})"


def superscript(text: str) -> str:
    """``text`` as a mathtext superscript, in the figure's own face rather than the math default."""
    return f"$^{{\\mathrm{{{text}}}}}$" if text.isalpha() else f"$^{{{text}}}$"


def draw_figure_one_legends(
    fig,
    axes,
    datasets: tuple[str, ...],
    *,
    labels: Sequence[str],
    markers: Mapping[str, str],
    style: RuntimeFigureStyle,
    ncol: int | None = None,
    drop: float | None = None,
    device_key: str = "",
) -> None:
    """Two small legends: the seven datasets with their cell counts, and the two-phase key.

    There is deliberately no method legend. Twenty-five entries would be unreadable and unnecessary
    -- every method is named on the y axis, which is the whole reason marker shape was free to take
    the dataset channel.
    """
    from matplotlib.lines import Line2D

    # Use one legend entry per dataset; device annotations belong to individual points.
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=markers[dataset],
            markersize=style.point_size**0.5,
            markerfacecolor=style.palette[dataset],
            markeredgecolor="none",
            label=label,
        )
        for dataset, label in zip(datasets, labels)
    ]
    # Size legend columns using the longest dataset label to prevent overflow.
    if ncol is None:
        ncol = layout.legend_columns(labels, fontsize=style.dataset_legend_pt)
    if drop is None:
        drop = layout.legend_height(len(labels), ncol, fontsize=style.dataset_legend_pt) / (
            fig.get_figheight()
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=ncol,
        frameon=False,
        fontsize=style.dataset_legend_pt,
        handletextpad=0.3,
        columnspacing=0.9,
        borderaxespad=0.25,
    )

    # Use one figure-level phase key for all panels containing paired markers.
    phase_handles = [
        Line2D(
            [],
            [],
            linestyle="-",
            linewidth=0.5,
            color="#4d4d4d",
            marker="o",
            markersize=style.hollow_size**0.5,
            markerfacecolor="#ffffff",
            markeredgecolor="#4d4d4d",
            markeredgewidth=0.6,
            # The hollow marker is integration time; the filled marker is total time.
            # Backend duration is the gap, not the filled marker's position.
            label="Integration runtime",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=style.point_size**0.5,
            markerfacecolor="#4d4d4d",
            markeredgecolor="none",
            label="+ t-SNE runtime (total)",
        ),
    ]
    phase_legend = fig.legend(
        handles=phase_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0 - drop),
        ncol=2,
        frameon=False,
        fontsize=style.phase_legend_pt,
        handletextpad=0.35,
        columnspacing=1.6,
        borderaxespad=0.0,
    )

    # Place device text beside the phase key and recenter their measured combined extent.
    if device_key:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inverse = fig.transFigure.inverted()
        box = phase_legend.get_window_extent(renderer).transformed(inverse)
        probe = fig.text(0.5, 0.5, device_key, fontsize=style.phase_legend_pt, alpha=0.0)
        fig.canvas.draw()
        key_width = probe.get_window_extent(renderer).transformed(inverse).width
        probe.remove()
        gap = style.device_key_gap_in / fig.get_figwidth()
        phase_legend.set_bbox_to_anchor((0.5 - (key_width + gap) / 2, 1.0 - drop), fig.transFigure)
        fig.text(
            0.5 + (box.width + gap - key_width) / 2,
            (box.y0 + box.y1) / 2,
            device_key,
            ha="left",
            va="center",
            fontsize=style.phase_legend_pt,
            color="#4d4d4d",
        )


def device_key_text(marks: tuple[str, str], *, device_names: tuple[str, str]) -> str:
    """The key for ``marks``, or ``""`` when the marks already name the devices.

    ``("CPU", "GPU")`` is self-describing and a key restating it is a line of furniture that says
    nothing. ``("*", "**")`` is not, so it gets one.
    """
    if tuple(mark.upper() for mark in marks) == device_names:
        return ""
    return "      ".join(f"{superscript(mark)} {name}" for mark, name in zip(marks, device_names))


def device_mark(ax, x: float, y: float, text: str, colour: str, *, style: RuntimeFigureStyle):
    """Print one device mark to the right of its filled marker, and return it.

    Always to the right. The obvious alternative -- outer flank of each pair, so the marks bracket
    the row -- puts the GPU mark left of the GPU pair, where on the OT row it lands 0.08 decades
    from PBMC's total. That is 0.06 in for a mark that needs 0.13, so it is not a tuning problem.
    The gap between the two pairs is 0.65 decades and takes either mark comfortably.
    """
    return ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(style.device_mark_pad, style.device_mark_rise),
        ha="left",
        va="center",
        fontsize=style.device_mark_pt,
        color=colour,
        zorder=4,
        annotation_clip=False,
    )


def right_of(fig, ax, artist, pad_pt: float = 2.0) -> float:
    """Data-x just past ``artist``'s drawn right edge, so a line can start clear of it.

    Measured from the rendered text rather than estimated from the string length: the mark is 3
    characters in one variant and 2 in the other, and a per-variant constant is a constant that
    goes wrong the first time a third variant is added.
    """
    fig.canvas.draw()
    box = artist.get_window_extent(fig.canvas.get_renderer())
    return float(ax.transData.inverted().transform((box.x1 + pad_pt * fig.dpi / 72.0, 0.0))[0])
