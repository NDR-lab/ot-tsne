"""Publication page geometry, figure styling, saving, and print-size checks.

Figures are authored at final printed size. ``save_figure`` therefore disables save-time tight
cropping, which would change the declared dimensions and trigger silent LaTeX rescaling.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.text import Text

from ne_aligned_utils import config

# Publication page geometry

#: Full text width used by the manuscript template.
FULL_WIDTH = 5.500
#: Full text height. A taller figure cannot share a page with anything.
TEXT_HEIGHT = 9.000

#: Grid margins in inches reserve tick, label, title, and legend space at print size.
MARGIN_LEFT = 0.42
MARGIN_RIGHT = 0.02
MARGIN_TOP = 0.18
MARGIN_BOTTOM = 0.36
GAP = 0.08

#: Margins for an axis-free gallery panel: no ticks, no labels, a thin frame and a corner caption.
GALLERY_MARGINS: dict[str, float] = {
    "left": 0.05,
    "right": 0.02,
    "top": 0.14,
    "bottom": 0.05,
    "wgap": 0.04,
    "hgap": 0.04,
}

#: Fixed seed for anything that jitters, subsamples or repels labels, so output is deterministic.
SEED = 0

#: Smallest font the paper permits. Held in reserve for the
#: 0.948 in panels of the appendix benchmark grid.
MIN_FONT_PT = 7.0

#: Smallest marker diameter, in points. Shape carries construction -- circle for integrator plus
#: embedder, square for direct 2D, triangle for MultiMAP -- so a marker too small to read as a
#: shape loses a channel. ``ROLE_SIZE`` bottoms out at 22, i.e. 4.69 pt.
MIN_MARKER_PT = 4.5
WARN_MARKER_PT = 5.0

_STYLE_DIR = Path(__file__).resolve().parent / "styles"
BASE_STYLE = _STYLE_DIR / "style.mplstyle"
APPENDIX_STYLE = _STYLE_DIR / "style_appendix.mplstyle"

#: Resolving to this means no Arial-metric font was found and the figure will not match the page.
_FALLBACK_FAMILIES = frozenset({"DejaVu Serif", "DejaVu Sans"})


def full_width(height: float) -> tuple[float, float]:
    """Figure size spanning the full manuscript text width."""
    return (FULL_WIDTH, float(height))


@dataclass(frozen=True)
class GridGeometry:
    """The result of :func:`square_grid`: a figure size and the panel side it implies."""

    figsize: tuple[float, float]
    panel: float
    ncols: int
    nrows: int

    def describe(self) -> str:
        w, h = self.figsize
        return (
            f"{self.ncols}x{self.nrows} grid, {w:.3f} x {h:.3f} in, "
            f"panels {self.panel:.3f} in square"
        )


def square_grid(
    ncols: int,
    nrows: int,
    *,
    width: float = FULL_WIDTH,
    left: float = MARGIN_LEFT,
    right: float = MARGIN_RIGHT,
    top: float = MARGIN_TOP,
    bottom: float = MARGIN_BOTTOM,
    wgap: float = GAP,
    hgap: float = GAP,
    legend: float = 0.0,
) -> GridGeometry:
    """Return figure geometry and target panel side in inches for a square-panel grid.

    ``legend`` reserves vertical space in inches. Actual panel dimensions depend on
    rendered content and constrained layout; the target guides sizing decisions.
    """
    if ncols < 1 or nrows < 1:
        raise ValueError(f"need at least one row and column, got {ncols}x{nrows}.")
    usable = width - left - right - (ncols - 1) * wgap
    if usable <= 0:
        raise ValueError(
            f"{ncols} columns do not fit in {width} in once margins are taken; "
            f"usable width would be {usable:.3f} in."
        )
    panel = usable / ncols
    height = top + bottom + nrows * panel + (nrows - 1) * hgap + legend
    return GridGeometry(figsize=(width, height), panel=panel, ncols=ncols, nrows=nrows)


# Shared legends

#: Conservative Arial character width in em for legend sizing. Underestimation
#: can shrink constrained-layout axes; overestimation only adds a legend row.
_EM_PER_CHAR = 0.53

#: Legend furniture per entry, in em: handle length, handle-to-text pad, inter-column spacing.
_LEGEND_FURNITURE_EM = 1.2 + 0.4 + 0.8


def legend_columns(
    labels: Sequence[str], *, width: float = FULL_WIDTH, fontsize: float | None = None
) -> int:
    """Return the legend column count fitting ``width`` inches at the supplied font size.

    Use the actual rendered ``fontsize``; oversized legends can make constrained layout
    shrink data axes instead of overflow.
    """
    if not labels:
        return 1
    if fontsize is None:
        fontsize = float(mpl.rcParams["legend.fontsize"])
    fontsize = float(fontsize)
    longest = max(len(str(label)) for label in labels)
    entry_in = (longest * _EM_PER_CHAR + _LEGEND_FURNITURE_EM) * fontsize / 72.0
    return max(1, min(len(labels), int(width // entry_in)))


def legend_height(n_entries: int, ncol: int, *, fontsize: float | None = None) -> float:
    """Inches a shared legend of ``n_entries`` needs at ``ncol`` columns.

    :param fontsize: as :func:`legend_columns`; defaults to the style's legend size.
    """
    if fontsize is None:
        fontsize = float(mpl.rcParams["legend.fontsize"])
    fontsize = float(fontsize)
    rows = math.ceil(n_entries / max(1, ncol))
    return rows * fontsize * 1.55 / 72.0 + 0.06


def grid_with_legend(
    ncols: int,
    nrows: int,
    labels: Sequence[str],
    *,
    width: float = FULL_WIDTH,
    **grid_kwargs,
) -> tuple[Figure, np.ndarray, Axes, GridGeometry]:
    """Create a panel grid with a dedicated legend row.

    Return ``(fig, axes, legend_ax, geometry)``. Draw the shared legend in ``legend_ax``
    to reserve its space without squeezing panels through a floating legend.
    """
    ncol = legend_columns(labels, width=width)
    reserved = legend_height(len(labels), ncol)
    geom = square_grid(ncols, nrows, width=width, legend=reserved, **grid_kwargs)

    fig = plt.figure(figsize=geom.figsize)
    panels_in = geom.figsize[1] - reserved
    gs = fig.add_gridspec(
        nrows + 1, ncols, height_ratios=[*([panels_in / nrows] * nrows), reserved]
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)])

    legend_ax = fig.add_subplot(gs[nrows, :])
    legend_ax.axis("off")
    return fig, (axes[0] if nrows == 1 else axes), legend_ax, geom


# Style


def _style_paths(appendix: bool) -> list[Path]:
    styles = [BASE_STYLE]
    if appendix:
        styles.append(APPENDIX_STYLE)
    missing = [p for p in styles if not p.exists()]
    if missing:
        raise FileNotFoundError(f"style file(s) not found: {missing}")
    return styles


def use_paper_style(appendix: bool = False) -> list[Path]:
    """Apply base and optional appendix styles globally.

    This mutates rcParams without restoration. Use ``paper_style`` for a scoped context
    in library code or tests.
    """
    styles = _style_paths(appendix)
    plt.style.use([str(p) for p in styles])
    return styles


def paper_style(appendix: bool = False):
    """The paper style as a context manager, restoring rcParams on exit.

    ``with paper_style(): ...`` is the form to use in tests, in library code and anywhere else
    that shares a process with code expecting matplotlib's defaults.
    """
    return plt.style.context([str(p) for p in _style_paths(appendix)])


def resolved_font_family() -> str:
    """The font family matplotlib actually found, not the one requested.

    ``font.sans-serif`` is a preference list. If none of Arial, Helvetica or the metric-compatible
    clones is installed, matplotlib falls through to DejaVu Sans without warning and the figure
    stops matching the rest of the set. :func:`print_preview` treats that as a failure.
    """
    from matplotlib.font_manager import FontProperties, findfont

    prop = FontProperties(family=mpl.rcParams["font.family"])
    return FontProperties(fname=findfont(prop)).get_name()


# Saving and verification


@dataclass
class PreviewReport:
    """What :func:`print_preview` measured. Truthy checks live in :meth:`raise_for_problems`."""

    path: Path
    declared: tuple[float, float]
    measured: tuple[float, float]
    min_font_pt: float
    min_marker_pt: float | None
    font_family: str
    kind: str
    problems: list[str] = field(default_factory=list)

    def raise_for_problems(self) -> None:
        if self.problems:
            raise LegibilityError(
                f"{self.path.name} is not legible at its final printed size:\n  - "
                + "\n  - ".join(self.problems)
            )

    def describe(self) -> str:
        marker = "n/a" if self.min_marker_pt is None else f"{self.min_marker_pt:.2f} pt"
        return (
            f"{self.path.name}: {self.measured[0]:.3f} x {self.measured[1]:.3f} in, "
            f"smallest text {self.min_font_pt:.1f} pt, smallest marker {marker}, "
            f"font {self.font_family}"
        )


class LegibilityError(RuntimeError):
    """A figure would be unreadable at the size it is going to be printed."""


def print_preview(
    fig: Figure,
    path: Path,
    *,
    kind: str = "glyph",
    dpi: int = 600,
    min_marker_pt: float | None = None,
    measure_path: Path | None = None,
) -> PreviewReport:
    """Render or inspect a PNG at final print size and report legibility measurements.

    ``kind="glyph"`` checks marker diameter; ``"density"`` omits that check for point
    clouds. ``measure_path`` reuses an existing PNG and requires its actual ``dpi``.
    """
    if kind not in {"glyph", "density"}:
        raise ValueError(f"kind must be 'glyph' or 'density', got {kind!r}.")

    if measure_path is None:
        png = path.with_suffix(".preview.png")
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=dpi, bbox_inches=None)
    else:
        png = measure_path

    from PIL import Image

    with Image.open(png) as image:
        measured = (image.width / dpi, image.height / dpi)

    declared = (float(fig.get_figwidth()), float(fig.get_figheight()))

    fonts = [
        t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible()
    ]
    min_font = float(min(fonts)) if fonts else float("nan")

    diameters: list[float] = []
    for collection in fig.findobj(PathCollection):
        sizes = collection.get_sizes()
        diameters.extend(float(s) ** 0.5 for s in sizes if s > 0)
    min_marker = min(diameters) if diameters else None

    family = resolved_font_family()

    problems: list[str] = []
    for axis, got, want in zip("wh", measured, declared):
        if abs(got - want) > 0.01:
            problems.append(
                f"{axis} is {got:.3f} in on disk but {want:.3f} in was declared -- something "
                "re-cropped the figure, which means LaTeX will rescale it"
            )
    if fonts and min_font < MIN_FONT_PT:
        problems.append(
            f"smallest text is {min_font:.1f} pt, below the {MIN_FONT_PT:.0f} pt floor "
            "(the manuscript's minimum readable size)"
        )
    if family in _FALLBACK_FAMILIES:
        problems.append(
            f"font resolved to {family}, which means no Arial-metric font was found; "
            "the figure will not match the manuscript typography"
        )
    # Allow an explicit marker-size floor for plots whose marker shapes carry no distinction.
    floor = MIN_MARKER_PT if min_marker_pt is None else float(min_marker_pt)
    if kind == "glyph" and min_marker is not None and min_marker < floor:
        problems.append(
            f"smallest marker is {min_marker:.2f} pt across, below the {floor} pt floor; "
            "marker shape carries the construction channel and stops reading below this"
        )

    report = PreviewReport(
        path=path,
        declared=declared,
        measured=measured,
        min_font_pt=min_font,
        min_marker_pt=min_marker,
        font_family=family,
        kind=kind,
        problems=problems,
    )
    if (
        kind == "glyph"
        and min_marker is not None
        and floor <= min_marker < max(WARN_MARKER_PT, floor)
    ):
        print(f"  warning: smallest marker {min_marker:.2f} pt is close to the {floor} pt floor")
    return report


def save_figure(
    fig: Figure,
    name: str,
    *,
    appendix: bool = False,
    preview: bool = True,
    kind: str = "glyph",
    dpi: int = 600,
    close: bool = True,
    subdir: str | None = None,
    min_marker_pt: float | None = None,
    root: Path | None = None,
    formats: tuple[str, ...] = ("png",),
) -> Path:
    """Save a figure at its declared size and check the PNG's print legibility.

    ``root`` overrides the default output directory; ``subdir`` appends a folder.
    The filename is used as supplied, including any desired sequence or appendix prefix.
    ``formats`` may also request vector output; the returned and measured artifact is
    the PNG. Tight cropping is disabled to preserve physical dimensions.
    """
    # Use the supplied filename unchanged; callers own ordering and appendix prefixes.
    stem = name
    base = config.PAPER_FIGURES_DIR if root is None else Path(root)
    path = (base if subdir is None else base / subdir) / f"{stem}.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    if "png" not in formats:
        raise ValueError(
            f"formats={formats!r} omits 'png'. The PNG is the measured artifact -- "
            "print_preview reads pixel dimensions, which a vector file does not have -- and it is "
            "the path this function returns. Ask for ('png', 'pdf') rather than ('pdf',)."
        )
    fig.savefig(path, dpi=dpi, bbox_inches=None)
    for suffix in formats:
        if suffix == "png":
            continue
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches=None)

    if preview:
        # Measure the file just written rather than rendering a second copy of it.
        report = print_preview(
            fig, path, dpi=dpi, kind=kind, min_marker_pt=min_marker_pt, measure_path=path
        )
        print(f"  {report.describe()}")
        report.raise_for_problems()

    if close:
        plt.close(fig)
    return path


def style_axes(ax, *, outward: float = 2.0) -> None:
    """Offset the spines so axes that do not meet at zero show a visible corner gap."""
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", outward))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
