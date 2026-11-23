"""Hyperparameter and initialisation figures, drawn as changes from a reference configuration.

The swept settings are the axes (rel_reg across, k up), never metric-vs-metric paths, and the
colours encode the size and sign of a change, not a method. Method colours from the paper registry
are deliberately not used here, so these figures cannot be read as a comparison of OT-tSNE
variants.

- Change colour: a diverging blue (better) / red (worse) scale with a grey midpoint, symmetric
  around zero. Cell text gives the change ×100, black where it counts and grey italic where not:
  below 1 point or within 2 SE of seed noise, for one dataset or for the mean over datasets (see
  :mod:`ne_aligned_utils.paper.summaries.hyperparameters`).
- The default point is outlined in black in the sensitivity grids. No cell is marked as best.
- Initialisations have fixed colours and marker shapes; every colour also has its own shape.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from ne_aligned_utils.paper import layout, registry
from ne_aligned_utils.paper.rendering.metrics import metric_span, round_ticks, tick_labels
from ne_aligned_utils.paper.rendering.styles import DATASET_COLORS
from ne_aligned_utils.paper.summaries.hyperparameters import (
    BASELINE_INIT,
    DEFAULT_POINT,
    init_order,
)

#: Worse (red) - no change (grey) - better (blue). All four metrics are higher-is-better.
CHANGE_CMAP = LinearSegmentedColormap.from_list(
    "change", ["#b2182b", "#ef8a62", "#e6e6e6", "#67a9cf", "#2166ac"]
)
#: The light half of CHANGE_CMAP, for cells whose text is always black or grey.
CHANGE_CMAP_LIGHT = LinearSegmentedColormap.from_list(
    "change_light", ["#ef8a62", "#e6e6e6", "#67a9cf"]
)
#: Mean rank: lighter is a worse (higher) rank.
RANK_CMAP = LinearSegmentedColormap.from_list("rank", ["#f2f2f2", "#9ecae1", "#2171b5", "#08306b"])
#: Init -> (colour, marker). Random is the neutral baseline.
INIT_STYLE: dict[str, tuple[str, str]] = {
    "random": ("#6f6f6f", "o"),
    "spectral": ("#2a78d6", "s"),
    "eot": ("#1baf7a", "^"),
    "pca": ("#6250d6", "D"),
}
INIT_LABELS: dict[str, str] = {
    "random": "random",
    "spectral": "spectral",
    "eot": "EOT",
    "pca": "PCA",
}
METRIC_LABELS: dict[str, str] = {
    "overall": "Overall",  # the mean of bio and mixing; the caption says so
    "bio": "Bio conservation",
    "mixing": "Batch mixing",
    **registry.METRICS,
}
#: Short names for the key panel of the per-dataset grids.
KEY_LABELS: dict[str, str] = {**METRIC_LABELS, "overall": "Overall"}
SETTING_LABELS: dict[str, str] = {
    "rel_reg": "r",
    "effective_k": "k",
}
#: Titles, axis labels, ticks, colorbars and legends take the paper style's sizes
#: (paper/styles/style.mplstyle), as the benchmark figures do. Only text drawn inside panels
#: (cell values, annotations) has a fixed size: TEXT_PT, the style's tick-label size.
TEXT_PT = 8.0
#: The printed change in the pooled grids (the per-dataset grids keep TEXT_PT).
POOLED_VALUE_PT = 9.0
NOISE_TEXT = "#8a8a8a"


def _dataset_strip(ax, dataset: str) -> None:
    """Name a row's dataset on its right edge, as the benchmark metric figures do."""
    strip = ax.twinx()
    strip.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        strip.spines[side].set_visible(False)
    strip.set_ylabel(
        registry.DATASET_LABELS.get(dataset, dataset), rotation=270, labelpad=10, va="center"
    )


def _shared_colorbar(fig, image, axes, label: str) -> None:
    """One short, narrow colorbar for all ``axes``, centred on the figure's right edge."""
    cb = fig.colorbar(image, ax=list(axes), shrink=0.45, aspect=30, pad=0.02)
    cb.ax.tick_params(length=2)
    cb.set_label(label)


def _fmt(value: float) -> str:
    return f"{value:g}"


def _change_norm(values: np.ndarray) -> TwoSlopeNorm:
    lim = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0.0
    lim = max(lim, 1e-3)
    return TwoSlopeNorm(0.0, -lim, lim)


def _change_texts(delta: np.ndarray) -> np.ndarray:
    """Signed changes to one decimal; "0.0" for one that rounds to zero, "" for a missing one."""
    texts = np.empty(delta.shape, dtype=object)
    for idx, value in np.ndenumerate(delta):
        if np.isnan(value):
            texts[idx] = ""
        else:
            texts[idx] = "0.0" if round(value, 1) == 0 else f"{value:+.1f}"
    return texts


def _ink(rgba) -> str:
    """Black text on light cells, white on dark ones."""
    r, g, b = rgba[:3]
    return "#000000" if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else "#ffffff"


def _heatmap(
    ax,
    values: np.ndarray,
    texts: np.ndarray,
    quiet: np.ndarray,
    *,
    cmap,
    norm,
    outline: tuple[int, int] | None,
    xticklabels: Sequence[str],
    yticklabels: Sequence[str],
    fontsize: float = TEXT_PT,
    best: Sequence[tuple[int, int]] = (),
    black_text: bool = False,
):
    """Cells with printed values; ``quiet`` cells (within noise) get grey italic text.

    The other cells get black or white text by cell darkness, or always black with
    ``black_text``.
    """
    image = ax.imshow(values, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    for (i, j), text in np.ndenumerate(texts):
        if not text:
            continue
        if quiet[i, j]:
            colour = NOISE_TEXT
        elif black_text:
            colour = "#000000"
        else:
            colour = _ink(cmap(norm(values[i, j])))
        ax.text(
            j,
            i,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=colour,
            fontstyle="italic" if quiet[i, j] else "normal",
            linespacing=0.95,
        )
    if outline is not None:
        i, j = outline
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, lw=1.4, ec="#000000"))
    for i, j in best:
        ax.add_patch(
            Rectangle(
                (j - 0.38, i - 0.38),
                0.76,
                0.76,
                fill=False,
                lw=1.0,
                ec="#000000",
                ls=(0, (2.0, 1.5)),
            )
        )
    ax.set_xticks(range(len(xticklabels)), xticklabels)
    ax.set_yticks(range(len(yticklabels)), yticklabels)
    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _save(fig, name: str, output_root: Path) -> Path:
    return layout.save_figure(fig, name, root=output_root, kind="glyph", min_marker_pt=4.5)


def _grid_axes(
    n_panels: int,
    ncols: int,
    panel_in: float,
    *,
    extra_h: float = 0.0,
    centre_last: bool = False,
):
    """Figure and axes for ``n_panels`` panels in rows of ``ncols``.

    ``centre_last``: a last row with a single panel puts it in the middle of the row instead
    of the first column; the axes then come back as a (1, n_panels) array with no spare panels.
    """
    nrows = int(np.ceil(n_panels / ncols))
    width = layout.FULL_WIDTH
    height = nrows * panel_in + extra_h
    if not (centre_last and n_panels % ncols == 1 and nrows > 1):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(width, height), squeeze=False, layout="constrained"
        )
        return fig, axes
    # Twice as many grid columns: each panel spans two, so one panel can sit centred.
    fig = plt.figure(figsize=(width, height), layout="constrained")
    spec = fig.add_gridspec(nrows, 2 * ncols)
    axes = [
        fig.add_subplot(spec[i // ncols, 2 * (i % ncols) : 2 * (i % ncols) + 2])
        for i in range(n_panels - 1)
    ]
    axes.append(fig.add_subplot(spec[nrows - 1, ncols - 1 : ncols + 1]))
    return fig, np.array(axes, dtype=object).reshape(1, -1)


def _settings_grid(frame: pd.DataFrame, column: str, rel_regs, ks) -> np.ndarray:
    table = frame.pivot(index="effective_k", columns="rel_reg", values=column)
    return table.reindex(index=ks, columns=rel_regs).to_numpy()


def _default_cell(rel_regs, ks, default) -> tuple[int, int] | None:
    rows = [i for i, k in enumerate(ks) if k == default[1]]
    cols = [j for j, r in enumerate(rel_regs) if np.isclose(r, default[0])]
    return (rows[0], cols[0]) if rows and cols else None


# A: per-dataset grids


def dataset_grids(
    changes: pd.DataFrame,
    metric: str,
    *,
    datasets: Sequence[str],
    output_root: Path,
    default: tuple[float, int] = DEFAULT_POINT,
    ncols: int = 3,
) -> Path:
    """One 5×5 grid per dataset for one score: change from the default point, ×100.

    Black where the change is beyond the noise rule, grey italic where not; the default point
    is outlined.
    """
    rows = changes[changes["metric"] == metric]
    rel_regs = sorted(rows["rel_reg"].unique())
    ks = sorted(rows["effective_k"].unique())
    norm = _change_norm(rows["delta"].to_numpy() * 100)
    n_slots = len(datasets) + 1
    fig, axes = _grid_axes(n_slots, ncols, 1.85, extra_h=0.25)
    flat = axes.ravel()
    image = None
    for ax, dataset in zip(flat, datasets, strict=False):
        sub = rows[rows["dataset"] == dataset]
        delta = _settings_grid(sub, "delta", rel_regs, ks) * 100
        direction = _settings_grid(sub, "direction", rel_regs, ks)
        image = _heatmap(
            ax,
            delta,
            _change_texts(delta),
            direction == 0,
            cmap=CHANGE_CMAP_LIGHT,
            norm=norm,
            outline=_default_cell(rel_regs, ks, default),
            xticklabels=[_fmt(v) for v in rel_regs],
            yticklabels=[_fmt(v) for v in ks],
            black_text=True,
        )
        ax.set_title(registry.DATASET_LABELS.get(dataset, dataset))
        ax.set_xlabel(SETTING_LABELS["rel_reg"])
        ax.set_ylabel("k")
    key_ax = flat[len(datasets)]
    for ax in flat[len(datasets) + 1 :]:
        ax.axis("off")
    _change_key(fig, key_ax, image, f"{KEY_LABELS[metric]}\nchange from the\ndefault (×100)")
    return _save(fig, f"sensitivity-datasets-{metric}", output_root)


def _change_key(fig, ax, image, label: str) -> None:
    """Colour bar in a spare panel slot."""
    ax.axis("off")
    bar = ax.inset_axes([0.08, 0.4, 0.84, 0.1])
    cb = fig.colorbar(image, cax=bar, orientation="horizontal")
    cb.ax.tick_params(length=2)
    cb.set_label(label)
    cb.ax.xaxis.set_label_position("top")


# C: pooled grids


def pooled_grids(
    pooled: pd.DataFrame,
    metrics: Sequence[str],
    *,
    name: str,
    output_root: Path,
    default: tuple[float, int] = DEFAULT_POINT,
    ncols: int = 2,
    centre_last: bool = False,
) -> Path:
    """Mean change over datasets per (rel_reg, k), ×100.

    Black where the mean change is beyond the noise rule (``pooled_grid``'s ``direction``), grey
    italic where not; the default point is outlined. All panels share one colour scale and one
    colorbar.
    """
    rel_regs = sorted(pooled["rel_reg"].unique())
    ks = sorted(pooled["effective_k"].unique())
    n_datasets = int(pooled["n_datasets"].max())
    fig, axes = _grid_axes(len(metrics), ncols, 2.55, extra_h=0.3, centre_last=centre_last)
    norm = _change_norm(pooled[pooled["metric"].isin(metrics)]["delta"].to_numpy() * 100)
    image = None
    for ax, metric in zip(axes.ravel(), metrics, strict=False):
        sub = pooled[pooled["metric"] == metric]
        delta = _settings_grid(sub, "delta", rel_regs, ks) * 100
        direction = _settings_grid(sub, "direction", rel_regs, ks)
        image = _heatmap(
            ax,
            delta,
            _change_texts(delta),
            direction == 0,
            cmap=CHANGE_CMAP_LIGHT,
            norm=norm,
            outline=_default_cell(rel_regs, ks, default),
            xticklabels=[_fmt(v) for v in rel_regs],
            yticklabels=[_fmt(v) for v in ks],
            fontsize=POOLED_VALUE_PT,
            black_text=True,
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel(SETTING_LABELS["rel_reg"])
        ax.set_ylabel("k")
    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")
    _shared_colorbar(
        fig,
        image,
        axes.ravel()[: len(metrics)].tolist(),
        f"mean change over {n_datasets} datasets (×100)",
    )
    return _save(fig, name, output_root)


# B: one-at-a-time lines


def line_panels(
    lines: Mapping[str, pd.DataFrame],
    metrics: Sequence[str],
    *,
    output_root: Path,
    name: str = "sensitivity-lines",
    default: tuple[float, int] = DEFAULT_POINT,
    datasets: Sequence[str] | None = None,
) -> Path:
    """Change from default along each one-at-a-time line: one coloured line per dataset, black mean.

    Filled markers: the change exceeds seed noise; open markers: it does not.
    """
    sweeps = list(lines)
    if datasets is None:
        datasets = list(dict.fromkeys(d for line in lines.values() for d in line["dataset"]))
    fig, axes = plt.subplots(
        len(metrics),
        len(sweeps),
        figsize=(layout.FULL_WIDTH, 1.75 * len(metrics) + 0.45),
        squeeze=False,
        sharey="row",
        layout="constrained",
    )
    for col, sweep in enumerate(sweeps):
        line = lines[sweep]
        values = sorted(line[sweep].unique())
        held = f"k = {default[1]}" if sweep == "rel_reg" else f"r = {_fmt(default[0])}"
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            sub = line[line["metric"] == metric]
            for dataset in datasets:
                data = sub[sub["dataset"] == dataset].sort_values(sweep)
                if data.empty:
                    continue
                colour = DATASET_COLORS.get(dataset, "#8c8c8c")
                ax.plot(data[sweep], data["delta"] * 100, color=colour, lw=0.9, zorder=1)
                real = data["direction"] != 0
                ax.scatter(
                    data.loc[real, sweep],
                    data.loc[real, "delta"] * 100,
                    s=22,
                    color=colour,
                    zorder=2,
                    linewidths=0,
                )
                ax.scatter(
                    data.loc[~real, sweep],
                    data.loc[~real, "delta"] * 100,
                    s=22,
                    facecolors="#ffffff",
                    edgecolors=colour,
                    linewidths=0.8,
                    zorder=2,
                )
            mean = sub.groupby(sweep)["delta"].mean() * 100
            ax.plot(mean.index, mean.values, color="#000000", lw=1.6, zorder=3)
            ax.scatter(mean.index, mean.values, s=26, color="#000000", zorder=4, linewidths=0)
            ax.axhline(0, color="#6f6f6f", lw=0.6, ls="--", zorder=0)
            reference = default[0] if sweep == "rel_reg" else default[1]
            ax.axvline(reference, color="#6f6f6f", lw=0.6, ls=":", zorder=0)
            ax.set_xscale("log")
            ax.set_xticks(values, [_fmt(v) for v in values])
            ax.minorticks_off()
            layout.style_axes(ax)
            if col == 0:
                ax.set_ylabel(f"{METRIC_LABELS[metric]}\nchange (×100)")
            if row == len(metrics) - 1:
                ax.set_xlabel(f"{SETTING_LABELS[sweep]}  ({held})")
    handles = [
        Line2D([], [], color=DATASET_COLORS.get(d, "#8c8c8c"), lw=0.9, marker="o", ms=4, mec="none")
        for d in datasets
    ]
    labels = [registry.DATASET_LABELS.get(d, d) for d in datasets]
    handles += [
        Line2D([], [], color="#000000", lw=1.6, marker="o", ms=4),
        Line2D([], [], ls="none", marker="o", ms=4, mfc="#6f6f6f", mec="none"),
        Line2D([], [], ls="none", marker="o", ms=4, mfc="#ffffff", mec="#6f6f6f"),
    ]
    labels += ["mean over datasets", "beyond seed noise", "within seed noise"]
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    return _save(fig, name, output_root)


# D: OT-tSNE's standing among all methods (not relative to the default)

BLOCK_TITLES = {"all": "Overall", "bio": "Bio conservation", "mixing": "Batch mixing"}


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def rank_grids(
    ranks: pd.DataFrame,
    *,
    output_root: Path,
    default: tuple[float, int] = DEFAULT_POINT,
    blocks: Sequence[str] = ("all", "bio", "mixing"),
) -> Path:
    """OT-tSNE's placing among all methods at each setting, every other method as reported.

    Cell: the placing by mean rank over datasets (1st = best mean rank), and below it the lead in
    mean rank over the best other method (negative: behind it).
    """
    sweep = ranks[ranks["source"] == "sweep"]
    reported = ranks[ranks["source"] == "reported"].set_index("block")
    rel_regs = sorted(sweep["rel_reg"].unique())
    ks = sorted(sweep["effective_k"].unique())
    n_methods = int(sweep["n_methods"].max())
    n_settings = len(rel_regs) * len(ks)
    normalization = str(ranks["normalization"].iloc[0])
    worst = int(sweep["placing"].max())
    norm = Normalize(1, max(worst, 4))
    fig, axes = plt.subplots(
        1, len(blocks), figsize=(layout.FULL_WIDTH, 2.45), squeeze=False, layout="constrained"
    )
    for ax, block in zip(axes[0], blocks, strict=True):
        sub = sweep[sweep["block"] == block]
        placing = _settings_grid(sub, "placing", rel_regs, ks)
        lead = _settings_grid(sub, "lead", rel_regs, ks)
        texts = np.empty(placing.shape, dtype=object)
        for idx, value in np.ndenumerate(placing):
            texts[idx] = f"{_ordinal(int(value))}\n{lead[idx]:+.1f}"
        image = _heatmap(
            ax,
            placing,
            texts,
            np.zeros(placing.shape, dtype=bool),
            cmap=RANK_CMAP.reversed(),
            norm=norm,
            outline=_default_cell(rel_regs, ks, default),
            xticklabels=[_fmt(v) for v in rel_regs],
            yticklabels=[_fmt(v) for v in ks],
        )
        first = int((sub["placing"] == 1).sum())
        own = int(reported.loc[block, "placing"]) if block in reported.index else None
        ax.set_title(
            f"{BLOCK_TITLES[block]}\n1st in {first} of {n_settings}"
            + (f"\n(reported: {_ordinal(own)})" if own else ""),
        )
        ax.set_xlabel(SETTING_LABELS["rel_reg"])
        ax.set_ylabel("k" if block == blocks[0] else "")
    cb = fig.colorbar(image, ax=axes[0].tolist(), shrink=0.9, pad=0.02)
    cb.ax.tick_params(length=2)
    cb.ax.invert_yaxis()
    cb.set_label(f"OT-tSNE's placing of {n_methods}")
    fig.supxlabel(
        f"Cell: OT-tSNE's placing among {n_methods} methods by mean rank over datasets,\n"
        "and its lead in mean rank over the best other method. Only OT-tSNE's setting\n"
        "changes; every other method is as reported. Black box: default.",
    )
    return _save(fig, f"sensitivity-placing-{normalization}", output_root)


_BLOCK_STYLE = {
    "all": dict(color="#000000", ls="-", marker="o"),
    "bio": dict(color="#6f6f6f", ls="--", marker="s"),
    "mixing": dict(color="#6f6f6f", ls=":", marker="^"),
}


def placing_lines(
    ranks: pd.DataFrame,
    *,
    output_root: Path,
    default: tuple[float, int] = DEFAULT_POINT,
    blocks: Sequence[str] = ("all", "bio", "mixing"),
) -> Path:
    """OT-tSNE's placing among all methods along each one-at-a-time line."""
    sweep = ranks[ranks["source"] == "sweep"]
    normalization = str(ranks["normalization"].iloc[0])
    n_methods = int(sweep["n_methods"].max())
    fig, axes = plt.subplots(
        1, 2, figsize=(layout.FULL_WIDTH, 2.3), squeeze=False, sharey=True, layout="constrained"
    )
    lines = (
        ("rel_reg", sweep[np.isclose(sweep["effective_k"], default[1])], f"k = {default[1]}"),
        (
            "effective_k",
            sweep[np.isclose(sweep["rel_reg"], default[0])],
            f"r = {_fmt(default[0])}",
        ),
    )
    top = int(sweep["placing"].max())
    for ax, (sweep_name, line, held) in zip(axes[0], lines, strict=True):
        values = sorted(line[sweep_name].unique())
        for block in blocks:
            data = line[line["block"] == block].sort_values(sweep_name)
            ax.plot(
                data[sweep_name],
                data["placing"],
                lw=1.1,
                ms=4.5,
                label=BLOCK_TITLES[block],
                **_BLOCK_STYLE[block],
            )
        ax.axvline(
            default[0] if sweep_name == "rel_reg" else default[1],
            color="#6f6f6f",
            lw=0.6,
            ls=":",
            zorder=0,
        )
        ax.set_xscale("log")
        ax.set_xticks(values, [_fmt(v) for v in values])
        ax.minorticks_off()
        ax.set_ylim(top + 0.5, 0.5)
        ax.set_yticks(range(1, top + 1), [_ordinal(i) for i in range(1, top + 1)])
        ax.set_xlabel(f"{SETTING_LABELS[sweep_name]}  ({held})")
        layout.style_axes(ax)
    axes[0, 0].set_ylabel(f"OT-tSNE's placing of {n_methods}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    return _save(fig, f"sensitivity-placing-lines-{normalization}", output_root)


# Absolute scores in the context of the other methods


def _others(context: pd.DataFrame, metric: str) -> pd.DataFrame:
    return context[(context["metric"] == metric) & (context["method_key"] != "ot")]


def absolute_lines(
    changes: pd.DataFrame,
    context: pd.DataFrame,
    metric: str,
    *,
    datasets: Sequence[str],
    output_root: Path,
    default: tuple[float, int] = DEFAULT_POINT,
) -> Path:
    """OT-tSNE's actual score along each one-at-a-time line, per dataset, among the others.

    Black: OT-tSNE's seed mean ± SD. Dashed: the best other method on that dataset; dotted: the
    median of all other methods (each as reported).
    """
    rows = changes[changes["metric"] == metric]
    others = _others(context, metric)
    lines = (
        ("rel_reg", rows[np.isclose(rows["effective_k"], default[1])], f"k = {default[1]}"),
        (
            "effective_k",
            rows[np.isclose(rows["rel_reg"], default[0])],
            f"r = {_fmt(default[0])}",
        ),
    )
    fig, axes = plt.subplots(
        len(datasets),
        2,
        figsize=(layout.FULL_WIDTH, 0.95 * len(datasets) + 0.75),
        squeeze=False,
        sharex="col",
        layout="constrained",
    )
    for r, dataset in enumerate(datasets):
        band = others[others["dataset"] == dataset]["mean"]
        for c, (sweep_name, line, held) in enumerate(lines):
            ax = axes[r, c]
            data = line[line["dataset"] == dataset].sort_values(sweep_name)
            ax.axhline(band.max(), color="#6f6f6f", lw=0.8, ls="--", zorder=1)
            ax.axhline(band.median(), color="#9a9a9a", lw=0.8, ls=":", zorder=1)
            ax.errorbar(
                data[sweep_name],
                data["mean"],
                yerr=data["sd"],
                color="#000000",
                lw=1.1,
                elinewidth=0.6,
                capsize=0,
                zorder=2,
            )
            ax.scatter(
                data[sweep_name], data["mean"], s=22, color="#000000", zorder=3, linewidths=0
            )
            ax.axvline(
                default[0] if sweep_name == "rel_reg" else default[1],
                color="#6f6f6f",
                lw=0.6,
                ls=":",
                zorder=0,
            )
            ax.set_xscale("log")
            values = sorted(line[sweep_name].unique())
            ax.set_xlim(values[0] / 1.35, values[-1] * 1.35)
            ax.set_xticks(values, [_fmt(v) for v in values])
            ax.minorticks_off()
            layout.style_axes(ax)
            if c == len(lines) - 1:
                _dataset_strip(ax, dataset)
            if r == len(datasets) - 1:
                ax.set_xlabel(f"{SETTING_LABELS[sweep_name]}  ({held})")
    handles = [
        Line2D([], [], color="#000000", lw=1.1, marker="o", ms=4),
        Line2D([], [], color="#6f6f6f", lw=0.8, ls="--"),
        Line2D([], [], color="#9a9a9a", lw=0.8, ls=":"),
    ]
    labels = ["OT-tSNE (seed mean ± SD)", "best other method", "median of other methods"]
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(f"{METRIC_LABELS[metric]} (raw score)")
    return _save(fig, f"sensitivity-absolute-{metric}", output_root)


def robustness_strip(
    changes: pd.DataFrame,
    context: pd.DataFrame,
    metric: str,
    *,
    datasets: Sequence[str],
    output_root: Path,
    default: tuple[float, int] = DEFAULT_POINT,
) -> Path:
    """Per dataset: OT-tSNE at every grid setting (black) against every other method (grey).

    The text at the right counts the other methods that OT-tSNE's worst setting still beats.
    """
    rows = changes[changes["metric"] == metric]
    others = _others(context, metric)
    fig, ax = plt.subplots(
        figsize=(layout.FULL_WIDTH, 0.42 * len(datasets) + 0.85), layout="constrained"
    )
    for i, dataset in enumerate(datasets):
        y = len(datasets) - 1 - i
        theirs = others[others["dataset"] == dataset]["mean"].to_numpy()
        ours = rows[rows["dataset"] == dataset]
        grid = ours["mean"].to_numpy()
        is_default = np.isclose(ours["rel_reg"], default[0]) & (ours["effective_k"] == default[1])
        ax.scatter(
            theirs,
            np.full(len(theirs), y - 0.13),
            s=22,
            color="#9a9a9a",
            alpha=0.8,
            linewidths=0,
            zorder=2,
        )
        ax.plot([grid.min(), grid.max()], [y + 0.13, y + 0.13], color="#000000", lw=1.0, zorder=2)
        ax.scatter(
            grid,
            np.full(len(grid), y + 0.13),
            s=21,
            color="#000000",
            alpha=0.55,
            linewidths=0,
            zorder=3,
        )
        ax.scatter(
            ours.loc[is_default, "mean"],
            [y + 0.13],
            s=40,
            marker="D",
            facecolors="#ffffff",
            edgecolors="#000000",
            linewidths=1.0,
            zorder=4,
        )
        beaten = int((theirs < grid.min()).sum())
        ax.text(
            1.01,
            y,
            f"worst setting beats {beaten}/{len(theirs)}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=TEXT_PT,
        )
    ax.set_yticks(
        range(len(datasets)),
        [registry.DATASET_LABELS.get(d, d) for d in reversed(datasets)],
    )
    ax.set_ylim(-0.6, len(datasets) - 0.4)
    ax.set_xlabel(f"{METRIC_LABELS[metric]} (raw score)")
    layout.style_axes(ax)
    handles = [
        Line2D([], [], ls="none", marker="o", ms=4.5, mfc="#000000", mec="none", alpha=0.55),
        Line2D([], [], ls="none", marker="D", ms=5, mfc="#ffffff", mec="#000000"),
        Line2D([], [], ls="none", marker="o", ms=4.5, mfc="#9a9a9a", mec="none"),
    ]
    labels = ["OT-tSNE, each of the 25 settings", "OT-tSNE, default", "other methods (as reported)"]
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    return _save(fig, f"sensitivity-strip-{metric}", output_root)


# I1: change from the random init


def _variant_label(method: str) -> str:
    entry = registry.resolve(method)
    return "default" if method == "ot" else entry.legend_label.replace(" diagonal", "")


def init_grids(
    table: pd.DataFrame,
    metrics: Sequence[str],
    *,
    methods: Sequence[str],
    name: str,
    output_root: Path,
    pooled: bool,
    datasets: Sequence[str] = (),
    centre_last: bool = False,
) -> Path:
    """Variants (rows) × inits (columns): change from the random init, ×100.

    ``pooled``: one panel per score, the mean change over datasets. Otherwise one panel per
    dataset for the first score in ``metrics``. A change is printed black where it is beyond the
    noise rule (``direction``) and grey italic where not, on light colours.
    """
    inits = [i for i in init_order(table["init"].unique()) if i != BASELINE_INIT]
    xt = [INIT_LABELS.get(i, i) for i in inits]
    yt = [_variant_label(m) for m in methods]
    if pooled:
        panels = [(METRIC_LABELS[m], table[table["metric"] == m]) for m in metrics]
        ncols, panel_in = 2, 2.3
    else:
        rows = table[table["metric"] == metrics[0]]
        panels = [(registry.DATASET_LABELS.get(d, d), rows[rows["dataset"] == d]) for d in datasets]
        ncols, panel_in = 4, 1.55
    values_all = table[table["metric"].isin(metrics)]["delta"].to_numpy() * 100
    norm = _change_norm(values_all)
    n_slots = len(panels) + (0 if pooled else 1)
    fig, axes = _grid_axes(
        n_slots, ncols, panel_in, extra_h=0.3, centre_last=centre_last and pooled
    )
    flat = axes.ravel()
    image = None
    for ax, (title, sub) in zip(flat, panels, strict=False):
        grid = sub.pivot(index="method", columns="init", values="delta").reindex(
            index=methods, columns=inits
        )
        delta = grid.to_numpy() * 100
        texts = _change_texts(delta)
        direction = sub.pivot(index="method", columns="init", values="direction").reindex(
            index=methods, columns=inits
        )
        quiet = direction.to_numpy() == 0
        image = _heatmap(
            ax,
            delta,
            texts,
            quiet,
            cmap=CHANGE_CMAP_LIGHT,
            norm=norm,
            outline=None,
            xticklabels=xt,
            yticklabels=yt,
            fontsize=POOLED_VALUE_PT if pooled else TEXT_PT,
            black_text=True,
        )
        ax.invert_yaxis()
        ax.set_title(title)
    if pooled:
        for ax in flat[len(panels) :]:
            ax.axis("off")
        _shared_colorbar(
            fig,
            image,
            flat[: len(panels)].tolist(),
            "mean change from random initialization (×100)",
        )
    else:
        for ax in flat[len(panels) + 1 :]:
            ax.axis("off")
        _change_key(
            fig,
            flat[len(panels)],
            image,
            f"{KEY_LABELS[metrics[0]]}\nchange from random\ninitialization (×100)",
        )
    return _save(fig, name, output_root)


# I2: seed spread


def _init_legend(fig, inits: Sequence[str]) -> None:
    handles = [
        Line2D([], [], ls="none", marker=INIT_STYLE[i][1], ms=5, mfc=INIT_STYLE[i][0], mec="none")
        for i in inits
    ]
    fig.legend(
        handles,
        [f"{INIT_LABELS.get(i, i)} initialization" for i in inits],
        loc="outside lower center",
        ncol=len(inits),
        frameon=False,
    )


def _dodged(ax, methods, inits, value, error=None) -> None:
    width = 0.7 / max(len(inits), 1)
    for j, init in enumerate(inits):
        colour, marker = INIT_STYLE.get(init, ("#000000", "o"))
        x = np.arange(len(methods)) + (j - (len(inits) - 1) / 2) * width
        y = [value(m, init) for m in methods]
        if error is not None:
            ax.errorbar(
                x,
                y,
                yerr=[error(m, init) for m in methods],
                fmt="none",
                ecolor=colour,
                elinewidth=0.7,
                capsize=0,
                zorder=1,
            )
        ax.scatter(x, y, s=24, marker=marker, color=colour, linewidths=0, zorder=2)
    ax.set_xticks(range(len(methods)), [_variant_label(m) for m in methods])
    layout.style_axes(ax)


def seed_spread(
    pooled: pd.DataFrame,
    metrics: Sequence[str],
    *,
    methods: Sequence[str],
    output_root: Path,
    agreement: pd.DataFrame | None = None,
) -> Path:
    """Seed-to-seed spread of the scores per init, and optionally of the layouts themselves."""
    inits = init_order(pooled["init"].unique())
    panels = list(metrics) + (["agreement"] if agreement is not None else [])
    fig, axes = plt.subplots(
        1, len(panels), figsize=(layout.FULL_WIDTH, 2.1), squeeze=False, layout="constrained"
    )
    lookup = pooled.set_index(["method", "init", "metric"])["mean_sd"]
    for ax, panel in zip(axes[0], panels, strict=True):
        if panel == "agreement":
            mean = agreement.groupby(["method", "init"])["agreement"].mean()
            _dodged(ax, methods, inits, lambda m, i, mean=mean: mean.get((m, i), np.nan))
            ax.set_ylabel("shared 15-NN between seeds")
            ax.set_title("Layout agreement")
        else:
            _dodged(ax, methods, inits, lambda m, i, p=panel: lookup.get((m, i, p), np.nan) * 100)
            ax.set_ylabel("seed SD (×100)")
            ax.set_title(METRIC_LABELS[panel])
        ax.tick_params(axis="x", rotation=30)
    _init_legend(fig, inits)
    return _save(fig, "init-seed-spread", output_root)


# I3: dot plot


def init_dots(
    changes: pd.DataFrame,
    metrics: Sequence[str],
    *,
    methods: Sequence[str],
    datasets: Sequence[str],
    output_root: Path,
    benchmark: pd.DataFrame | None = None,
) -> Path:
    """Seed mean ± SD per variant and init: one row per dataset, one column per metric.

    ``benchmark``: the benchmark's seed summary of every method
    (:func:`~ne_aligned_utils.paper.summaries.hyperparameters.benchmark_metric_summary`). Each
    panel then takes the y range and ticks of that dataset and metric in the benchmark metric
    figures, so the inits can be read against the other methods. Points outside it are listed.
    """
    inits = init_order(changes["init"].unique())
    outside: list[str] = []
    fig, axes = plt.subplots(
        len(datasets),
        len(metrics),
        figsize=(layout.FULL_WIDTH, 1.15 * len(datasets) + 0.5),
        squeeze=False,
        layout="constrained",
    )
    table = changes.set_index(["dataset", "method", "init", "metric"])
    for r, dataset in enumerate(datasets):
        for c, metric in enumerate(metrics):
            ax = axes[r, c]

            def value(m, i, d=dataset, k=metric):
                return table["mean"].get((d, m, i, k), np.nan)

            def error(m, i, d=dataset, k=metric):
                return table["sd"].get((d, m, i, k), np.nan)

            _dodged(ax, methods, inits, value, error)
            if benchmark is not None:
                rows = benchmark[benchmark["dataset"] == dataset]
                rounded = round_ticks(metric_span(rows, metric)) if len(rows) else None
                if rounded is None:
                    raise ValueError(f"no benchmark {metric} scores for {dataset}")
                ticks, (low, high) = rounded
                ax.set_yticks(ticks, tick_labels(ticks, 0.05))
                ax.set_ylim(low, high)
                ours = table.xs((dataset, metric), level=("dataset", "metric"))
                beyond = ours[(ours["mean"] < low) | (ours["mean"] > high)]
                outside += [f"{dataset}/{metric}/{m}/{i}" for m, i in beyond.index]
            if r < len(datasets) - 1:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis="x", rotation=45)
            if r == 0:
                ax.set_title(METRIC_LABELS[metric])
            if c == len(metrics) - 1:
                _dataset_strip(ax, dataset)
    _init_legend(fig, inits)
    if outside:
        print(f"init-dots: seed means outside the benchmark range: {', '.join(outside)}")
    return _save(fig, "init-dots", output_root)


# I4: seed galleries per init


def init_gallery(
    frame: pd.DataFrame,
    *,
    dataset: str,
    method: str,
    inits: Sequence[str],
    output_root: Path,
) -> Path:
    """One column per init (random first), two rows (batch, cell type) per seed."""
    from ne_aligned_utils.analysis import cell_type_palette
    from ne_aligned_utils.paper.rendering import presentation
    from ne_aligned_utils.paper.summaries.hyperparameters import seed_layouts

    loaded = {
        init: seed_layouts(frame, dataset=dataset, method=method, init=init) for init in inits
    }
    seed_sets = {init: tuple(layouts) for init, (layouts, _cache) in loaded.items()}
    if len(set(seed_sets.values())) != 1:
        raise ValueError(f"{dataset}/{method}: inits hold different seeds {seed_sets}")
    sizes = {init: len(next(iter(layouts.values()))) for init, (layouts, _c) in loaded.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"{dataset}/{method}: inits embed different numbers of cells {sizes}")
    seeds = next(iter(seed_sets.values()))
    cached = {
        seed: {init: (loaded[init][0][seed], loaded[init][1]) for init in inits} for seed in seeds
    }
    first_cache = loaded[inits[0]][1]
    palette = cell_type_palette(tuple(str(v) for v in first_cache["label_names"]))
    columns = tuple((init, f"{INIT_LABELS.get(init, init)}\ninitialization") for init in inits)
    fig = presentation.qualitative_composer().seed_panels(cached, palette, columns=columns)
    return layout.save_figure(
        fig, f"init-seeds-{method}-{dataset}", kind="density", root=output_root
    )


__all__ = [
    "CHANGE_CMAP",
    "INIT_LABELS",
    "INIT_STYLE",
    "dataset_grids",
    "init_dots",
    "init_gallery",
    "init_grids",
    "line_panels",
    "pooled_grids",
    "absolute_lines",
    "placing_lines",
    "rank_grids",
    "robustness_strip",
    "seed_spread",
]
