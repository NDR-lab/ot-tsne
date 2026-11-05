"""Registry-independent point, error-bar and metric-axis drawing primitives."""

import numpy as np
import pandas as pd


def frame_embedding_panel(ax, embedding: np.ndarray, *, trim: float = 0.002) -> None:
    """Frame the bulk of coordinates at 1:1 aspect without discarding points."""
    lo = np.quantile(embedding, trim, axis=0)
    hi = np.quantile(embedding, 1.0 - trim, axis=0)
    centre = (lo + hi) / 2
    half = float(np.max(hi - lo)) / 2 * 1.06
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal", adjustable="box")


def clipped_error(mean: float, sd: float | None, *, metric_range=(0.0, 1.0)) -> tuple[float, float]:
    """Return asymmetric lower/upper error lengths clipped to ``metric_range``.

    Clip each side independently so a bound does not shorten the opposite side.
    """
    low, high = metric_range
    spread = 0.0 if sd is None or not np.isfinite(sd) else float(sd)
    return (float(mean) - max(low, mean - spread), min(high, mean + spread) - float(mean))


def metric_span(block: pd.DataFrame, stem: str, *, metric_range=(0.0, 1.0)) -> pd.Series:
    """Return means and clipped error endpoints used to determine metric-axis limits."""
    mean = block[f"{stem}_mean"]
    sd = block[f"{stem}_sd"].fillna(0.0) if f"{stem}_sd" in block else 0.0
    low, high = metric_range
    return pd.concat([mean, mean.sub(sd).clip(lower=low), mean.add(sd).clip(upper=high)])


def draw_metric_point(
    ax, row, *, x, y, scatter, error_color, errorbars=True, metric_range=(0.0, 1.0)
):
    """Draw one summarised point and its error bars, clipped to the metric range."""
    if errorbars:
        xerr = clipped_error(row[x], row.get(x.replace("_mean", "_sd")), metric_range=metric_range)
        yerr = clipped_error(row[y], row.get(y.replace("_mean", "_sd")), metric_range=metric_range)
        ax.errorbar(
            row[x],
            row[y],
            xerr=[[xerr[0]], [xerr[1]]],
            yerr=[[yerr[0]], [yerr[1]]],
            fmt="none",
            ecolor=error_color,
            elinewidth=0.55,
            alpha=0.55,
            zorder=-1,
            clip_on=False,
        )
    ax.scatter(row[x], row[y], clip_on=False, **scatter)


def round_three_ticks(
    ax,
    xvalues,
    yvalues,
    *,
    step: float = 0.05,
    step_x: float | None = None,
    step_y: float | None = None,
    n_x: int = 3,
    n_y: int = 3,
) -> None:
    """Place round ticks enclosing the data using per-axis steps and counts.

    Coarser steps reduce decimal-label width for narrow panels. Midpoint rounding can
    leave the middle tick slightly off center.
    """
    for values, set_ticks, set_lim, count, axis_step in (
        (xvalues, ax.set_xticks, ax.set_xlim, n_x, step if step_x is None else step_x),
        (yvalues, ax.set_yticks, ax.set_ylim, n_y, step if step_y is None else step_y),
    ):
        finite = np.asarray([v for v in np.ravel(values) if np.isfinite(v)], dtype=float)
        if finite.size == 0:
            continue
        low = np.floor(float(finite.min()) / axis_step) * axis_step
        high = np.ceil(float(finite.max()) / axis_step) * axis_step
        if high - low < axis_step:
            high = low + axis_step
        wanted = {round(low, 10), round(high, 10)}
        if count >= 3:
            wanted.add(round(round(((low + high) / 2.0) / axis_step) * axis_step, 10))
        ticks = sorted(wanted)
        set_ticks(ticks)
        pad = 0.05 * (high - low)
        set_lim(low - pad, high + pad)

    def _fmt(values, axis_step):
        decimals = 1 if axis_step >= 0.1 - 1e-9 else 2
        return [f"{t:.{decimals}f}" for t in values]

    ax.set_xticklabels(_fmt(ax.get_xticks(), step if step_x is None else step_x))
    ax.set_yticklabels(_fmt(ax.get_yticks(), step if step_y is None else step_y))
