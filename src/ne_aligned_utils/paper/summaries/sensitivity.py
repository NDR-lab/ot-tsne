"""Seed means along one swept setting, for the sensitivity trajectories.

A trajectory is one run configuration with every setting fixed except the swept one. Seeds are
averaged within a run, and nothing is averaged across settings: rows that differ in any other
recorded setting (k, symmetrisation, backend, device, method settings, ...) become separate
trajectories, never one mean.

The swept setting is either a metric-table column (``rel_reg``, ``effective_k``, ``rel_reg_m``)
or a key of the ``method_settings`` JSON (for example the rc-tsne ``beta``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ne_aligned_utils.evaluation.rank_aggregation import check_seed_groups, seed_group_keys
from ne_aligned_utils.paper.registry import BIO_METRICS, METRIC_SOURCE_COLUMNS, MIXING_METRICS

#: Composite scores: the plain mean of each half's metrics, computed per seed before averaging.
COMPOSITES: dict[str, tuple[str, ...]] = {"bio": BIO_METRICS, "mixing": MIXING_METRICS}
METRICS: tuple[str, ...] = (*METRIC_SOURCE_COLUMNS.values(), *COMPOSITES)

#: Identifies a trajectory within one dataset.
LINE = "line"


def _from_method_settings(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    """Move ``key`` out of the ``method_settings`` JSON into its own column."""
    if "method_settings" not in frame.columns or frame["method_settings"].isna().any():
        raise ValueError(
            f"{key!r} is not a metric-table column, and the rows carry no complete "
            "method_settings to read it from"
        )
    settings = frame["method_settings"].map(json.loads)
    missing = ~settings.map(lambda s: key in s)
    if missing.any():
        methods = sorted(frame.loc[missing, "method"].unique())
        raise ValueError(f"method_settings has no {key!r} for {methods}; filter them out first")
    out = frame.copy()
    out[key] = settings.map(lambda s: s[key])
    # The rest of the settings still tell runs apart.
    out["method_settings"] = settings.map(
        lambda s: json.dumps({k: v for k, v in s.items() if k != key}, sort_keys=True)
    )
    return out


def _natural(values: pd.Series) -> pd.Series:
    """Sort numbers as numbers, everything else as text."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric if numeric.notna().sum() == values.notna().sum() else values.astype(str)


def _apply_filters(frame: pd.DataFrame, fixed: Mapping[str, object]) -> pd.DataFrame:
    for column, value in fixed.items():
        if column not in frame.columns:
            raise ValueError(f"cannot fix {column!r}: the metric rows have no such column")
        values = frame[column]
        try:
            keep = np.isclose(pd.to_numeric(values), float(value))
        except (TypeError, ValueError):
            keep = values.astype(str) == str(value)
        frame = frame[keep]
        if frame.empty:
            raise ValueError(f"no rows left after fixing {column}={value!r}")
    return frame


def _line_labels(points: pd.DataFrame, line_keys: Sequence[str], method_labels) -> pd.Series:
    """Method label plus every line setting that differs between trajectories of that method."""
    lines = points[[*line_keys, LINE]].drop_duplicates(LINE).set_index(LINE)
    labels = {}
    for line, row in lines.iterrows():
        same = lines[lines["method"] == row["method"]]
        parts = []
        for column in line_keys:
            if column == "method" or same[column].astype(str).nunique() < 2:
                continue
            if column == "method_settings":
                mine = json.loads(row[column])
                others = [json.loads(v) for v in same[column].unique()]
                parts += [
                    f"{k}={v}"
                    for k, v in sorted(mine.items())
                    if any(o.get(k) != v for o in others)
                ]
            elif column == "effective_k":
                parts.append(f"k={float(row[column]):g}")
            else:
                value = row[column]
                parts.append(
                    f"{column}={value:g}" if isinstance(value, float) else f"{column}={value}"
                )
        label = method_labels(row["method"])
        labels[line] = f"{label}, {', '.join(parts)}" if parts else label
    return points[LINE].map(labels)


def sweep_points(
    frame: pd.DataFrame,
    *,
    sweep: str,
    datasets: Sequence[str],
    fixed: Mapping[str, object] | None = None,
    methods: Sequence[str] | None = None,
    method_labels=str,
) -> pd.DataFrame:
    """One row per (dataset, trajectory, swept value): seed mean and SD of every metric.

    :param frame: benchmark metric rows (``metrics.csv``), one per task and seed.
    :param sweep: the setting along the trajectory.
    :param fixed: ``column -> value`` filters applied first (for example ``backend="tsne"``).
    :param method_labels: maps a method name to its display name.
    """
    rows = frame.copy()
    if methods is not None:
        rows = rows[rows["method"].isin(list(methods))]
    rows = rows[rows["dataset"].isin(list(datasets))]
    if sweep not in rows.columns:
        rows = _from_method_settings(rows, sweep)
    rows = _apply_filters(rows, dict(fixed or {}))
    # Methods that do not have the swept setting (non-OT methods for rel_reg) have no trajectory.
    rows = rows[rows[sweep].notna() & (rows[sweep].astype(str) != "")]
    if rows.empty:
        raise ValueError(f"no rows record {sweep!r}")
    rows[sweep] = pd.to_numeric(rows[sweep], errors="raise")
    missing = sorted(set(datasets) - set(rows["dataset"]))
    if missing:
        raise ValueError(f"no {sweep} rows for datasets {missing}")

    rows = rows.rename(columns=METRIC_SOURCE_COLUMNS)
    for name, parts in COMPOSITES.items():
        rows[name] = rows[list(parts)].mean(axis=1, skipna=False)

    keys = seed_group_keys(rows)
    if sweep not in keys:
        keys.append(sweep)
    check_seed_groups(rows, keys)
    grouped = rows.groupby(keys, dropna=False)
    mean = grouped[list(METRICS)].mean().add_suffix("_mean")
    sd = grouped[list(METRICS)].std(ddof=1).add_suffix("_sd")
    n = grouped["seed"].nunique().rename("n_seeds")
    points = mean.join(sd).join(n).reset_index()

    line_keys = [k for k in keys if k not in ("dataset", sweep)]
    # Number trajectories in the natural order of their settings (k=5 before k=10).
    ids = points[line_keys].astype(str).agg(" | ".join, axis=1)
    first = points.loc[~ids.duplicated(), line_keys].assign(_id=ids[~ids.duplicated()])
    first = first.sort_values(line_keys, key=_natural, na_position="last")
    points[LINE] = ids.map(dict(zip(first["_id"], range(len(first)))))
    points["line_label"] = _line_labels(points, line_keys, method_labels)
    return points.sort_values(["dataset", LINE, sweep], ignore_index=True)


__all__ = ["COMPOSITES", "LINE", "METRICS", "sweep_points"]
