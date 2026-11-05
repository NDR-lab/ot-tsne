"""Aggregate normalized metric values before ranking methods.

This is the score-before-rank alternative to :mod:`ne_aligned_utils.evaluation.rank_aggregation`. It shares
that module's metric directions, families, method pools, and regularization grid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ne_aligned_utils.benchmark.scoring import method_display_label
from ne_aligned_utils.evaluation.rank_aggregation import (
    METHOD_FAMILY,
    METRIC_DIRECTION,
    METRIC_FAMILY,
    RANK_METRICS,
    _expand_non_rel_reg_methods,
    resolve_method_pool,
)

# Vocabulary

#: Min-max is the primary variant; raw is a sensitivity check for metrics already in [0, 1].
NORMALIZATIONS: tuple[str, ...] = ("minmax", "raw")

#: Composite scores, constructed before ranking.
SCORES: tuple[str, ...] = ("bio", "mixing", "overall")

#: Reported regularization anchors: 0.1 and the library default 1.
REPORTED_REL_REG: tuple[float, ...] = (1.0, 1 / 10)

#: Suffix for a metric's normalized column. Spelled out rather than appended to ``_score`` so a
#: normalized column can never be mistaken for the raw metric it came from.
NORM_SUFFIX = "__norm"

#: Map shared metric families to this module's bio/mixing column names.
#: Figure-facing frames retain the shared batch spelling.
FAMILY_TO_SCORE: dict[str, str] = {"bio": "bio", "batch": "mixing"}


def _check_normalization(normalization: str) -> str:
    if normalization not in NORMALIZATIONS:
        raise ValueError(
            f"normalization must be one of {NORMALIZATIONS}, got {normalization!r}"
        )
    return normalization


def metric_sides(metrics: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Partition metrics into bio and mixing families, requiring at least one in each."""
    sides: dict[str, list[str]] = {"bio": [], "mixing": []}
    for metric in metrics:
        family = METRIC_FAMILY.get(metric)
        if family is None:
            raise ValueError(
                f"{metric!r} has no entry in rank_aggregation.METRIC_FAMILY, so it cannot be "
                "attributed to the bio or the mixing side of the trade-off."
            )
        sides[FAMILY_TO_SCORE[family]].append(metric)
    empty = [side for side, members in sides.items() if not members]
    if empty:
        raise ValueError(
            f"no {' or '.join(empty)} metric among {list(metrics)}; both sides must be populated "
            "or the composite is not the trade-off it is named after."
        )
    return {side: tuple(members) for side, members in sides.items()}


# Steps 1-6


def _oriented(values: pd.Series, metric: str) -> pd.Series:
    """Orient scores so larger is better while preserving ``[0, 1]``.

    Reflect lower-is-better values as ``1 - x``; negation would distort raw composites.
    """
    if METRIC_DIRECTION.get(metric, +1) > 0:
        return values
    outside = values.dropna()
    if not outside.empty and (outside.min() < 0.0 or outside.max() > 1.0):
        raise ValueError(
            f"{metric!r} is a lower-is-better metric outside [0, 1] "
            f"([{outside.min():.4g}, {outside.max():.4g}]); reflecting it as 1 - x would not "
            "orient it. Rescale it to [0, 1] first, or give this module an explicit reflection."
        )
    return 1.0 - values


def _minmax(values: pd.Series) -> pd.Series:
    """Min-max normalize one ranking cell, warning and returning zeros for a constant cell.

    A constant column contributes equally to every method and cannot affect within-cell ranks.
    """
    low, high = float(values.min()), float(values.max())
    span = high - low
    if not np.isfinite(span):
        raise ValueError(f"non-finite values in a normalisation cell: [{low}, {high}]")
    if span == 0.0:
        return pd.Series(0.0, index=values.index)
    return (values - low) / span


def score_within(
    rows: pd.DataFrame,
    *,
    metrics: Sequence[str] = RANK_METRICS,
    methods: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    rel_reg_values: Sequence[float] = REPORTED_REL_REG,
    effective_k: float | None = 15.0,
    normalization: str = "minmax",
    bio_weight: float = 0.5,
) -> pd.DataFrame:
    """Average seeds, normalize metrics, and rank bio, mixing, and overall scores per dataset.

    Return metric means and normalized values with composite scores, ranks, and pool
    metadata. Exact ties average occupied ranks. ``bio_weight`` controls the overall
    composite; mixing receives the remainder (equal weight by default).
    """
    _check_normalization(normalization)
    if not 0.0 <= float(bio_weight) <= 1.0:
        raise ValueError(f"bio_weight must lie in [0, 1], got {bio_weight!r}")

    frame = rows.copy()
    if datasets is not None:
        frame = frame[frame["dataset"].isin(list(datasets))]
    if methods is not None:
        frame = frame[frame["method"].isin(list(methods))]
    if effective_k is not None and "effective_k" in frame.columns:
        # keep methods that carry no k at all, plus those matching the requested k
        frame = frame[frame["effective_k"].isna() | (frame["effective_k"] == effective_k)]
    if frame.empty:
        raise ValueError("no rows left after filtering; check datasets/methods/effective_k")

    metrics = [m for m in metrics if m in frame.columns]
    if not metrics:
        raise ValueError("none of the requested metrics are present in the input frame")
    sides = metric_sides(metrics)

    # step 1: mean over seeds. No spread is carried: unlike the rank-first pipeline this one has
    # no tie tolerance to derive from it, and a spread in a metric's units cannot be compared
    # against a composite of normalized metrics anyway.
    keys = ["dataset", "method", "rel_reg"]
    grouped = frame.groupby(keys, dropna=False)[list(metrics)].mean().reset_index()

    # A method with no rel_reg knob holds one score valid in every regularization's table, so it
    # is broadcast across the grid exactly as the rank-first pipeline broadcasts it. Its own value
    # is then fixed across the tables while its rank still moves, because its competitors' are not.
    grouped = _expand_non_rel_reg_methods(grouped, rel_reg_values)
    wanted = [float(e) for e in rel_reg_values]
    grouped = grouped[grouped["rel_reg"].isin(wanted)]
    if grouped.empty:
        raise ValueError(
            f"no rows at rel_reg {wanted}; the frame holds "
            f"{sorted(rows['rel_reg'].dropna().unique())}."
        )

    present = sorted(grouped["dataset"].unique()) if datasets is None else list(datasets)
    pool = resolve_method_pool(grouped, present)
    if not pool:
        raise ValueError(f"no method is present on all of {present}")
    grouped = grouped[grouped["method"].isin(pool)]

    _check_complete_cells(grouped, pool, metrics)

    # steps 2-3: orient, then normalise inside each (dataset, rel_reg, metric).
    cells = grouped.groupby(["dataset", "rel_reg"], sort=False)
    pieces = []
    for _cell, block in cells:
        scored = block.copy()
        for metric in metrics:
            oriented = _oriented(scored[metric], metric)
            scored[f"{metric}{NORM_SUFFIX}"] = (
                _minmax(oriented) if normalization == "minmax" else oriented
            )
        # step 4: the composites, from values and never from ranks.
        scored["bio_score"] = scored[
            [f"{m}{NORM_SUFFIX}" for m in sides["bio"]]
        ].mean(axis=1)
        scored["mixing_score"] = scored[
            [f"{m}{NORM_SUFFIX}" for m in sides["mixing"]]
        ].mean(axis=1)
        scored["overall_score"] = (
            float(bio_weight) * scored["bio_score"]
            + (1.0 - float(bio_weight)) * scored["mixing_score"]
        )
        # step 5: rank inside this one (dataset, rel_reg), separately per score. overall_rank
        # ranks overall_score -- it is emphatically not the mean of bio_rank and mixing_rank.
        for side in SCORES:
            scored[f"{side}_rank"] = scored[f"{side}_score"].rank(
                ascending=False, method="average"
            )
        pieces.append(scored)

    out = pd.concat(pieces, ignore_index=True)
    out["n_competitors"] = len(pool)
    out["normalization"] = normalization
    out["bio_weight"] = float(bio_weight)
    out["method_label"] = out["method"].map(method_display_label)
    out["family"] = out["method"].map(METHOD_FAMILY).fillna("other")
    return out.sort_values(["rel_reg", "dataset", "overall_rank"]).reset_index(drop=True)


def _check_complete_cells(
    grouped: pd.DataFrame, pool: Sequence[str], metrics: Sequence[str]
) -> None:
    """Require every pooled method and metric in each dataset/regularization cell.

    Missing methods change normalization ranges; missing metrics change composite weights.
    """
    counts = grouped.groupby(["dataset", "rel_reg"], dropna=False).size()
    short = counts[counts != len(pool)]
    if not short.empty:
        raise ValueError(
            f"the competitor pool holds {len(pool)} methods but these cells hold a different "
            f"number, so min-max would be computed against a different field in each:\n{short}"
        )
    missing = grouped[grouped[list(metrics)].isna().any(axis=1)]
    if not missing.empty:
        offenders = missing[["dataset", "method", "rel_reg"]].head(10)
        raise ValueError(
            f"{len(missing)} row(s) have a missing metric, so their composite would average over "
            f"fewer terms than their competitors':\n{offenders}"
        )


def aggregate_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """Average dataset-level scores and ranks per method and regularization.

    Sort by mean overall rank and retain coverage and normalization metadata. Mean ranks
    need not be integers; normalized scores are comparable only within the same pool.
    """
    pools = sorted(scored["n_competitors"].unique())
    if len(pools) != 1:
        raise ValueError(
            f"the rows were scored against different competitor pools ({pools}), so their "
            "placings are not on one scale and their means are not comparable."
        )
    variants = sorted(scored["normalization"].unique())
    if len(variants) != 1:
        raise ValueError(
            f"the rows mix normalisation variants ({variants}); aggregate each on its own, or "
            "the mean is over two different quantities."
        )
    aggregation = {f"{s}_rank": (f"{s}_rank", "mean") for s in SCORES}
    aggregation |= {f"{s}_score": (f"{s}_score", "mean") for s in SCORES}
    aggregation |= {
        "n_datasets": ("dataset", "nunique"),
        "n_competitors": ("n_competitors", "max"),
    }
    agg = scored.groupby(["method", "rel_reg"], as_index=False).agg(**aggregation)
    agg["normalization"] = variants[0]
    agg["method_label"] = agg["method"].map(method_display_label)
    agg["family"] = agg["method"].map(METHOD_FAMILY).fillna("other")
    return agg.sort_values(["rel_reg", "overall_rank"]).reset_index(drop=True)


def score_matrix(agg: pd.DataFrame, side: str = "overall") -> pd.DataFrame:
    """Method x ``rel_reg`` matrix of one side's mean rank, rows ordered by best rank achieved."""
    if side not in SCORES:
        raise ValueError(f"side must be one of {SCORES}, got {side!r}")
    wide = agg.pivot_table(index="method", columns="rel_reg", values=f"{side}_rank")
    return wide.loc[wide.min(axis=1).sort_values().index]


# Comparing two ranking procedures
#
# Report top-set agreement alongside correlation: high correlation can hide changed leaders.


def leaders(ranks: pd.Series, k: int = 3) -> tuple[str, ...]:
    """Return top-k methods by name, including all methods tied at the cutoff."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    cutoff = float(ranks.nsmallest(k).max())
    return tuple(sorted(ranks.index[ranks <= cutoff]))


def compare_rankings(
    left: pd.Series, right: pd.Series, *, k: int = 3
) -> dict[str, object]:
    """Compare aligned method rankings with Spearman correlation and top-set agreement.

    Both inputs use lower-is-better ranks and must contain exactly the same methods.
    """
    only_left = sorted(set(left.index) - set(right.index))
    only_right = sorted(set(right.index) - set(left.index))
    if only_left or only_right:
        raise ValueError(
            "the two rankings do not cover the same methods; "
            f"only on the left: {only_left}, only on the right: {only_right}"
        )
    if left.index.has_duplicates or right.index.has_duplicates:
        raise ValueError("a method appears twice in one of the rankings")
    ordered = right.reindex(left.index)
    coefficient = float(spearmanr(left.to_numpy(float), ordered.to_numpy(float)).statistic)
    top_left, top_right = leaders(left, 1), leaders(ordered, 1)
    set_left, set_right = leaders(left, k), leaders(ordered, k)
    return {
        "spearman_r": coefficient,
        "n_methods": int(len(left)),
        "top1_left": "|".join(top_left),
        "top1_right": "|".join(top_right),
        "top1_changed": top_left != top_right,
        f"top{k}_left": "|".join(set_left),
        f"top{k}_right": "|".join(set_right),
        f"n_shared_top{k}": int(len(set(set_left) & set(set_right))),
        f"top{k}_changed": set(set_left) != set(set_right),
    }


def comparison_table(
    rankings: Mapping[tuple[float, str, str], pd.Series],
    pairs: Sequence[tuple[str, str]],
    *,
    k: int = 3,
) -> pd.DataFrame:
    """Compare pipeline pairs from method-indexed rank series.

    ``rankings`` keys are ``(rel_reg, rank_type, pipeline)``; ``pairs`` sets comparison order.
    """
    axes = sorted(
        {(rel_reg, rank_type) for rel_reg, rank_type, _ in rankings},
        key=lambda key: (key[0], SCORES.index(key[1]) if key[1] in SCORES else 99),
    )
    rows = []
    for rel_reg, rank_type in axes:
        for first, second in pairs:
            keys = ((rel_reg, rank_type, first), (rel_reg, rank_type, second))
            if not all(key in rankings for key in keys):
                continue
            row = {
                "rel_reg": rel_reg,
                "rank_type": rank_type,
                "pipeline_1": first,
                "pipeline_2": second,
            }
            row |= compare_rankings(rankings[keys[0]], rankings[keys[1]], k=k)
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.reset_index(drop=True)


__all__ = [
    "FAMILY_TO_SCORE",
    "NORMALIZATIONS",
    "NORM_SUFFIX",
    "REPORTED_REL_REG",
    "SCORES",
    "aggregate_scores",
    "compare_rankings",
    "comparison_table",
    "leaders",
    "metric_sides",
    "score_matrix",
    "score_within",
]
