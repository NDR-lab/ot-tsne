"""Rank tables for the paper figures.

The numbers come from :mod:`ne_aligned_utils.evaluation.score_aggregation`, which scores before
it ranks. Per dataset it averages each metric over seeds, optionally min-max rescales it across
methods, averages into a bio score (purity, silhouette) and a mixing score (iLISI, kBET), and
averages those two into an overall score. Each score is ranked within the dataset (1 = best, ties
averaged), and the ranks are averaged over datasets. This module only adapts the paper frame to
that input shape and reshapes the result for the figures.

Methods are identified by ``method_key``, never by the raw ``method`` name. Both MultiMAP arms are
called ``multimap`` in the metric tables; ``method_key`` tells them apart (``multimap`` is native,
``multimap-k15`` the matched-k arm). Ranks always use native MultiMAP, and the matched-k arm is
refused rather than silently ranked in its place.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ne_aligned_utils.evaluation.score_aggregation import (
    MATCHED_K_MULTIMAP,
    aggregate_scores,
    score_within,
)
from ne_aligned_utils.paper.registry import BIO_METRICS, METRIC_SOURCE_COLUMNS, MIXING_METRICS

METRICS = (*BIO_METRICS, *MIXING_METRICS)
NORMALIZATIONS = ("raw", "minmax")
NATIVE_MULTIMAP = "multimap"

#: Rank block -> score column it ranks.
BLOCKS = {"all": "overall", "bio": "bio", "mixing": "mixing"}

_SOURCE_COLUMNS = {short: source for source, short in METRIC_SOURCE_COLUMNS.items()}


def _require_native_multimap(frame: pd.DataFrame, datasets: Sequence[str]) -> None:
    """Every ranked dataset carries native MultiMAP, and none carries the matched-k arm."""
    keys = frame.groupby("dataset")["method_key"].agg(set)
    matched = sorted(d for d in datasets if MATCHED_K_MULTIMAP in keys.get(d, set()))
    if matched:
        raise ValueError(
            f"ranks use native MultiMAP only, but {matched} carry the matched-k arm "
            f"{MATCHED_K_MULTIMAP!r}. Point the build at results with native MultiMAP (for an "
            "October-style tree, pass the supplementary root that holds multimap_native)."
        )
    missing = sorted(d for d in datasets if NATIVE_MULTIMAP not in keys.get(d, set()))
    if missing:
        raise ValueError(f"native MultiMAP is missing from {missing}; ranks need it everywhere")


def _scored(
    frame: pd.DataFrame,
    *,
    datasets: Sequence[str],
    normalization: str,
    rel_reg: float,
    effective_k: float,
) -> pd.DataFrame:
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
    data = frame[frame["dataset"].isin(datasets)]
    _require_native_multimap(data, datasets)
    # score_within keys on "method"; give it method_key so the MultiMAP arms stay distinct.
    rows = data.drop(columns=["method"]).rename(columns={**_SOURCE_COLUMNS, "method_key": "method"})
    scored = score_within(
        rows,
        metrics=tuple(_SOURCE_COLUMNS[metric] for metric in METRICS),
        datasets=list(datasets),
        rel_reg_values=(float(rel_reg),),
        effective_k=float(effective_k),
        normalization=normalization,
    )
    # score_within keeps only methods present on every dataset; the figures must show them all.
    dropped = sorted(set(rows["method"]) - set(scored["method"]))
    if dropped:
        raise ValueError(f"methods missing from some dataset, so they cannot be ranked: {dropped}")
    return scored.rename(columns={"method": "method_key"})


def aggregate_ranks(
    frame: pd.DataFrame,
    *,
    datasets: Sequence[str],
    normalization: str,
    rel_reg: float,
    effective_k: float,
) -> pd.DataFrame:
    """Mean within-dataset rank per method, one row per (method, block)."""
    scored = _scored(
        frame,
        datasets=datasets,
        normalization=normalization,
        rel_reg=rel_reg,
        effective_k=effective_k,
    )
    agg = aggregate_scores(scored.rename(columns={"method_key": "method"})).sort_values("method")
    pieces = []
    for block, score in BLOCKS.items():
        piece = agg[["method", f"{score}_rank"]].rename(
            columns={"method": "method_key", f"{score}_rank": "aggregate_rank"}
        )
        piece["block"] = block
        piece["rel_reg"] = rel_reg
        piece["n_datasets"] = len(datasets)
        piece["n_competitors"] = int(agg["n_competitors"].iloc[0])
        piece["normalization"] = normalization
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def top_k(
    frame: pd.DataFrame,
    *,
    datasets: Sequence[str],
    normalization: str,
    k: int,
    rel_reg: float,
    effective_k: float,
) -> pd.DataFrame:
    """Per dataset and method: one placing each for a bio rank and a mixing rank within ``k``."""
    if k < 1:
        raise ValueError("k must be at least one")
    scored = _scored(
        frame,
        datasets=datasets,
        normalization=normalization,
        rel_reg=rel_reg,
        effective_k=effective_k,
    ).sort_values(["dataset", "method_key"], ignore_index=True)
    out = scored[["dataset", "method_key"]].copy()
    out["n_top_k_bio"] = (scored["bio_rank"] <= k).astype(int)
    out["n_top_k_mixing"] = (scored["mixing_rank"] <= k).astype(int)
    out["n_top_k"] = out["n_top_k_bio"] + out["n_top_k_mixing"]
    out["n_cells"] = 2
    out["top_k"] = k
    return out


__all__ = ["BLOCKS", "NATIVE_MULTIMAP", "NORMALIZATIONS", "aggregate_ranks", "top_k"]
