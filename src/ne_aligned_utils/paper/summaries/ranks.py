"""Score-before-rank summaries over the canonical dataset pool.

Per dataset: average each metric over seeds, optionally min-max rescale it across methods, average
into a bio score (purity, silhouette) and a mixing score (iLISI, kBET), and average those two into
an overall score. Each score is ranked within the dataset (1 = best, ties averaged).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ne_aligned_utils.paper.registry import BIO_METRICS, MIXING_METRICS

METRICS = (*BIO_METRICS, *MIXING_METRICS)
NORMALIZATIONS = ("raw", "minmax")

#: Rank block -> score column it ranks.
BLOCKS = {"all": "overall", "bio": "bio", "mixing": "mixing"}


def _scores(frame: pd.DataFrame, datasets: Sequence[str], normalization: str) -> pd.DataFrame:
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
    means = (
        frame[frame["dataset"].isin(datasets)]
        .groupby(["dataset", "method_key"], as_index=False)[list(METRICS)]
        .mean()
    )
    if normalization == "minmax":
        for metric in METRICS:
            means[metric] = means.groupby("dataset")[metric].transform(
                lambda values: (
                    (values - values.min()) / (values.max() - values.min())
                    if values.max() > values.min()
                    else 0.0
                )
            )
    means["bio_score"] = means[list(BIO_METRICS)].mean(axis=1)
    means["mixing_score"] = means[list(MIXING_METRICS)].mean(axis=1)
    means["overall_score"] = means[["bio_score", "mixing_score"]].mean(axis=1)
    for score in ("bio", "mixing", "overall"):
        means[f"{score}_rank"] = means.groupby("dataset")[f"{score}_score"].rank(
            ascending=False, method="average"
        )
    return means


def aggregate_ranks(
    frame: pd.DataFrame, *, datasets: Sequence[str], normalization: str, rel_reg: float
) -> pd.DataFrame:
    """Mean within-dataset rank per method, one row per (method, block)."""
    scored = _scores(frame, datasets, normalization)
    n_competitors = int(scored.groupby("dataset")["method_key"].nunique().iloc[0])
    pieces = []
    for block, score in BLOCKS.items():
        piece = (
            scored.groupby("method_key", as_index=False)[f"{score}_rank"]
            .mean()
            .rename(columns={f"{score}_rank": "aggregate_rank"})
        )
        piece["block"] = block
        piece["rel_reg"] = rel_reg
        piece["n_datasets"] = len(datasets)
        piece["n_competitors"] = n_competitors
        piece["normalization"] = normalization
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def top_k(
    frame: pd.DataFrame,
    *,
    datasets: Sequence[str],
    normalization: str,
    k: int,
) -> pd.DataFrame:
    """Per dataset and method: one placing each for a bio rank and a mixing rank within ``k``."""
    if k < 1:
        raise ValueError("k must be at least one")
    scored = _scores(frame, datasets, normalization)
    out = scored[["dataset", "method_key"]].copy()
    out["n_top_k_bio"] = (scored["bio_rank"] <= k).astype(int)
    out["n_top_k_mixing"] = (scored["mixing_rank"] <= k).astype(int)
    out["n_top_k"] = out["n_top_k_bio"] + out["n_top_k_mixing"]
    out["n_cells"] = 2
    out["top_k"] = k
    return out


__all__ = ["BLOCKS", "NORMALIZATIONS", "aggregate_ranks", "top_k"]
