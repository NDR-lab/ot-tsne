"""Per-dataset seed summaries, separate from scientific selection and rendering."""

from collections.abc import Sequence

import pandas as pd


def summarise_seeds(frame: pd.DataFrame, *, metrics: Sequence[str]) -> pd.DataFrame:
    """Return mean and sample standard deviation across seeds for each configuration.

    A single observed seed has unknown spread (NaN), not zero.
    """
    keys = ["dataset", "method_key", "method_label", "family"]
    metrics = list(metrics)
    mean = frame.groupby(keys, dropna=False)[metrics].mean().add_suffix("_mean")
    std = frame.groupby(keys, dropna=False)[metrics].std(ddof=1).add_suffix("_sd")
    count = frame.groupby(keys, dropna=False)["seed"].nunique().rename("n")
    return mean.join(std).join(count).reset_index()


def pool_datasets(summary: pd.DataFrame, *, datasets: Sequence[str]) -> pd.DataFrame:
    """Per method: the mean of its per-dataset means, and their SD across ``datasets``.

    The spread is between datasets, not between seeds, and the caption must say so. A method
    missing from any pooled dataset raises: its mean would cover a different pool.
    """
    block = summary[summary["dataset"].isin(list(datasets))]
    missing = sorted(set(datasets) - set(block["dataset"]))
    if missing:
        raise ValueError(f"no summarised rows for {missing}")
    keys = ["method_key", "method_label", "family"]
    stems = [c.removesuffix("_mean") for c in summary.columns if c.endswith("_mean")]
    aggregation = {"n": ("dataset", "nunique")}
    for stem in stems:
        aggregation[f"{stem}_mean"] = (f"{stem}_mean", "mean")
        aggregation[f"{stem}_sd"] = (f"{stem}_mean", lambda v: v.std(ddof=1))
    out = block.groupby(keys, dropna=False).agg(**aggregation).reset_index()
    incomplete = out[out["n"] != len(set(datasets))]
    if not incomplete.empty:
        raise ValueError(f"methods missing from a pooled dataset: {list(incomplete['method_key'])}")
    out["dataset"] = "pooled"
    return out
