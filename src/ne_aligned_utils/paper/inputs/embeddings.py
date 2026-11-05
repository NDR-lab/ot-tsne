"""Resolve reported seed-0 coordinates from completed aggregate caches only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ne_aligned_utils import config
from ne_aligned_utils.benchmark.scoring import iter_aggregate_embeddings, load_aggregate_cache
from ne_aligned_utils.paper import registry


def load_reported_embedding(
    frame: pd.DataFrame,
    dataset: str,
    key: str,
    *,
    rel_reg: float,
    backend: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Seed-0 coordinates for one method on one dataset, plus the cache they came from.

    ``backend`` overrides the method's registered backend (t-SNE or UMAP).
    """
    entry = registry.resolve(key)
    method, backend_source = entry.method, entry.backend_source
    has_backend_axis = config.has_backend_axis(method)
    rel_reg = float(rel_reg)
    rows = frame[(frame["dataset"] == dataset) & (frame["method_key"] == key)]
    wanted = backend or backend_source
    if has_backend_axis and wanted in {"tsne", "umap"}:
        rows = rows[rows["backend"] == wanted]
    rows = rows[(rows["seed"] == 0) & (rows["init"] == "random")]
    if rows["rel_reg"].notna().any():
        rows = rows[rows["rel_reg"] == rel_reg]
    if rows.empty:
        raise FileNotFoundError(f"no seed-0 reported row for {dataset}/{key}/{wanted}")
    cache = load_aggregate_cache(Path(rows.iloc[0]["source_cache"]))
    records = [r for r in iter_aggregate_embeddings(cache, (method,)) if int(r["seed_index"]) == 0]
    if rows.iloc[0]["rel_reg"] == rows.iloc[0]["rel_reg"]:
        records = [
            r for r in records if float(r.get("rel_reg", np.nan)) == float(rows.iloc[0]["rel_reg"])
        ]
    if len(records) != 1:
        raise ValueError(f"expected one embedding for {dataset}/{key}, found {len(records)}")
    return np.asarray(records[0]["embedding"]), cache
