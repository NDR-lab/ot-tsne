"""Resolve reported coordinates (seed 0, or every seed) from completed aggregate caches only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ne_aligned_utils import config
from ne_aligned_utils.benchmark.scoring import iter_aggregate_embeddings, load_aggregate_cache
from ne_aligned_utils.paper import registry


def _reported_row(
    frame: pd.DataFrame,
    dataset: str,
    key: str,
    *,
    rel_reg: float,
    backend: str | None,
    seed: int,
    init: str | None = None,
) -> pd.Series:
    """The selected row for one seed. ``init=None`` takes the one init the rows hold."""
    entry = registry.resolve(key)
    rows = frame[(frame["dataset"] == dataset) & (frame["method_key"] == key)]
    wanted = backend or entry.backend_source
    if config.has_backend_axis(entry.method) and wanted in {"tsne", "umap"}:
        rows = rows[rows["backend"] == wanted]
    rows = rows[rows["seed"] == seed]
    if init is not None:
        rows = rows[rows["init"] == init]
    if rows["rel_reg"].notna().any():
        rows = rows[rows["rel_reg"] == float(rel_reg)]
    if rows.empty:
        raise FileNotFoundError(f"no seed-{seed} reported row for {dataset}/{key}/{wanted}")
    inits = sorted(rows["init"].unique())
    if len(inits) > 1:
        raise ValueError(f"{dataset}/{key} has rows for inits {inits}; pass init= to choose one")
    return rows.iloc[0]


def _seed_record(cache, method: str, row: pd.Series, seed: int, label: str) -> np.ndarray:
    records = [
        r for r in iter_aggregate_embeddings(cache, (method,)) if int(r["seed_index"]) == seed
    ]
    if row["rel_reg"] == row["rel_reg"]:
        records = [r for r in records if float(r.get("rel_reg", np.nan)) == float(row["rel_reg"])]
    if len(records) != 1:
        raise ValueError(f"expected one embedding for {label}, found {len(records)}")
    return np.asarray(records[0]["embedding"])


def load_reported_embedding(
    frame: pd.DataFrame,
    dataset: str,
    key: str,
    *,
    rel_reg: float,
    backend: str | None = None,
    init: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Seed-0 coordinates for one method on one dataset, plus the cache they came from.

    ``backend`` overrides the method's registered backend (t-SNE or UMAP). ``init`` picks
    the initialization; ``None`` takes the one the frame holds for this method.
    """
    embeddings, cache = load_seed_embeddings(
        frame, dataset, key, rel_reg=rel_reg, seeds=(0,), backend=backend, init=init
    )
    return embeddings[0], cache


def load_seed_embeddings(
    frame: pd.DataFrame,
    dataset: str,
    key: str,
    *,
    rel_reg: float,
    seeds: tuple[int, ...],
    backend: str | None = None,
    init: str | None = None,
) -> tuple[dict[int, np.ndarray], dict]:
    """Coordinates of each seed (by seed index) for one method, plus the cache they came from.

    Every seed must come from the same cell set; a cache is read once however many seeds it holds.
    """
    method = registry.resolve(key).method
    embeddings: dict[int, np.ndarray] = {}
    loaded: dict[str, dict] = {}
    for seed in seeds:
        row = _reported_row(
            frame, dataset, key, rel_reg=rel_reg, backend=backend, seed=seed, init=init
        )
        path = str(row["source_cache"])
        if path not in loaded:
            loaded[path] = load_aggregate_cache(Path(path))
        label = f"{dataset}/{key}" if seed == 0 else f"{dataset}/{key} seed {seed}"
        embeddings[seed] = _seed_record(loaded[path], method, row, seed, label)
    sizes = {len(e) for e in embeddings.values()}
    if len(sizes) != 1:
        raise ValueError(f"{dataset}/{key}: seeds embed different numbers of cells {sorted(sizes)}")
    return embeddings, next(iter(loaded.values()))
