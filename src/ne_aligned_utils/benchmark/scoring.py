"""Score aggregate embedding caches into tidy metric rows and summaries.

Unavailable metrics carry a ``*_reason`` value instead of an unexplained NaN. Method names must
use the canonical registry spelling, with the embedding backend stored separately.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ne_aligned_utils.config import (
    BACKENDS,
    DEFAULT_BACKEND,
    METHOD_SPECS,
    NO_BACKEND,
    has_backend_axis,
    method_spec,
    method_stem,
    resolve_backend,
    split_name,
)
from ne_aligned_utils.evaluation.metrics import (
    DEFAULT_PURITY_K,
    compute_concordance,
    compute_cross_concordance,
    compute_foscttm,
    compute_ilisi,
    compute_kbet,
    compute_kmeans_cluster_scores,
    compute_knn_other_batch_fraction,
    compute_neighbor_purity,
    compute_silhouette_index,
    compute_silhouette_score,
)

DEFAULT_METHODS = (
    "eot-t0",
    "ot",
    "eot-50",
    "joint-pca",
    "joint",
    "cca-2d",
    "cca-50",
    "harmony-2d",
    "harmony-50",
    "scvi-2d",
    "scvi-50",
)

METRIC_COLUMNS = (
    "neighborhood_purity_score",
    "silhouette_index",
    "silhouette_score",
    "ilisi_score",
    "kbet_score",
    "shared_cell_type_batch_mixing_score",
    "kmeans_rand_index",
    "kmeans_adjusted_rand_index",
    "concordance_score",
    "cross_concordance_score",
    "foscttm_score",
)


def load_aggregate_cache(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path).expanduser()
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def aggregate_cache_methods(
    cache: Mapping[str, np.ndarray],
    requested_methods: Iterable[str] | None = None,
) -> tuple[str, ...]:
    if requested_methods is not None:
        return tuple(_base_of(m) for m in requested_methods)
    if "methods" in cache:
        # Reduce canonical names to bases; backend remains a separate field.
        return tuple(_base_of(str(value)) for value in np.asarray(cache["methods"]).tolist())

    available = []
    for method, key in _embedding_keys(cache_backend(cache)).items():
        if key in cache:
            available.append(method)
    return tuple(available) or DEFAULT_METHODS


def method_display_label(
    method: str,
    *,
    rel_reg: float | None = None,
    effective_k: int | None = None,
    backend: str | None = None,
) -> str:
    """Return a display label, optionally including the embedding backend."""
    labels = {
        "rc-tsne": "RC-tSNE",
        "eot-t0": "EOT t=0",
        "ot": "OT-tSNE",
        "eot-50": "EOT50 + tSNE",
        "joint-pca": "Joint PCA",
        "joint": "Joint tSNE",
        "cca-2d": "CCA (2D)",
        "cca-50": "CCA50 + tSNE",
        "cca-2d-uncentered": "CCA (2D, uncentred)",
        "cca-50-uncentered": "CCA50 + tSNE (uncentred)",
        "harmony-2d": "Harmony 2D",
        "harmony-50": "Harmony50 + tSNE",
        "scvi-2d": "scVI 2D",
        "scvi-50": "scVI50 + tSNE",
        "ot-knndiag": "OT-tSNE (kNN diag)",
        "ot-selfotmedian": "OT-tSNE (self-OT diag, own median)",
        "ot-gramdiag": "OT-tSNE (Gram diag)",
        "ot-zerodiag": "OT-tSNE (zero diag)",
        "ot-zerodiag-split": "OT-tSNE (zero diag, split budget)",
        "ref": "Ref t-SNE",
        "ref-our": "Ref t-SNE-our",
        "scanorama-2d": "Scanorama 2D",
        "scanorama-50": "Scanorama50 + tSNE",
        "seurat-2d": "Seurat 2D",
        "seurat-50": "Seurat50 + tSNE",
        "seurat3-2d": "Seurat 3 workflow (2D)",
        "seurat3-50": "Seurat 3 workflow (50D) + tSNE",
        "liger1-2d": "LIGER 1 (2D)",
        "liger1-50": "LIGER 1 (50D) + tSNE",
        "liger-2d": "LIGER (2D)",
        "liger-50": "LIGER (50D) + tSNE",
        "bc-tsne": "BC-tSNE",
        "multimap": "MultiMAP",
    }
    label = labels.get(method, method)
    if backend is not None and has_backend_axis(method) and backend != DEFAULT_BACKEND:
        # Replace the stored t-SNE label with the backend that ran.
        label = re.sub("tSNE", backend.upper(), label)
    details = []
    if rel_reg is not None:
        details.append(f"rel_reg={float(rel_reg):g}")
    if effective_k is not None:
        details.append(f"k={int(effective_k)}")
    return label if not details else f"{label} ({', '.join(details)})"


def _embedding_keys(backend: str = DEFAULT_BACKEND) -> dict[str, str]:
    """Return aggregate-cache array keys for one backend."""
    return {base: f"{method_stem(base, backend)}_embeddings" for base in METHOD_SPECS}


def cache_backend(cache: Mapping[str, np.ndarray]) -> str:
    """Infer the visualizer from cache keys, rejecting mixed backends."""
    declared = None
    if "backend" in cache:
        candidate = str(np.asarray(cache["backend"]).item())
        if candidate in (*BACKENDS, NO_BACKEND):
            declared = candidate
    found: set[str] = set()
    for key in cache:
        if not key.endswith("_embeddings"):
            continue
        name = key[: -len("_embeddings")].replace("_", "-")
        try:
            base, backend = split_name(name)
        except KeyError:
            continue
        if method_spec(base).backend_axis:
            found.add(backend)
    if len(found) > 1:
        raise ValueError(
            f"cache mixes backends {sorted(found)}; one aggregate holds one backend. "
            "This is a merge that replaced rather than extended -- do not score it."
        )
    inferred = found.pop() if found else None
    if declared in BACKENDS and inferred is not None and declared != inferred:
        raise ValueError(
            f"cache declares backend {declared!r}, but embedding keys imply {inferred!r}."
        )
    return declared or inferred or NO_BACKEND


def _base_of(method: str) -> str:
    """A stored or requested method name reduced to its base identity."""
    name = str(method)
    if name in METHOD_SPECS:
        return name
    return split_name(name)[0]


def _require_cache_key(cache: Mapping[str, np.ndarray], key: str, method: str) -> np.ndarray:
    if key not in cache:
        raise KeyError(f"Aggregate cache is missing {key!r}; was method {method!r} aggregated?")
    return np.asarray(cache[key])


def iter_aggregate_embeddings(
    cache: Mapping[str, np.ndarray],
    methods: Iterable[str] | None = None,
    *,
    backend: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield embeddings with their method identity and sweep coordinates."""
    methods = aggregate_cache_methods(cache, methods)
    backend = cache_backend(cache) if backend is None else backend
    seeds = np.asarray(cache["seeds"], dtype=int)
    if "eps_times" in cache:
        raise ValueError(
            "this aggregate stores eps_times, the pre-migration reciprocal of rel_reg. Scoring it "
            "as a rel_reg cache would mislabel every regularisation value. Score it with the "
            "historical code. This conference artifact supports canonical rel_reg caches only."
        )
    rel_reg_values = np.asarray(cache.get("rel_reg", []), dtype=float)
    effective_ks = np.asarray(cache.get("effective_ks", cache.get("percentages", [])), dtype=float)

    simple_methods = {
        "rc-tsne",
        "joint-pca",
        "joint",
        "cca-2d",
        "cca-50",
        "cca-2d-uncentered",
        "cca-50-uncentered",
        "harmony-2d",
        "harmony-50",
        "scvi-2d",
        "scvi-50",
        "ref",
        "ref-our",
        "scanorama-2d",
        "scanorama-50",
        "seurat-2d",
        "seurat-50",
        "seurat3-2d",
        "seurat3-50",
        "liger1-2d",
        "liger1-50",
        "liger-2d",
        "liger-50",
        "bc-tsne",
        "multimap",
    }
    eps_methods = {"eot-t0", "eot-50"}
    embedding_keys = _embedding_keys(backend)

    for method in methods:
        method = _base_of(method)
        if method not in embedding_keys:
            raise ValueError(f"Unsupported aggregate metric method={method!r}.")
        embedding_key = embedding_keys[method]
        # `na` for a method with no backend axis, whatever the cache's own backend is. This is the
        # value that reaches the scored table, so a direct-2D row can never claim a visualizer.
        method_backend = resolve_backend(method, backend)
        embeddings = _require_cache_key(cache, embedding_key, method)

        if method in simple_methods:
            for seed_index, seed in enumerate(seeds.tolist()):
                yield {
                    "method": method,
                    "backend": method_backend,
                    "method_label": method_display_label(method, backend=method_backend),
                    "seed_index": int(seed_index),
                    "seed": int(seed),
                    "rel_reg": np.nan,
                    "effective_k": np.nan,
                    "embedding_key": embedding_key,
                    "embedding": np.asarray(embeddings[seed_index], dtype=float),
                }
            continue

        if method in eps_methods:
            for seed_index, seed in enumerate(seeds.tolist()):
                for rel_reg_index, rel_reg in enumerate(rel_reg_values.tolist()):
                    yield {
                        "method": method,
                        "backend": method_backend,
                        "method_label": method_display_label(
                            method, rel_reg=float(rel_reg), backend=method_backend
                        ),
                        "seed_index": int(seed_index),
                        "seed": int(seed),
                        "rel_reg": float(rel_reg),
                        "effective_k": np.nan,
                        "embedding_key": embedding_key,
                        "embedding": np.asarray(embeddings[seed_index, rel_reg_index], dtype=float),
                    }
            continue

        for seed_index, seed in enumerate(seeds.tolist()):
            for rel_reg_index, rel_reg in enumerate(rel_reg_values.tolist()):
                for k_index, effective_k in enumerate(effective_ks.tolist()):
                    integer_k = (
                        int(effective_k) if float(effective_k).is_integer() else float(effective_k)
                    )
                    yield {
                        "method": method,
                        "backend": method_backend,
                        "method_label": method_display_label(
                            method,
                            rel_reg=float(rel_reg),
                            effective_k=(integer_k if isinstance(integer_k, int) else None),
                            backend=method_backend,
                        ),
                        "seed_index": int(seed_index),
                        "seed": int(seed),
                        "rel_reg": float(rel_reg),
                        "effective_k": integer_k,
                        "embedding_key": embedding_key,
                        "embedding": np.asarray(
                            embeddings[seed_index, rel_reg_index, k_index], dtype=float
                        ),
                    }


def _metric_result(function, *args, **kwargs) -> tuple[float, str]:
    try:
        value = float(function(*args, **kwargs))
    except Exception as exc:  # A report should retain the rest of the run.
        return float("nan"), f"{type(exc).__name__}: {exc}"
    if not np.isfinite(value):
        return value, "metric was not applicable to the available labels or batches"
    return value, ""


def _reference_batches(
    cache: Mapping[str, np.ndarray], seed_index: int
) -> tuple[np.ndarray, np.ndarray] | None:
    key_pairs = (("reference_x0", "reference_x1"), ("clean0", "clean1"))
    for key0, key1 in key_pairs:
        if key0 not in cache or key1 not in cache:
            continue
        x0 = np.asarray(cache[key0], dtype=float)
        x1 = np.asarray(cache[key1], dtype=float)
        if x0.ndim == 3:
            x0 = x0[seed_index]
        if x1.ndim == 3:
            x1 = x1[seed_index]
        return x0, x1
    return None


def score_aggregate_cache(
    cache: Mapping[str, np.ndarray],
    methods: Iterable[str] | None = None,
    *,
    source_cache: str | Path | None = None,
    metric_names: Iterable[str] | None = None,
    backend: str | None = None,
) -> pd.DataFrame:
    """Score aggregate records, optionally restricting the metric columns."""
    if not bool(np.asarray(cache.get("labels_available", False))):
        raise ValueError("Aggregate cache does not contain labels.")

    selected_metrics = set(METRIC_COLUMNS if metric_names is None else metric_names)
    unknown_metrics = selected_metrics.difference(METRIC_COLUMNS)
    if unknown_metrics:
        raise ValueError(f"Unknown metric names: {sorted(unknown_metrics)}")

    dataset = str(np.asarray(cache["dataset"]).item())
    batch_ids = np.asarray(cache["batch_ids"], dtype=int)
    labels0 = np.asarray(cache["labels0"])
    labels1 = np.asarray(cache["labels1"])
    n0 = labels0.shape[1]
    rows: list[dict[str, Any]] = []

    for record in iter_aggregate_embeddings(cache, methods, backend=backend):
        seed_index = int(record["seed_index"])
        labels = np.concatenate([labels0[seed_index], labels1[seed_index]])
        embedding = np.asarray(record.pop("embedding"), dtype=float)
        y0, y1 = embedding[:n0], embedding[n0:]

        row = {
            "dataset": dataset,
            "source_cache": "" if source_cache is None else str(source_cache),
            # Initialization is cache-wide; older caches used random initialization.
            "init": str(cache.get("init", "random")),
            **record,
        }
        metric_calls = {
            "neighborhood_purity_score": (
                compute_neighbor_purity, (embedding, labels), {"k": DEFAULT_PURITY_K}
            ),
            "silhouette_index": (compute_silhouette_index, (embedding, labels), {}),
            "silhouette_score": (compute_silhouette_score, (embedding, labels), {}),
            "ilisi_score": (compute_ilisi, (embedding, batch_ids), {}),
            "kbet_score": (compute_kbet, (embedding, batch_ids, labels), {}),
            "shared_cell_type_batch_mixing_score": (
                compute_knn_other_batch_fraction,
                (embedding, batch_ids, labels),
                {},
            ),
            "kmeans_rand_index": (
                compute_kmeans_cluster_scores,
                (embedding, labels),
                {},
            ),
        }

        for metric_name, (function, args, kwargs) in metric_calls.items():
            if metric_name == "kmeans_rand_index":
                requested_kmeans = selected_metrics.intersection(
                    {"kmeans_rand_index", "kmeans_adjusted_rand_index"}
                )
                if not requested_kmeans:
                    continue
                try:
                    scores = function(*args, **kwargs)
                    values = {
                        "kmeans_rand_index": float(scores["rand_index"]),
                        "kmeans_adjusted_rand_index": float(scores["adjusted_rand_index"]),
                    }
                    for requested in requested_kmeans:
                        row[requested] = values[requested]
                        row[f"{requested}_reason"] = ""
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    for requested in requested_kmeans:
                        row[requested] = np.nan
                        row[f"{requested}_reason"] = reason
                continue
            if metric_name not in selected_metrics:
                continue
            value, reason = _metric_result(function, *args, **kwargs)
            row[metric_name] = value
            row[f"{metric_name}_reason"] = reason

        reference_metrics = selected_metrics.intersection(
            {"concordance_score", "cross_concordance_score", "foscttm_score"}
        )
        references = _reference_batches(cache, seed_index) if reference_metrics else None
        if reference_metrics and references is None:
            reason = "aggregate cache does not include reference_x0/reference_x1"
            for metric_name in reference_metrics:
                row[metric_name] = np.nan
                row[f"{metric_name}_reason"] = reason
        elif references is not None:
            x0, x1 = references
            reference = np.concatenate([x0, x1])
            reference_calls = {
                "concordance_score": (
                    compute_concordance,
                    (reference, embedding),
                    {"k": min(50, reference.shape[0])},
                ),
                "cross_concordance_score": (
                    compute_cross_concordance,
                    (x0, x1, y0, y1),
                    {"k": min(50, x0.shape[0], x1.shape[0])},
                ),
                "foscttm_score": (compute_foscttm, (x0, x1, y0, y1), {}),
            }
            for metric_name, (function, args, kwargs) in reference_calls.items():
                if metric_name not in reference_metrics:
                    continue
                value, reason = _metric_result(function, *args, **kwargs)
                row[metric_name] = value
                row[f"{metric_name}_reason"] = reason

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_metric_rows(rows: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "dataset",
        "method",
        "backend",
        "rel_reg",
        "effective_k",
        "init",
    ]
    group_columns = [column for column in group_columns if column in rows]
    available_metrics = [column for column in METRIC_COLUMNS if column in rows]
    if rows.empty:
        return pd.DataFrame(columns=group_columns)

    summary = (
        rows.groupby(group_columns, dropna=False)[available_metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns.to_flat_index()
    ]
    return summary


def write_metric_reports(
    cache_paths: Iterable[str | Path],
    output_csv: str | Path,
    summary_csv: str | Path,
    methods: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for cache_path in cache_paths:
        path = Path(cache_path).expanduser()
        frames.append(score_aggregate_cache(load_aggregate_cache(path), methods, source_cache=path))
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = summarize_metric_rows(rows)

    output_path = Path(output_csv).expanduser()
    summary_path = Path(summary_csv).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)
    return rows, summary
