from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ne_aligned_utils.metrics import (
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
    "ot-tsne",
    "eot-50-tsne",
    "joint-pca",
    "joint-tsne",
    "seurat",
    "harmony-2d",
    "harmony-50-tsne",
    "scvi-2d",
    "scvi-50-tsne",
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
        return tuple(requested_methods)
    if "methods" in cache:
        return tuple(str(value) for value in np.asarray(cache["methods"]).tolist())

    available = []
    for method, key in _embedding_keys().items():
        if key in cache:
            available.append(method)
    return tuple(available) or DEFAULT_METHODS


def method_display_label(
    method: str,
    *,
    eps_times: float | None = None,
    effective_k: int | None = None,
) -> str:
    labels = {
        "eot-t0": "EOT t=0",
        "ot-tsne": "OT-tSNE",
        "eot-50-tsne": "EOT50 + tSNE",
        "joint-pca": "Joint PCA",
        "joint-tsne": "Joint tSNE",
        "seurat": "Seurat",
        "harmony-2d": "Harmony 2D",
        "harmony-50-tsne": "Harmony50 + tSNE",
        "scvi-2d": "scVI 2D",
        "scvi-50-tsne": "scVI50 + tSNE",
    }
    label = labels.get(method, method)
    details = []
    if eps_times is not None:
        details.append(f"eps={float(eps_times):g}")
    if effective_k is not None:
        details.append(f"k={int(effective_k)}")
    return label if not details else f"{label} ({', '.join(details)})"


def _embedding_keys() -> dict[str, str]:
    return {
        "joint-pca": "joint_pca_embeddings",
        "joint-tsne": "joint_tsne_embeddings",
        "seurat": "seurat_embeddings",
        "harmony-2d": "harmony_2d_embeddings",
        "harmony-50-tsne": "harmony_50_tsne_embeddings",
        "scvi-2d": "scvi_2d_embeddings",
        "scvi-50-tsne": "scvi_50_tsne_embeddings",
        "eot-t0": "eot_t0_embeddings",
        "eot-50-tsne": "eot_50_tsne_embeddings",
        "ot-tsne": "ot_tsne_embeddings",
    }


def _require_cache_key(
    cache: Mapping[str, np.ndarray], key: str, method: str
) -> np.ndarray:
    if key not in cache:
        raise KeyError(
            f"Aggregate cache is missing {key!r}; was method {method!r} aggregated?"
        )
    return np.asarray(cache[key])


def iter_aggregate_embeddings(
    cache: Mapping[str, np.ndarray],
    methods: Iterable[str] | None = None,
) -> Iterator[dict[str, Any]]:
    methods = aggregate_cache_methods(cache, methods)
    seeds = np.asarray(cache["seeds"], dtype=int)
    eps_values = np.asarray(cache.get("eps_times", []), dtype=float)
    effective_ks = np.asarray(
        cache.get("effective_ks", cache.get("percentages", [])), dtype=float
    )

    simple_methods = {
        "joint-pca",
        "joint-tsne",
        "seurat",
        "harmony-2d",
        "harmony-50-tsne",
        "scvi-2d",
        "scvi-50-tsne",
    }
    eps_methods = {"eot-t0", "eot-50-tsne"}
    embedding_keys = _embedding_keys()

    for method in methods:
        if method not in embedding_keys:
            raise ValueError(f"Unsupported aggregate metric method={method!r}.")
        embedding_key = embedding_keys[method]
        embeddings = _require_cache_key(cache, embedding_key, method)

        if method in simple_methods:
            for seed_index, seed in enumerate(seeds.tolist()):
                yield {
                    "method": method,
                    "method_label": method_display_label(method),
                    "seed_index": int(seed_index),
                    "seed": int(seed),
                    "eps_times": np.nan,
                    "effective_k": np.nan,
                    "embedding_key": embedding_key,
                    "embedding": np.asarray(embeddings[seed_index], dtype=float),
                }
            continue

        if method in eps_methods:
            for seed_index, seed in enumerate(seeds.tolist()):
                for eps_index, eps_times in enumerate(eps_values.tolist()):
                    yield {
                        "method": method,
                        "method_label": method_display_label(
                            method, eps_times=float(eps_times)
                        ),
                        "seed_index": int(seed_index),
                        "seed": int(seed),
                        "eps_times": float(eps_times),
                        "effective_k": np.nan,
                        "embedding_key": embedding_key,
                        "embedding": np.asarray(
                            embeddings[seed_index, eps_index], dtype=float
                        ),
                    }
            continue

        for seed_index, seed in enumerate(seeds.tolist()):
            for eps_index, eps_times in enumerate(eps_values.tolist()):
                for k_index, effective_k in enumerate(effective_ks.tolist()):
                    integer_k = (
                        int(effective_k)
                        if float(effective_k).is_integer()
                        else float(effective_k)
                    )
                    yield {
                        "method": method,
                        "method_label": method_display_label(
                            method,
                            eps_times=float(eps_times),
                            effective_k=(
                                integer_k if isinstance(integer_k, int) else None
                            ),
                        ),
                        "seed_index": int(seed_index),
                        "seed": int(seed),
                        "eps_times": float(eps_times),
                        "effective_k": integer_k,
                        "embedding_key": embedding_key,
                        "embedding": np.asarray(
                            embeddings[seed_index, eps_index, k_index], dtype=float
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
) -> pd.DataFrame:
    if not bool(np.asarray(cache.get("labels_available", False))):
        raise ValueError("Aggregate cache does not contain labels.")

    dataset = str(np.asarray(cache["dataset"]).item())
    batch_ids = np.asarray(cache["batch_ids"], dtype=int)
    labels0 = np.asarray(cache["labels0"])
    labels1 = np.asarray(cache["labels1"])
    n0 = labels0.shape[1]
    rows: list[dict[str, Any]] = []

    for record in iter_aggregate_embeddings(cache, methods):
        seed_index = int(record["seed_index"])
        labels = np.concatenate([labels0[seed_index], labels1[seed_index]])
        embedding = np.asarray(record.pop("embedding"), dtype=float)
        y0, y1 = embedding[:n0], embedding[n0:]

        row = {
            "dataset": dataset,
            "source_cache": "" if source_cache is None else str(source_cache),
            **record,
        }
        metric_calls = {
            "neighborhood_purity_score": (compute_neighbor_purity, (embedding, labels), {}),
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
                try:
                    scores = function(*args, **kwargs)
                    row["kmeans_rand_index"] = float(scores["rand_index"])
                    row["kmeans_adjusted_rand_index"] = float(
                        scores["adjusted_rand_index"]
                    )
                    row["kmeans_rand_index_reason"] = ""
                    row["kmeans_adjusted_rand_index_reason"] = ""
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    row["kmeans_rand_index"] = np.nan
                    row["kmeans_adjusted_rand_index"] = np.nan
                    row["kmeans_rand_index_reason"] = reason
                    row["kmeans_adjusted_rand_index_reason"] = reason
                continue
            value, reason = _metric_result(function, *args, **kwargs)
            row[metric_name] = value
            row[f"{metric_name}_reason"] = reason

        references = _reference_batches(cache, seed_index)
        if references is None:
            reason = "aggregate cache does not include reference_x0/reference_x1"
            for metric_name in (
                "concordance_score",
                "cross_concordance_score",
                "foscttm_score",
            ):
                row[metric_name] = np.nan
                row[f"{metric_name}_reason"] = reason
        else:
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
                value, reason = _metric_result(function, *args, **kwargs)
                row[metric_name] = value
                row[f"{metric_name}_reason"] = reason

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_metric_rows(rows: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "dataset",
        "method",
        "eps_times",
        "effective_k",
    ]
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
        frames.append(
            score_aggregate_cache(
                load_aggregate_cache(path), methods, source_cache=path
            )
        )
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = summarize_metric_rows(rows)

    output_path = Path(output_csv).expanduser()
    summary_path = Path(summary_csv).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False)
    summary.to_csv(summary_path, index=False)
    return rows, summary
