"""Aggregate-cache construction and scoring."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ne_aligned_utils.benchmark.data import (
    canonical_dataset_name as _canonical_dataset_name,
    load_benchmark_data as _load_benchmark_data,
    resolve_data_root as _resolve_data_root,
)
from ne_aligned_utils.benchmark.sweep.config import (
    DEFAULT_REL_REG_M,
    DEFAULT_SYMMETRIZATION,
    _resolved_n_neighbors,
    resolve_aggregate_output_path,
    resolve_task_output_path,
    resolve_task_root,
)
from ne_aligned_utils.benchmark.sweep.registry import (
    UOT_METHODS,
    ZERODIAG_METHODS,
    input_kind_for_method,
)
from ne_aligned_utils.benchmark.sweep.tasks import save_npz
from ne_aligned_utils.config import DEFAULT_BACKEND, canonical_name, method_stem


def _load_task_embedding(
    path: Path,
    *,
    n_total: int,
    dataset: str,
    task_kind: str,
    seed: int,
    sample_indices0: np.ndarray,
    sample_indices1: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Load one task only after proving that it belongs in this aggregate."""
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as cache:
        required = {
            "embedding",
            "dataset",
            "task_kind",
            "seed",
            "batch_sizes",
            "sample_indices0",
            "sample_indices1",
            "input_source",
            "source_path",
            "n_genes",
        }
        missing = sorted(required.difference(cache.files))
        if missing:
            raise ValueError(f"{path} lacks required task metadata: {', '.join(missing)}.")

        actual_dataset = _canonical_dataset_name(str(np.asarray(cache["dataset"]).item()))
        expected_dataset = _canonical_dataset_name(dataset)
        if actual_dataset != expected_dataset:
            raise ValueError(
                f"{path} contains dataset {actual_dataset!r}, expected {expected_dataset!r}."
            )
        actual_kind = str(np.asarray(cache["task_kind"]).item())
        if actual_kind != task_kind:
            raise ValueError(f"{path} contains task_kind {actual_kind!r}, expected {task_kind!r}.")
        actual_seed = int(np.asarray(cache["seed"]).item())
        if actual_seed != int(seed):
            raise ValueError(f"{path} contains seed {actual_seed}, expected {seed}.")

        expected_indices = (
            np.asarray(sample_indices0, dtype=int),
            np.asarray(sample_indices1, dtype=int),
        )
        expected_sizes = np.asarray([len(expected_indices[0]), len(expected_indices[1])], dtype=int)
        if not np.array_equal(np.asarray(cache["batch_sizes"], dtype=int), expected_sizes):
            raise ValueError(
                f"{path} has batch sizes that do not match the requested dataset sample."
            )
        for key, expected in zip(("sample_indices0", "sample_indices1"), expected_indices):
            if not np.array_equal(np.asarray(cache[key], dtype=int), expected):
                raise ValueError(f"{path} has {key} for a different cell sample.")

        input_source = str(np.asarray(cache["input_source"]).item())
        expected_source = input_kind_for_method(task_kind)
        if input_source != expected_source:
            raise ValueError(
                f"{path} records input_source {input_source!r}, expected {expected_source!r} "
                f"for {task_kind!r}."
            )
        embedding = np.asarray(cache["embedding"], dtype=float)
        provenance = {
            "input_source": input_source,
            "source_path": str(np.asarray(cache["source_path"]).item()),
            "n_genes": int(np.asarray(cache["n_genes"]).item()),
        }
    if embedding.ndim != 2 or embedding.shape[0] != int(n_total):
        raise ValueError(f"{path} has embedding shape {embedding.shape}, expected {n_total} rows.")
    return embedding, provenance


def _simple_paths(
    task_root: Path, task_kind: str, seeds: np.ndarray, backend: str = DEFAULT_BACKEND
) -> np.ndarray:
    return np.asarray(
        [
            str(
                resolve_task_output_path(
                    task_root, task_kind=task_kind, seed=int(seed), backend=backend
                )
            )
            for seed in seeds.tolist()
        ]
    )


def _eps_paths(
    task_root: Path,
    task_kind: str,
    seeds: np.ndarray,
    rel_reg_values: np.ndarray,
    backend: str = DEFAULT_BACKEND,
) -> np.ndarray:
    paths = np.empty((len(seeds), len(rel_reg_values)), dtype=object)
    for seed_idx, seed in enumerate(seeds.tolist()):
        for rel_reg_idx, rel_reg in enumerate(rel_reg_values.tolist()):
            paths[seed_idx, rel_reg_idx] = str(
                resolve_task_output_path(
                    task_root,
                    task_kind=task_kind,
                    seed=int(seed),
                    rel_reg=float(rel_reg),
                    backend=backend,
                )
            )
    return paths


def _eps_k_paths(
    task_root: Path,
    task_kind: str,
    seeds: np.ndarray,
    rel_reg_values: np.ndarray,
    effective_ks: np.ndarray,
    backend: str = DEFAULT_BACKEND,
) -> np.ndarray:
    paths = np.empty((len(seeds), len(rel_reg_values), len(effective_ks)), dtype=object)
    for seed_idx, seed in enumerate(seeds.tolist()):
        for rel_reg_idx, rel_reg in enumerate(rel_reg_values.tolist()):
            for k_idx, effective_k in enumerate(effective_ks.tolist()):
                paths[seed_idx, rel_reg_idx, k_idx] = str(
                    resolve_task_output_path(
                        task_root,
                        task_kind=task_kind,
                        seed=int(seed),
                        rel_reg=float(rel_reg),
                        effective_k=int(effective_k),
                        backend=backend,
                    )
                )
    return paths


def _stack_path_array(
    paths: np.ndarray,
    *,
    n_total: int,
    dataset: str,
    task_kind: str,
    seeds: np.ndarray,
    sample_indices0: np.ndarray,
    sample_indices1: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    flat_embeddings = []
    provenance = None
    for index in np.ndindex(paths.shape):
        seed_index = index[0]
        embedding, current = _load_task_embedding(
            Path(paths[index]),
            n_total=n_total,
            dataset=dataset,
            task_kind=task_kind,
            seed=int(seeds[seed_index]),
            sample_indices0=sample_indices0[seed_index],
            sample_indices1=sample_indices1[seed_index],
        )
        if provenance is None:
            provenance = current
        elif current != provenance:
            raise ValueError(
                f"{paths[index]} has different input provenance from this method's other tasks."
            )
        flat_embeddings.append(embedding)
    first_shape = flat_embeddings[0].shape
    for path, embedding in zip(paths.reshape(-1), flat_embeddings):
        if embedding.shape != first_shape:
            raise ValueError(
                f"{path} has embedding shape {embedding.shape}, expected {first_shape}."
            )
    return np.stack(flat_embeddings, axis=0).reshape(paths.shape + first_shape), provenance


def _store_aggregate_method(
    payload: dict[str, Any],
    *,
    method: str,
    backend: str,
    paths: np.ndarray,
    n_total: int,
    dataset: str,
    seeds: np.ndarray,
    sample_indices0: np.ndarray,
    sample_indices1: np.ndarray,
) -> None:
    """Validate and store one method's embeddings, paths, and input provenance."""
    embedding_key, path_key = cache_keys(method, backend)
    embeddings, provenance = _stack_path_array(
        paths,
        n_total=n_total,
        dataset=dataset,
        task_kind=method,
        seeds=seeds,
        sample_indices0=sample_indices0,
        sample_indices1=sample_indices1,
    )
    stem = embedding_key.removesuffix("_embeddings")
    payload[path_key] = paths
    payload[embedding_key] = embeddings
    payload[f"{stem}_input_source"] = provenance["input_source"]
    payload[f"{stem}_source_path"] = provenance["source_path"]
    payload[f"{stem}_n_genes"] = provenance["n_genes"]


#: Metadata that must agree before aggregate arrays can be merged.
AGGREGATE_GEOMETRY_KEYS = (
    "dataset",
    "seeds",
    "rel_reg",
    "effective_ks",
    "batch_sizes",
    "embedding_dim",
    "n_epochs",
    "pca_components",
    "perplexity",
    "init",
    "backend",
    "n_neighbors",
    "rel_reg_m",
    "graph_symmetrization",
    "knndiag_symmetrization",
)

#: Semantics of geometry fields absent from older caches.
AGGREGATE_GEOMETRY_DEFAULTS: dict[str, object] = {
    "backend": DEFAULT_BACKEND,
    "n_neighbors": -1,
    # Older caches used the balanced limit.
    "rel_reg_m": DEFAULT_REL_REG_M,
    # Older graph caches used union ("or") symmetrization.
    "graph_symmetrization": DEFAULT_SYMMETRIZATION,
    "knndiag_symmetrization": DEFAULT_SYMMETRIZATION,
}


def _merge_previous_methods(payload: dict[str, Any], output_path: Path) -> dict[str, Any]:
    """Carry forward unrequested methods when aggregate geometry matches."""
    if not output_path.exists():
        return payload
    with np.load(output_path, allow_pickle=True) as previous:
        if "eps_times" in previous.files:
            # The reciprocal eps_times axis is incompatible with rel_reg indexing.
            raise ValueError(
                f"{output_path} is an eps_times aggregate and cannot be merged with a rel_reg "
                "run: the grid axis is reciprocal, so the stored arrays would be misindexed. "
                "Write the rel_reg sweep to its own results tree (NE_ALIGNED_RESULTS_DIR)."
            )
        for key in AGGREGATE_GEOMETRY_KEYS:
            if key in previous.files:
                earlier = previous[key]
            elif key in AGGREGATE_GEOMETRY_DEFAULTS:
                # Absent means the cache predates this key, and the default says what it meant.
                earlier = np.asarray(AGGREGATE_GEOMETRY_DEFAULTS[key])
            else:
                continue
            if key not in payload:
                continue
            current = payload[key]
            if not np.array_equal(np.asarray(earlier), np.asarray(current)):
                print(
                    f"aggregate: {key} differs from the existing cache, so it is replaced rather "
                    f"than extended (was {earlier!r}, now {payload[key]!r})"
                )
                return payload
        kept = []
        provenance_suffixes = ("_input_source", "_source_path", "_n_genes")
        for key in previous.files:
            if key.endswith(("_embeddings", "_paths", *provenance_suffixes)) and key not in payload:
                payload[key] = previous[key]
                kept.append(key)
        if kept:
            earlier = [str(m) for m in np.asarray(previous["methods"]).tolist()]
            current = [str(m) for m in np.asarray(payload["methods"]).tolist()]
            # Existing order first, so methods keep their position and downstream row order.
            merged = earlier + [m for m in current if m not in earlier]
            payload["methods"] = np.asarray(merged)
            print(
                f"aggregate: extended the existing cache, keeping "
                f"{sum(key.endswith('_embeddings') for key in kept)} method(s) it already held"
            )
    return payload


def aggregate(args: argparse.Namespace) -> None:
    task_root = resolve_task_root(args)
    # The filenames carry the backend token, so aggregate has to know it or it looks for t-SNE
    # names inside another backend's tree and reports every task as missing.
    backend = str(getattr(args, "backend", DEFAULT_BACKEND))
    output_path = (
        resolve_aggregate_output_path(task_root, args.dataset)
        if args.output_cache is None
        else Path(args.output_cache).expanduser()
    )

    seeds = np.arange(int(args.n_seeds), dtype=int)
    rel_reg_values = np.asarray(args.rel_reg, dtype=float)
    effective_ks = np.asarray(args.effective_ks, dtype=int)
    methods = tuple(args.methods)

    sample_seed = int(seeds[0] if getattr(args, "sample_seed", None) is None else args.sample_seed)
    data0 = _load_benchmark_data(
        args.dataset,
        data_root=_resolve_data_root(args.data_root),
        seed=sample_seed,
        max_cells_per_batch=args.max_cells_per_batch,
        max_cells_batch0=args.max_cells_batch0,
        max_cells_batch1=args.max_cells_batch1,
    )
    n0, n1 = data0.x0.shape[0], data0.x1.shape[0]
    n_total = n0 + n1

    sample_indices0 = np.empty((len(seeds), n0), dtype=int)
    sample_indices1 = np.empty((len(seeds), n1), dtype=int)
    labels_available = bool(data0.labels_available)
    labels0 = (
        np.empty((len(seeds), n0), dtype=int) if labels_available else np.empty((0,), dtype=int)
    )
    labels1 = (
        np.empty((len(seeds), n1), dtype=int) if labels_available else np.empty((0,), dtype=int)
    )

    for seed_idx, _seed in enumerate(seeds.tolist()):
        data = _load_benchmark_data(
            args.dataset,
            data_root=_resolve_data_root(args.data_root),
            seed=sample_seed,
            max_cells_per_batch=args.max_cells_per_batch,
            max_cells_batch0=args.max_cells_batch0,
            max_cells_batch1=args.max_cells_batch1,
        )
        if data.x0.shape[0] != n0 or data.x1.shape[0] != n1:
            raise ValueError("Aggregate requires identical sampled batch sizes across seeds.")
        sample_indices0[seed_idx] = data.sample_indices0
        sample_indices1[seed_idx] = data.sample_indices1
        if labels_available:
            if not data.labels_available:
                raise ValueError("Labels were available for seed 0 but missing for a later seed.")
            labels0[seed_idx] = data.labels0
            labels1[seed_idx] = data.labels1

    payload: dict[str, Any] = {
        "dataset": args.dataset,
        "methods": np.asarray([canonical_name(m, backend) for m in methods]),
        "seeds": seeds,
        "sample_seed": sample_seed,
        "rel_reg": rel_reg_values,
        "effective_ks": effective_ks,
        # These settings describe the whole cache, not additional array axes.
        "init": str(getattr(args, "init", "random")),
        "rel_reg_m": float(getattr(args, "rel_reg_m", DEFAULT_REL_REG_M)),
        "graph_symmetrization": str(getattr(args, "graph_symmetrization", DEFAULT_SYMMETRIZATION)),
        "knndiag_symmetrization": str(
            getattr(args, "knndiag_symmetrization", DEFAULT_SYMMETRIZATION)
        ),
        "embedding_dim": int(args.embedding_dim),
        "n_epochs": int(args.n_epochs),
        # PCA width applies to either visualization backend.
        "pca_components": int(args.pca_components),
        "perplexity": float(args.perplexity),
        "backend": str(getattr(args, "backend", DEFAULT_BACKEND)),
        "n_neighbors": _resolved_n_neighbors(args),
        "max_cells_per_batch": -1
        if args.max_cells_per_batch is None
        else int(args.max_cells_per_batch),
        "max_cells_batch0": -1 if args.max_cells_batch0 is None else int(args.max_cells_batch0),
        "max_cells_batch1": -1 if args.max_cells_batch1 is None else int(args.max_cells_batch1),
        "batch_sizes": np.asarray([n0, n1], dtype=int),
        "batch_ids": np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]),
        "dataset_names": np.asarray(data0.dataset_names),
        "labels_available": labels_available,
        "labels0": labels0,
        "labels1": labels1,
        "label_names": np.asarray(data0.label_names),
        "sample_indices0": sample_indices0,
        "sample_indices1": sample_indices1,
    }

    for method in SIMPLE_CACHE_METHODS:
        if method not in methods:
            continue
        paths = _simple_paths(task_root, method, seeds, backend=backend)
        _store_aggregate_method(
            payload,
            method=method,
            backend=backend,
            paths=paths,
            n_total=n_total,
            dataset=args.dataset,
            seeds=seeds,
            sample_indices0=sample_indices0,
            sample_indices1=sample_indices1,
        )

    for method in EPS_CACHE_METHODS:
        if method not in methods:
            continue
        paths = _eps_paths(task_root, method, seeds, rel_reg_values, backend=backend)
        _store_aggregate_method(
            payload,
            method=method,
            backend=backend,
            paths=paths,
            n_total=n_total,
            dataset=args.dataset,
            seeds=seeds,
            sample_indices0=sample_indices0,
            sample_indices1=sample_indices1,
        )

    for method in EPS_K_CACHE_METHODS:
        if method not in methods:
            continue
        paths = _eps_k_paths(
            task_root, method, seeds, rel_reg_values, effective_ks, backend=backend
        )
        _store_aggregate_method(
            payload,
            method=method,
            backend=backend,
            paths=paths,
            n_total=n_total,
            dataset=args.dataset,
            seeds=seeds,
            sample_indices0=sample_indices0,
            sample_indices1=sample_indices1,
        )

    if not getattr(args, "replace", False):
        payload = _merge_previous_methods(payload, output_path)
    save_npz(output_path, **payload)
    print(f"Saved aggregated embedding cache to {output_path}")


#: Number of sweep axes in each method's stored embedding array.
SIMPLE_CACHE_METHODS: tuple[str, ...] = (
    "joint-pca",
    "rc-tsne",
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
)
EPS_CACHE_METHODS: tuple[str, ...] = ("eot-t0", "eot-50")
EPS_K_CACHE_METHODS: tuple[str, ...] = (
    "ot",
    "ot-knndiag",
    "ot-selfotmedian",
    "ot-gramdiag",
    *UOT_METHODS,
    *ZERODIAG_METHODS,
)


def cache_keys(method: str, backend: str = DEFAULT_BACKEND) -> tuple[str, str]:
    """The ``(embeddings, paths)`` array names a method's entry takes in an aggregate cache."""
    stem = method_stem(method, backend)
    return f"{stem}_embeddings", f"{stem}_paths"


def score_aggregate(args: argparse.Namespace) -> None:
    from ne_aligned_utils.benchmark.scoring import (
        load_aggregate_cache,
        score_aggregate_cache,
    )

    aggregate_path = Path(args.aggregate_cache).expanduser()
    rows = score_aggregate_cache(
        load_aggregate_cache(aggregate_path),
        methods=args.methods,
        source_cache=aggregate_path,
    )

    output_csv = (
        aggregate_path.with_name(f"{aggregate_path.stem}_metrics.csv")
        if args.output_csv is None
        else Path(args.output_csv).expanduser()
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)
    print(f"Saved aggregate metrics to {output_csv}")



__all__ = [
    "AGGREGATE_GEOMETRY_DEFAULTS",
    "AGGREGATE_GEOMETRY_KEYS",
    "EPS_CACHE_METHODS",
    "EPS_K_CACHE_METHODS",
    "SIMPLE_CACHE_METHODS",
    "_merge_previous_methods",
    "aggregate",
    "cache_keys",
    "resolve_aggregate_output_path",
    "score_aggregate",
]
