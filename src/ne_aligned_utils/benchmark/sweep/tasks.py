"""Single-task execution and artifact handling."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from ne_aligned_utils.benchmark.data import (
    BenchmarkData as _BenchmarkData,
    all_gene_library_sizes as _all_gene_library_sizes,
    load_benchmark_data as _load_benchmark_data,
    resolve_data_root as _resolve_data_root,
)
from ne_aligned_utils.benchmark.sweep.config import (
    COUNT_NORMALIZED_METHODS,
    DEFAULT_COUNT_LIBRARY_SIZE,
    DEFAULT_REL_REG_M,
    DEFAULT_SYMMETRIZATION,
    DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    LEGACY_COUNT_LIBRARY_SIZE,
    _legacy_watched_values,
    _resolved_n_neighbors,
    assert_no_parameter_collision,
    assert_output_backend_matches,
    resolve_task_output_path,
    resolve_task_root,
)
from ne_aligned_utils.benchmark.sweep.registry import (
    EPS_K_METHODS,
    KNNDIAG_SYMMETRIZATION_METHODS,
    OT_PLAN_METHODS,
    R_METHODS,
    SYMMETRIZATION_METHODS,
    UOT_METHODS,
    VISUALIZED_METHODS,
    input_kind_for_method,
)
from ne_aligned_utils.config import (
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_MULTIMAP_INIT,
    DEFAULT_MULTIMAP_MIN_DIST,
    DEFAULT_MULTIMAP_SPREAD,
    DEFAULT_MULTIMAP_USE_NATIVE_SCALING,
    DEFAULT_REF_TSNE_PRESELECTED_FEATURES,
    DEFAULT_SCVI_BATCH_SIZE,
    DEFAULT_SCVI_MAX_EPOCHS,
    DEFAULT_SEURAT_PRESELECTED_FEATURES,
    DEFAULT_SEURAT_WORKFLOW,
    METHOD_SPECS,
    REPRESENTATION_ARTIFACT_KEYS,
    has_backend_axis,
    method_backend_label,
    resolve_backend,
    result_dir,
    run_uses_perplexity,
)

UNPERSISTED_ARTIFACT_KEYS = frozenset(
    {
        "four_block_matrix", "sparse_four_block_matrix", "graph", "ot_plan", "u", "vt",
        "pca_embedding", "reference_input", "query_input", "r_timings", "multimap_timings",
    }
)


def _tsne_early_exaggeration_iter(args: argparse.Namespace) -> int | None:
    """Early-exaggeration length for t-SNE layouts; UMAP has no such phase."""
    if str(getattr(args, "backend", DEFAULT_BACKEND)) != "tsne":
        return None
    return int(
        getattr(args, "tsne_early_exaggeration_iter", DEFAULT_TSNE_EARLY_EXAGGERATION_ITER)
    )


def _count_library_size(args: argparse.Namespace) -> str:
    return str(getattr(args, "count_library_size", DEFAULT_COUNT_LIBRARY_SIZE))


def _count_library_sizes(args: argparse.Namespace, data: _BenchmarkData) -> np.ndarray | None:
    """Per-cell divisors for count normalisation, batch 0 then batch 1.

    ``selected_genes`` returns None: the method divides by each cell's total over the supplied
    genes, as before. ``all_genes`` returns each cell's total over all genes, the divisor the
    processed features were normalised with, recovered from the same cells' processed view.
    """
    if _count_library_size(args) != "all_genes":
        return None
    processed = _load_benchmark_data(
        args.dataset,
        data_root=_resolve_data_root(args.data_root),
        seed=int(args.seed if getattr(args, "sample_seed", None) is None else args.sample_seed),
        max_cells_per_batch=args.max_cells_per_batch,
        max_cells_batch0=args.max_cells_batch0,
        max_cells_batch1=args.max_cells_batch1,
        input_kind="processed",
    )
    sizes0, sizes1 = _all_gene_library_sizes(processed, data)
    return np.concatenate([sizes0, sizes1])


def _npz_value(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray(-1)
    if isinstance(value, Path):
        return np.asarray(str(value))
    if isinstance(value, dict):
        return np.asarray(str(value))
    array = np.asarray(value)
    if array.dtype == object:
        return array.astype(str)
    return array


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: _npz_value(value) for key, value in arrays.items()})


def load_cached_representation(
    args: argparse.Namespace,
) -> tuple[np.ndarray, Path, str]:
    """Load an explicitly requested representation and return its path and SHA-256."""
    import hashlib

    key = REPRESENTATION_ARTIFACT_KEYS.get(args.task_kind)
    if key is None:
        raise ValueError(
            f"{args.task_kind!r} does not persist a reusable representation. "
            f"Reusable kinds: {sorted(REPRESENTATION_ARTIFACT_KEYS)}. "
            "joint and ref are excluded on purpose -- their representations are "
            "unpersisted and are cheap deterministic PCAs."
        )
    # Resolve tree names centrally so the dataset component is retained.
    source_root = resolve_task_root(
        argparse.Namespace(
            **{
                **vars(args),
                "output_dir": result_dir(DEFAULT_BACKEND, str(args.reuse_from), args.dataset),
                # The source t-SNE tree has no UMAP neighborhood component.
                "backend": DEFAULT_BACKEND,
                "n_neighbors": None,
            }
        )
    )
    source = resolve_task_output_path(
        source_root,
        task_kind=args.task_kind,
        seed=int(args.seed),
        rel_reg=getattr(args, "rel_reg", None),
        effective_k=getattr(args, "effective_k", None),
        backend=DEFAULT_BACKEND,
    )
    if not source.exists():
        raise FileNotFoundError(
            f"No cached representation for {args.task_kind!r} seed={args.seed} at {source}. "
            "Run the default backend first, or drop --reuse-from to recompute the integration."
        )
    with np.load(source, allow_pickle=True) as cached:
        if key not in cached.files:
            raise KeyError(
                f"{source} has no {key!r}. It may predate the artifact being persisted, or the "
                "key may be in UNPERSISTED_ARTIFACT_KEYS for this method."
            )
        representation = np.asarray(cached[key], dtype=float)
    digest = hashlib.sha256(representation.tobytes()).hexdigest()
    return representation, source, digest


def common_metadata(
    data: _BenchmarkData, args: argparse.Namespace, *, task_kind: str
) -> dict[str, Any]:
    """Build the applicable metadata fields for one stored result."""
    labels_available = bool(data.labels_available)
    metadata: dict[str, Any] = {
        # task_kind is backend-neutral; method below records what actually ran.
        "task_kind": task_kind,
        "dataset": data.dataset,
        "seed": int(args.seed),
        "preprocessing_seed": int(getattr(args, "preprocessing_seed", -1)),
        "sample_seed": int(
            args.seed if getattr(args, "sample_seed", None) is None else args.sample_seed
        ),
        "method_seed": int(
            args.seed if getattr(args, "method_seed", None) is None else args.method_seed
        ),
        "visualization_seed": int(
            args.seed
            if getattr(args, "visualization_seed", None) is None
            else args.visualization_seed
        ),
        "embedding_dim": int(args.embedding_dim),
        "n_epochs": int(args.n_epochs),
        "tsne_early_exaggeration_iter": _tsne_early_exaggeration_iter(args),
        "pca_components": int(args.pca_components),
        "perplexity": float(args.perplexity),
        "backend": str(getattr(args, "backend", DEFAULT_BACKEND)),
        "n_neighbors": _resolved_n_neighbors(args),
        "max_cells_per_batch": -1
        if args.max_cells_per_batch is None
        else int(args.max_cells_per_batch),
        "max_cells_batch0": -1 if args.max_cells_batch0 is None else int(args.max_cells_batch0),
        "max_cells_batch1": -1 if args.max_cells_batch1 is None else int(args.max_cells_batch1),
        "batch_sizes": np.asarray([data.x0.shape[0], data.x1.shape[0]], dtype=int),
        "batch_ids": data.batch_ids,
        "sample_indices0": np.asarray(data.sample_indices0, dtype=int),
        "sample_indices1": np.asarray(data.sample_indices1, dtype=int),
        "dataset_names": np.asarray(data.dataset_names),
        "labels_available": labels_available,
        "labels0": np.asarray(data.labels0 if labels_available else [], dtype=int),
        "labels1": np.asarray(data.labels1 if labels_available else [], dtype=int),
        "label_names": np.asarray(data.label_names),
        "input_source": str(data.input_source),
        "source_path": "" if data.source_path is None else str(data.source_path),
        "n_genes": int(data.x0.shape[1]),
    }
    for field in (
        "run_id",
        "preprocessing_hash",
        "sample_hash",
        "representation_family",
        "method_pipeline",
        "benchmark_manifest_hash",
    ):
        value = getattr(args, field, None)
        if value is not None:
            metadata[field] = str(value)
    if data.feature_names:
        metadata["feature_names"] = np.asarray(data.feature_names)
    # Remove metadata that does not apply to this method.
    backend = str(metadata["backend"])
    if task_kind in METHOD_SPECS:
        spec = METHOD_SPECS[task_kind]
        metadata["backend"] = resolve_backend(task_kind, backend)
        # Store the backend-qualified method name.
        metadata["method"] = (
            method_backend_label(task_kind, metadata["backend"]) if spec.backend_axis else task_kind
        )
        if not run_uses_perplexity(task_kind, backend):
            metadata.pop("perplexity", None)
        if spec.tree == "2d_outputs":
            # Direct 2D layouts have no visualizer metadata.
            metadata.pop("n_epochs", None)
            metadata.pop("tsne_early_exaggeration_iter", None)
            metadata.pop("pca_components", None)
    return metadata


def _compute_task(
    data: _BenchmarkData,
    args: argparse.Namespace,
    timer: Any | None = None,
    library_sizes: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Dispatch one benchmark method and collect its artifacts and metadata.

    ``library_sizes`` are the count-normalisation divisors from ``_count_library_sizes``,
    computed by the caller outside the timed region.
    """
    from ne_aligned_utils.methods import (
        compute_cca_embedding_from_arrays,
        compute_cca_then_tsne_embedding_from_arrays,
        compute_eot_embedding_from_arrays,
        compute_eot_then_tsne_embedding_from_arrays,
        compute_harmony_embedding_from_arrays,
        compute_harmony_then_tsne_embedding_from_arrays,
        compute_ot_tsne_embedding_from_arrays,
        compute_ot_tsne_gramdiag_embedding_from_arrays,
        compute_ot_tsne_knndiag_embedding_from_arrays,
        compute_ot_tsne_selfotmedian_embedding_from_arrays,
        compute_ot_tsne_uot_mass_embedding_from_arrays,
        compute_ot_tsne_uot_topk_embedding_from_arrays,
        compute_ot_tsne_zerodiag_embedding_from_arrays,
        compute_ot_tsne_zerodiag_split_embedding_from_arrays,
        compute_pca_embedding_from_arrays,
        compute_policar_ref_tsne_embedding_from_arrays,
        compute_ref_tsne_embedding_from_arrays,
        compute_scanorama_embedding_from_arrays,
        compute_scanorama_then_tsne_embedding_from_arrays,
        compute_scvi_embedding_from_arrays,
        compute_scvi_then_tsne_embedding_from_arrays,
        compute_tsne_embedding_from_arrays,
    )

    # Resolve the OT device once for every OT branch.
    ot_device = str(getattr(args, "ot_device", "cpu"))

    if args.task_kind == "eot-t0":
        embedding, artifacts = compute_eot_embedding_from_arrays(
            data.x0,
            data.x1,
            t=0.0,
            rel_reg=float(args.rel_reg),
            tol=float(args.tol),
            max_iter=int(args.max_iter),
            embd_dim=int(args.embedding_dim),
            device=ot_device,
            return_artifacts=True,
            timer=timer,
        )
        return embedding, artifacts, {"rel_reg": float(args.rel_reg), "t": 0.0}

    # Initialization applies only to the OT-tSNE family.
    ot_init = str(getattr(args, "init", "random"))
    # Only the epsilon-plus-k methods require both values.
    ot_params = (
        {
            "rel_reg": float(args.rel_reg),
            "effective_k": int(args.effective_k),
            "init": ot_init,
        }
        if args.task_kind in EPS_K_METHODS
        else {}
    )
    if args.task_kind in UOT_METHODS:
        # Record relaxation in the task file as well as the directory.
        ot_params["rel_reg_m"] = float(getattr(args, "rel_reg_m", DEFAULT_REL_REG_M))
    # Record graph construction choices in each task file.
    symmetrization_params: dict[str, object] = {}
    if args.task_kind in SYMMETRIZATION_METHODS:
        symmetrization_params["graph_symmetrization"] = str(
            getattr(args, "graph_symmetrization", DEFAULT_SYMMETRIZATION)
        )
        ot_params["graph_symmetrization"] = symmetrization_params["graph_symmetrization"]
    if args.task_kind in KNNDIAG_SYMMETRIZATION_METHODS:
        symmetrization_params["knndiag_symmetrization"] = str(
            getattr(args, "knndiag_symmetrization", DEFAULT_SYMMETRIZATION)
        )
        ot_params["knndiag_symmetrization"] = symmetrization_params["knndiag_symmetrization"]

    if args.task_kind == "ot":
        embedding, artifacts = compute_ot_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            rel_reg=float(args.rel_reg),
            effective_k=int(args.effective_k),
            random_state=int(args.seed),
            n_epochs=int(args.n_epochs),
            early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
            n_jobs=int(args.n_jobs),
            tol=float(args.tol),
            max_iter=int(args.max_iter),
            init=ot_init,
            backend=str(args.backend),
            device=ot_device,
            **symmetrization_params,
            return_artifacts=True,
            timer=timer,
        )
        return embedding, artifacts, dict(ot_params)

    if args.task_kind == "ot-knndiag":
        embedding, artifacts = compute_ot_tsne_knndiag_embedding_from_arrays(
            data.x0,
            data.x1,
            rel_reg=float(args.rel_reg),
            effective_k=int(args.effective_k),
            random_state=int(args.seed),
            n_epochs=int(args.n_epochs),
            early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
            n_jobs=int(args.n_jobs),
            tol=float(args.tol),
            max_iter=int(args.max_iter),
            init=ot_init,
            backend=str(args.backend),
            device=ot_device,
            **symmetrization_params,
            return_artifacts=True,
            timer=timer,
        )
        return embedding, artifacts, dict(ot_params)

    if args.task_kind == "ot-selfotmedian":
        embedding, artifacts = compute_ot_tsne_selfotmedian_embedding_from_arrays(
            data.x0,
            data.x1,
            rel_reg=float(args.rel_reg),
            effective_k=int(args.effective_k),
            random_state=int(args.seed),
            n_epochs=int(args.n_epochs),
            early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
            n_jobs=int(args.n_jobs),
            tol=float(args.tol),
            max_iter=int(args.max_iter),
            init=ot_init,
            backend=str(args.backend),
            device=ot_device,
            **symmetrization_params,
            return_artifacts=True,
            timer=timer,
        )
        return embedding, artifacts, dict(ot_params)

    if args.task_kind == "ot-gramdiag":
        embedding, artifacts = compute_ot_tsne_gramdiag_embedding_from_arrays(
            data.x0,
            data.x1,
            rel_reg=float(args.rel_reg),
            effective_k=int(args.effective_k),
            random_state=int(args.seed),
            n_epochs=int(args.n_epochs),
            early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
            n_jobs=int(args.n_jobs),
            tol=float(args.tol),
            max_iter=int(args.max_iter),
            init=ot_init,
            backend=str(args.backend),
            device=ot_device,
            **symmetrization_params,
            return_artifacts=True,
            timer=timer,
        )
        return embedding, artifacts, dict(ot_params)

    if args.task_kind == "ot-zerodiag":
        embedding, artifacts = compute_ot_tsne_zerodiag_embedding_from_arrays(
            data.x0,
            data.x1,
            rel_reg=float(args.rel_reg),
            effective_k=int(args.effective_k),
            random_state=int(args.seed),
            n_epochs=int(args.n_epochs),
            early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
            n_jobs=int(args.n_jobs),
            tol=float(args.tol),
            max_iter=int(args.max_iter),
            init=ot_init,
            backend=str(args.backend),
            device=ot_device,
            **symmetrization_params,
            return_artifacts=True,
            timer=timer,
        )
        return embedding, artifacts, dict(ot_params)

    if args.task_kind == "ot-zerodiag-split":
        embedding, artifacts = compute_ot_tsne_zerodiag_split_embedding_from_arrays(
            data.x0,
            data.x1,
            rel_reg=float(args.rel_reg),
            effective_k=int(args.effective_k),
            random_state=int(args.seed),
            n_epochs=int(args.n_epochs),
            early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
            n_jobs=int(args.n_jobs),
            tol=float(args.tol),
            max_iter=int(args.max_iter),
            init=ot_init,
            backend=str(args.backend),
            device=ot_device,
            **symmetrization_params,
            return_artifacts=True,
            timer=timer,
        )
        return embedding, artifacts, dict(ot_params)

    if args.task_kind in UOT_METHODS:
        uot_method = (
            compute_ot_tsne_uot_topk_embedding_from_arrays
            if args.task_kind == "ot-uot-topk"
            else compute_ot_tsne_uot_mass_embedding_from_arrays
        )
        embedding, artifacts = uot_method(
            data.x0,
            data.x1,
            rel_reg=float(args.rel_reg),
            rel_reg_m=float(getattr(args, "rel_reg_m", DEFAULT_REL_REG_M)),
            effective_k=int(args.effective_k),
            random_state=int(args.seed),
            n_epochs=int(args.n_epochs),
            early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
            n_jobs=int(args.n_jobs),
            tol=float(args.tol),
            max_iter=int(args.max_iter),
            init=ot_init,
            backend=str(args.backend),
            device=ot_device,
            return_artifacts=True,
            timer=timer,
        )
        return embedding, artifacts, dict(ot_params)

    if args.task_kind == "eot-50":
        embedding, artifacts = compute_eot_then_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            t=0.0,
            rel_reg=float(args.rel_reg),
            eot_dim=50,
            random_state=int(args.seed),
            perplexity=float(args.perplexity),
            backend=str(args.backend),
            n_neighbors=args.n_neighbors,
            n_jobs=int(args.n_jobs),
            tol=float(args.tol),
            max_iter=int(args.max_iter),
            n_iter=int(args.n_epochs),
            early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
            device=ot_device,
            return_artifacts=True,
            timer=timer,
        )
        return embedding, artifacts, {"rel_reg": float(args.rel_reg), "t": 0.0, "eot_dim": 50}

    if args.task_kind == "joint-pca":
        # Joint PCA has no separable stages; the parent measures its total.
        embedding, artifacts = compute_pca_embedding_from_arrays(
            data.x0,
            data.x1,
            embd_dim=int(args.embedding_dim),
            random_state=int(args.seed),
            return_artifacts=True,
        )
        return embedding, artifacts, {}

    if args.task_kind == "joint":
        embedding, artifacts = compute_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            random_state=int(args.seed),
            perplexity=float(args.perplexity),
            backend=str(args.backend),
            n_neighbors=args.n_neighbors,
            n_jobs=int(args.n_jobs),
            pca_components=int(args.pca_components),
            n_iter=int(args.n_epochs),
            early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
            return_artifacts=True,
            timer=timer,
        )
        return embedding, artifacts, {}

    if args.task_kind in ("cca-2d", "cca-2d-uncentered"):
        embedding, artifacts = compute_cca_embedding_from_arrays(
            data.x0,
            data.x1,
            embd_dim=int(args.embedding_dim),
            return_artifacts=True,
            center=args.task_kind == "cca-2d",
        )
        return embedding, artifacts, {}

    if args.task_kind in ("cca-50", "cca-50-uncentered"):
        embedding, artifacts = compute_cca_then_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            cca_dim=50,
            random_state=int(args.seed),
            perplexity=float(args.perplexity),
            backend=str(args.backend),
            n_neighbors=args.n_neighbors,
            n_jobs=int(args.n_jobs),
            n_iter=int(args.n_epochs),
            early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
            return_artifacts=True,
            center=args.task_kind == "cca-50",
            timer=timer,
        )
        return embedding, artifacts, {"cca_dim": 50}

    if args.task_kind == "harmony-2d":
        embedding, artifacts = compute_harmony_embedding_from_arrays(
            data.x0,
            data.x1,
            embd_dim=2,
            random_state=int(args.seed),
            input_is_pca=True,
            return_artifacts=True,
        )
        return embedding, artifacts, {"harmony_dim": 2}

    if args.task_kind == "harmony-50":
        embedding, artifacts = compute_harmony_then_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            harmony_dim=50,
            random_state=int(args.seed),
            input_is_pca=True,
            perplexity=float(args.perplexity),
            backend=str(args.backend),
            n_neighbors=args.n_neighbors,
            n_jobs=int(args.n_jobs),
            n_iter=int(args.n_epochs),
            early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
            return_artifacts=True,
            timer=timer,
        )
        return embedding, artifacts, {"harmony_dim": 50}

    if args.task_kind == "scvi-2d":
        embedding, artifacts = compute_scvi_embedding_from_arrays(
            data.x0,
            data.x1,
            latent_dim=2,
            random_state=int(args.seed),
            max_epochs=DEFAULT_SCVI_MAX_EPOCHS,
            batch_size=DEFAULT_SCVI_BATCH_SIZE,
            train_kwargs={"enable_progress_bar": False},
            return_artifacts=True,
            require_cpu=bool(getattr(args, "require_cpu", False)),
            threads=getattr(args, "threads", None),
            timer=timer,
        )
        return embedding, artifacts, {"scvi_latent_dim": 2}

    if args.task_kind == "scvi-50":
        embedding, artifacts = compute_scvi_then_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            latent_dim=50,
            random_state=int(args.seed),
            max_epochs=DEFAULT_SCVI_MAX_EPOCHS,
            batch_size=DEFAULT_SCVI_BATCH_SIZE,
            perplexity=float(args.perplexity),
            backend=str(args.backend),
            n_neighbors=args.n_neighbors,
            n_jobs=int(args.n_jobs),
            n_iter=int(args.n_epochs),
            early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
            train_kwargs={"enable_progress_bar": False},
            return_artifacts=True,
            require_cpu=bool(getattr(args, "require_cpu", False)),
            threads=getattr(args, "threads", None),
            timer=timer,
        )
        return embedding, artifacts, {"scvi_latent_dim": 50}

    if args.task_kind == "ref-our":
        embedding, artifacts = compute_ref_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            random_state=int(args.seed),
            perplexity=float(args.perplexity),
            n_jobs=int(args.n_jobs),
            pca_components=int(args.pca_components),
            n_iter=int(args.n_epochs),
            early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
            backend=str(args.backend),
            n_neighbors=args.n_neighbors,
            return_artifacts=True,
            timer=timer,
        )
        return embedding, artifacts, {}

    if args.task_kind == "ref":
        embedding, artifacts = compute_policar_ref_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            random_state=int(args.seed),
            n_jobs=int(args.n_jobs),
            preselected_features=DEFAULT_REF_TSNE_PRESELECTED_FEATURES,
            return_artifacts=True,
            timer=timer,
            # Only the reference batch is normalised; query mapping uses cosine on raw counts.
            reference_library_sizes=(
                None if library_sizes is None else library_sizes[: data.x0.shape[0]]
            ),
        )
        return embedding, artifacts, {
            "preselected_features": DEFAULT_REF_TSNE_PRESELECTED_FEATURES,
            "count_library_size": _count_library_size(args),
        }

    if args.task_kind == "scanorama-2d":
        embedding, artifacts = compute_scanorama_embedding_from_arrays(
            data.x0,
            data.x1,
            embd_dim=2,
            random_state=int(args.seed),
            return_artifacts=True,
        )
        return embedding, artifacts, {"scanorama_dim": 2}

    if args.task_kind == "scanorama-50":
        embedding, artifacts = compute_scanorama_then_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            scanorama_dim=50,
            random_state=int(args.seed),
            perplexity=float(args.perplexity),
            backend=str(args.backend),
            n_neighbors=args.n_neighbors,
            n_jobs=int(args.n_jobs),
            n_iter=int(args.n_epochs),
            early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
            return_artifacts=True,
            timer=timer,
        )
        return embedding, artifacts, {"scanorama_dim": 50}

    if args.task_kind in R_METHODS:
        from ne_aligned_utils.methods.external.r import (
            compute_bc_tsne_embedding_from_arrays,
            compute_liger1_embedding_from_arrays,
            compute_liger1_then_tsne_embedding_from_arrays,
            compute_liger_embedding_from_arrays,
            compute_liger_then_tsne_embedding_from_arrays,
            compute_seurat3_embedding_from_arrays,
            compute_seurat3_then_tsne_embedding_from_arrays,
            compute_seurat_embedding_from_arrays,
            compute_seurat_then_tsne_embedding_from_arrays,
        )

        if args.task_kind == "seurat-2d":
            embedding, artifacts = compute_seurat_embedding_from_arrays(
                data.x0,
                data.x1,
                embd_dim=2,
                random_state=int(args.seed),
                preselected_features=DEFAULT_SEURAT_PRESELECTED_FEATURES,
                workflow=DEFAULT_SEURAT_WORKFLOW,
                return_artifacts=True,
                timer=timer,
                library_sizes=library_sizes,
            )
            return embedding, artifacts, {
                "seurat_dim": 2,
                "count_library_size": _count_library_size(args),
            }
        if args.task_kind == "seurat-50":
            embedding, artifacts = compute_seurat_then_tsne_embedding_from_arrays(
                data.x0,
                data.x1,
                seurat_dim=50,
                random_state=int(args.seed),
                perplexity=float(args.perplexity),
                backend=str(args.backend),
                n_neighbors=args.n_neighbors,
                n_jobs=int(args.n_jobs),
                n_iter=int(args.n_epochs),
                early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
                preselected_features=DEFAULT_SEURAT_PRESELECTED_FEATURES,
                workflow=DEFAULT_SEURAT_WORKFLOW,
                return_artifacts=True,
                timer=timer,
                library_sizes=library_sizes,
            )
            return embedding, artifacts, {
                "seurat_dim": 50,
                "count_library_size": _count_library_size(args),
            }
        if args.task_kind == "seurat3-2d":
            embedding, artifacts = compute_seurat3_embedding_from_arrays(
                data.x0,
                data.x1,
                embd_dim=2,
                random_state=int(args.seed),
                return_artifacts=True,
                timer=timer,
            )
            return embedding, artifacts, {"seurat3_dim": 2}
        if args.task_kind == "seurat3-50":
            embedding, artifacts = compute_seurat3_then_tsne_embedding_from_arrays(
                data.x0,
                data.x1,
                seurat_dim=50,
                random_state=int(args.seed),
                perplexity=float(args.perplexity),
                backend=str(args.backend),
                n_neighbors=args.n_neighbors,
                n_jobs=int(args.n_jobs),
                n_iter=int(args.n_epochs),
                early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
                return_artifacts=True,
                timer=timer,
            )
            return embedding, artifacts, {"seurat3_dim": 50}
        if args.task_kind == "liger1-2d":
            embedding, artifacts = compute_liger1_embedding_from_arrays(
                data.x0,
                data.x1,
                embd_dim=2,
                random_state=int(args.seed),
                couple_quantile_seed=True,
                return_artifacts=True,
                timer=timer,
            )
            return embedding, artifacts, {"liger1_k": 2}
        if args.task_kind == "liger1-50":
            embedding, artifacts = compute_liger1_then_tsne_embedding_from_arrays(
                data.x0,
                data.x1,
                liger_dim=50,
                random_state=int(args.seed),
                perplexity=float(args.perplexity),
                backend=str(args.backend),
                n_neighbors=args.n_neighbors,
                n_jobs=int(args.n_jobs),
                n_iter=int(args.n_epochs),
                early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
                couple_quantile_seed=True,
                return_artifacts=True,
                timer=timer,
            )
            return embedding, artifacts, {"liger1_k": 50}
        if args.task_kind == "liger-2d":
            embedding, artifacts = compute_liger_embedding_from_arrays(
                data.x0,
                data.x1,
                embd_dim=2,
                random_state=int(args.seed),
                preselected_features=True,
                return_artifacts=True,
                timer=timer,
                library_sizes=library_sizes,
            )
            return embedding, artifacts, {
                "liger_k": 2,
                "count_library_size": _count_library_size(args),
            }
        if args.task_kind == "liger-50":
            embedding, artifacts = compute_liger_then_tsne_embedding_from_arrays(
                data.x0,
                data.x1,
                liger_dim=50,
                random_state=int(args.seed),
                perplexity=float(args.perplexity),
                backend=str(args.backend),
                n_neighbors=args.n_neighbors,
                n_jobs=int(args.n_jobs),
                n_iter=int(args.n_epochs),
                early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
                preselected_features=True,
                return_artifacts=True,
                timer=timer,
                library_sizes=library_sizes,
            )
            return embedding, artifacts, {
                "liger_k": 50,
                "count_library_size": _count_library_size(args),
            }
        if args.task_kind == "bc-tsne":
            embedding, artifacts = compute_bc_tsne_embedding_from_arrays(
                data.x0,
                data.x1,
                random_state=int(args.seed),
                perplexity=float(args.perplexity),
                return_artifacts=True,
                timer=timer,
            )
            return embedding, artifacts, {}

    if args.task_kind == "rc-tsne":
        # The Python driver invokes an external binary that produces the final layout.
        from ne_aligned_utils.methods.external.rctsne import compute_rc_tsne_embedding_from_arrays

        embedding, artifacts = compute_rc_tsne_embedding_from_arrays(
            data.x0,
            data.x1,
            random_state=int(args.seed),
            perplexity=float(args.perplexity),
            beta=float(args.rc_beta),
            pca_components=int(args.pca_components),
            n_jobs=int(args.n_jobs),
            return_artifacts=True,
        )
        return embedding, artifacts, {"rc_beta": float(args.rc_beta)}

    if args.task_kind == "multimap":
        from ne_aligned_utils.methods.external.multimap import (
            compute_multimap_embedding_from_arrays,
        )

        # Omit layout parameters for MultiMAP's native profile, including its dataset-size epoch
        # rule.
        native = bool(getattr(args, "multimap_native", False))
        embedding, artifacts = compute_multimap_embedding_from_arrays(
            data.x0,
            data.x1,
            embd_dim=2,
            random_state=int(args.seed),
            n_neighbors=None if native else int(args.n_neighbors),
            min_dist=None if native else DEFAULT_MULTIMAP_MIN_DIST,
            spread=None if native else DEFAULT_MULTIMAP_SPREAD,
            n_epochs=None if native else int(args.n_epochs),
            init=None if native else DEFAULT_MULTIMAP_INIT,
            use_native_scaling=DEFAULT_MULTIMAP_USE_NATIVE_SCALING,
            return_artifacts=True,
            timer=timer,
        )
        return embedding, artifacts, {"multimap_dim": 2, "multimap_native": native}

    raise ValueError(f"Unsupported task_kind={args.task_kind!r}.")


def _resource_metadata(
    args: argparse.Namespace,
    data: _BenchmarkData,
    directory: Path,
    threads: int | None,
) -> dict[str, Any]:
    """Record task resource settings and a fingerprint of detailed environment provenance.

    Task fields include thread allocation, worker count, BLAS, resolved device, and
    Slurm CPU allocation. Full provenance is stored once per content hash.
    """
    from ne_aligned_utils.timing import (
        environment_record,
        write_environment_record,
    )

    record = environment_record(threads)
    fingerprint = write_environment_record(directory, record)
    return {
        "threads_requested": -1 if threads is None else int(threads),
        "n_jobs": int(args.n_jobs),
        "blas_name": str(record["blas_name"]),
        # Use CPU unless method artifacts report a resolved device.
        "device": "cpu",
        "slurm_cpus_per_task": str(record["slurm_cpus_per_task"]),
        "env_fingerprint": fingerprint,
        "n_obs": int(data.x0.shape[0] + data.x1.shape[0]),
        "n_batches": 2,
        "n_features_input": int(data.x0.shape[1]),
    }


def _cached_task_mismatches(payload: Any, args: argparse.Namespace) -> list[str]:
    """Return identity/content mismatches that make a cached task unsafe to resume."""
    expected: dict[str, Any] = {
        "task_kind": args.task_kind,
        "seed": int(args.seed),
        "method_seed": int(args.seed if args.method_seed is None else args.method_seed),
        "visualization_seed": int(
            args.seed if args.visualization_seed is None else args.visualization_seed
        ),
        "input_source": input_kind_for_method(str(args.task_kind)),
    }
    for field in (
        "run_id",
        "preprocessing_hash",
        "sample_hash",
        "representation_family",
        "method_pipeline",
    ):
        value = getattr(args, field, None)
        if value is not None:
            expected[field] = str(value)
    mismatches: list[str] = []
    for field, value in expected.items():
        if field not in payload:
            mismatches.append(f"missing {field}")
            continue
        observed = np.asarray(payload[field]).item()
        if observed != value:
            mismatches.append(f"{field}={observed!r}, expected {value!r}")
    if "embedding" not in payload:
        mismatches.append("missing embedding")
    else:
        embedding = np.asarray(payload["embedding"])
        if embedding.ndim != 2 or embedding.shape[1] != int(args.embedding_dim):
            mismatches.append(
                f"embedding shape={embedding.shape}, expected (*, {int(args.embedding_dim)})"
            )
        elif not np.isfinite(embedding).all():
            mismatches.append("embedding contains non-finite values")
    mismatches.extend(_cached_setting_mismatches(payload, args))
    return mismatches


def _cached_setting_mismatches(payload: Any, args: argparse.Namespace) -> list[str]:
    """Scientific settings a cached task was computed with that differ from this run's.

    The pipeline's run id already covers every setting. A direct ``run-task`` without
    ``--run-id`` relies on this check, so a rerun with, say, another ``--rc-beta`` recomputes
    instead of keeping the old file. Only settings the file records are compared, except where
    older files lack a setting but are known to have used a specific value.
    """
    kind = str(args.task_kind)
    backend = str(getattr(args, "backend", DEFAULT_BACKEND))
    expected: dict[str, Any] = {}
    # Only methods whose t-SNE (or UMAP) layout we optimise use these; the other direct 2D
    # methods record some of them but ignore them, so a change must not force a recompute.
    if kind in VISUALIZED_METHODS:
        expected["n_epochs"] = int(args.n_epochs)
        expected["tsne_early_exaggeration_iter"] = _npz_value(
            _tsne_early_exaggeration_iter(args)
        ).item()
    if run_uses_perplexity(kind, backend):
        expected["perplexity"] = float(args.perplexity)
    if kind == "rc-tsne":
        expected["rc_beta"] = float(args.rc_beta)
    if kind in COUNT_NORMALIZED_METHODS:
        expected["count_library_size"] = _count_library_size(args)
    if kind in UOT_METHODS:
        expected["rel_reg_m"] = float(getattr(args, "rel_reg_m", DEFAULT_REL_REG_M))
    if kind in SYMMETRIZATION_METHODS:
        expected["graph_symmetrization"] = str(
            getattr(args, "graph_symmetrization", DEFAULT_SYMMETRIZATION)
        )
    if kind in KNNDIAG_SYMMETRIZATION_METHODS:
        expected["knndiag_symmetrization"] = str(
            getattr(args, "knndiag_symmetrization", DEFAULT_SYMMETRIZATION)
        )
    if kind in OT_PLAN_METHODS:
        expected["ot_device"] = str(getattr(args, "ot_device", "cpu"))
    # Values files written before a setting was recorded must have used.
    legacy = _legacy_watched_values(payload)
    if kind in COUNT_NORMALIZED_METHODS:
        legacy["count_library_size"] = LEGACY_COUNT_LIBRARY_SIZE

    mismatches: list[str] = []
    for field, value in expected.items():
        if field in payload:
            observed = np.asarray(payload[field]).item()
        elif field in legacy:
            observed = legacy[field]
        else:
            continue
        if isinstance(value, float):
            same = np.isclose(float(observed), value, rtol=1e-12, atol=0.0)
        elif isinstance(value, int):
            same = int(observed) == value
        else:
            same = str(observed) == value
        if not same:
            mismatches.append(f"{field}={observed!r}, expected {value!r}")
    return mismatches


def run_task(args: argparse.Namespace) -> None:
    output_path = Path(args.task_output).expanduser()
    backend_requested = str(getattr(args, "backend", DEFAULT_BACKEND))
    # Allow na for direct layouts; reject a replaceable backend that would mislabel them.
    if (
        backend_requested in BACKENDS
        and backend_requested != DEFAULT_BACKEND
        and not has_backend_axis(args.task_kind)
    ):
        raise ValueError(
            f"{args.task_kind!r} has no embedding backend to swap, so there is nothing to compute "
            f"under {backend_requested!r} -- a run would reproduce the {DEFAULT_BACKEND} file byte "
            "for byte. Write it once through a repository --tree run, which routes it to the "
            "canonical shared results tree."
        )
    assert_output_backend_matches(output_path, backend_requested)
    assert_no_parameter_collision(output_path, args)
    if output_path.exists() and not args.force:
        try:
            with np.load(output_path, allow_pickle=False) as cached:
                mismatches = _cached_task_mismatches(cached, args)
        except (OSError, KeyError, ValueError):
            mismatches = ["archive is unreadable"]
        if not mismatches:
            print(f"Task output already exists at {output_path}; use --force to recompute.")
            return
        print(
            f"Task output at {output_path} is incompatible ({'; '.join(mismatches)}); "
            "recomputing it."
        )

    if getattr(args, "reuse_from", None) is not None:
        print(
            "run-task: --reuse-from loads a cached representation, so this run measures only the "
            "final embedding step. It forfeits time_total (recorded as NaN) and is excluded from "
            "every primary runtime table. Drop --reuse-from for a run whose total runtime is to "
            "be reported."
        )

    data = _load_benchmark_data(
        args.dataset,
        data_root=_resolve_data_root(args.data_root),
        seed=int(args.seed if getattr(args, "sample_seed", None) is None else args.sample_seed),
        max_cells_per_batch=args.max_cells_per_batch,
        max_cells_batch0=args.max_cells_batch0,
        max_cells_batch1=args.max_cells_batch1,
        input_kind=input_kind_for_method(args.task_kind),
    )
    from ne_aligned_utils.methods import embed_2d, embedder_artifacts
    from ne_aligned_utils.timing import (
        StageTimer,
        finalise,
        instrument_embedders,
        resolve_schema,
        warmup,
    )

    schema = resolve_schema(args.task_kind)
    timer = StageTimer()
    threads = getattr(args, "threads", None)
    # Exclude lazy imports and JIT compilation from algorithm time; report them as time_warmup.
    warmup_seconds = warmup(str(args.task_kind), str(getattr(args, "backend", DEFAULT_BACKEND)))
    # Loading the processed view for library sizes is data loading, not algorithm time.
    library_sizes = (
        _count_library_sizes(args, data) if args.task_kind in COUNT_NORMALIZED_METHODS else None
    )

    started = time.perf_counter()
    reuse_info: dict[str, object] = {}
    if getattr(args, "reuse_from", None) is not None:
        representation, source, digest = load_cached_representation(args)
        if representation.shape[0] != data.x0.shape[0] + data.x1.shape[0]:
            raise ValueError(
                f"cached representation has {representation.shape[0]} rows but this task's "
                "data has "
                f"{data.x0.shape[0] + data.x1.shape[0]}; the cache is for a different cell subset."
            )
        with instrument_embedders(timer):
            embedding, embed_artifacts = embed_2d(
                representation,
                str(args.backend),
                random_state=int(args.seed),
                perplexity=float(args.perplexity),
                n_neighbors=args.n_neighbors,
                n_jobs=int(args.n_jobs),
                n_iter=int(args.n_epochs),
                early_exaggeration_iter=_tsne_early_exaggeration_iter(args),
                return_artifacts=True,
                timer=timer,
            )
        n0 = data.x0.shape[0]
        artifacts = {
            "left_embedding": embedding[:n0],
            "right_embedding": embedding[n0:],
            **embedder_artifacts(str(args.backend), embed_artifacts),
            "random_state": int(args.seed),
            "n_jobs": int(args.n_jobs),
        }
        # Hash reused array contents to identify the exact representation independently of its path.
        reuse_info = {
            "representation_reused": True,
            "representation_source": str(source),
            "representation_sha256": digest,
            "representation_shape": np.asarray(representation.shape),
        }
        method_params = {}
    else:
        with instrument_embedders(timer):
            embedding, artifacts, method_params = _compute_task(
                data, args, timer=timer, library_sizes=library_sizes
            )
        reuse_info = {"representation_reused": False}
    # Measure the method call without loading or persistence. finalise resolves subprocess overhead.
    wall_outer = time.perf_counter() - started

    payload = common_metadata(data, args, task_kind=args.task_kind)
    if (
        str(getattr(args, "graph_symmetrization", DEFAULT_SYMMETRIZATION)) == "and"
        and "graph" in artifacts
    ):
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components

        graph = artifacts["graph"]
        support = sparse.csr_matrix(graph != 0)
        support.setdiag(0)
        support.eliminate_zeros()
        degrees = np.diff(support.indptr).astype(int)
        directed_budget = support.shape[0] * int(getattr(args, "effective_k", 0) or 0)
        payload.update(
            {
                "mutual_nonempty_row_fraction": float(np.mean(degrees > 0)),
                "mutual_retained_edge_fraction": (
                    float(support.nnz / directed_budget) if directed_budget else float("nan")
                ),
                "mutual_degree_distribution": degrees,
                "mutual_total_edge_count": int(sparse.triu(support, k=1).nnz),
                "mutual_connected_components": int(
                    connected_components(support, directed=False, return_labels=False)
                ),
                "mutual_retention_denominator": int(directed_budget),
            }
        )
    payload.update(
        {key: value for key, value in artifacts.items() if key not in UNPERSISTED_ARTIFACT_KEYS}
    )
    payload.update(method_params)
    # Apply task provenance last so method artifacts cannot overwrite it.
    payload.update(reuse_info)
    payload["embedding"] = np.asarray(embedding, dtype=float)
    # Keep total_seconds as outer wall time; time_total records canonical algorithm time.
    payload["total_seconds"] = float(wall_outer)
    payload["timing_schema"] = schema
    payload.update(
        finalise(
            timer,
            schema=schema,
            wall_outer=wall_outer,
            task_kind=str(args.task_kind),
            representation_reused=bool(reuse_info.get("representation_reused", False)),
            warmup=None if np.isnan(warmup_seconds) else warmup_seconds,
        )
    )
    payload.update(_resource_metadata(args, data, output_path.parent, threads))
    if "torch_device" in artifacts:
        # Record the resolved device for scVI and OT, rather than the requested default.
        payload["device"] = str(artifacts["torch_device"])
    save_npz(output_path, **payload)
    print(f"Saved {args.task_kind} embedding to {output_path}")



__all__ = [
    "UNPERSISTED_ARTIFACT_KEYS",
    "_compute_task",
    "common_metadata",
    "load_cached_representation",
    "run_task",
    "save_npz",
]
