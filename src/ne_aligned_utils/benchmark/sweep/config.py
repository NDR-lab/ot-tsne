"""Sweep defaults, naming, output paths, and collision validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ne_aligned_utils.benchmark.sweep.registry import (
    EPS_K_METHODS,
    KNNDIAG_SYMMETRIZATION_METHODS,
    OT_PLAN_METHODS,
    SYMMETRIZATION_METHODS,
    TASK_KINDS,
    UOT_METHODS,
    VISUALIZED_METHODS,
)
from ne_aligned_utils.config import (
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_MULTIMAP_INIT,
    DEFAULT_MULTIMAP_MIN_DIST,
    DEFAULT_MULTIMAP_N_EPOCHS,
    DEFAULT_MULTIMAP_SPREAD,
    DEFAULT_PERPLEXITY as _DEFAULT_PERPLEXITY,
    DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    DEFAULT_TSNE_N_ITER,
    DEFAULT_UMAP_N_NEIGHBORS,
    METHOD_SPECS,
    NO_BACKEND,
    ROOT,
    method_stem,
)

DEFAULT_SYMMETRIZATION = "or"
SYMMETRIZATIONS = ("or", "and")
DEFAULT_REL_REG_M = float("inf")
DEFAULT_REL_REG = (1.0,)
DEFAULT_EFFECTIVE_KS = (DEFAULT_UMAP_N_NEIGHBORS,)
DEFAULT_EMBEDDING_DIM = 2
DEFAULT_N_EPOCHS = DEFAULT_TSNE_N_ITER
# Divisor in the count normalisation of seurat-*, liger-* and ref: each cell's total over all
# genes, as in the processed features (default), or over the supplied genes only (earlier runs).
COUNT_LIBRARY_SIZES = ("selected_genes", "all_genes")
DEFAULT_COUNT_LIBRARY_SIZE = "all_genes"
# Manifests written before the setting existed ran with this divisor.
LEGACY_COUNT_LIBRARY_SIZE = "selected_genes"
COUNT_NORMALIZED_METHODS = ("seurat-2d", "seurat-50", "liger-2d", "liger-50", "ref")
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_PERPLEXITY = _DEFAULT_PERPLEXITY
DEFAULT_TOL = 1e-12
DEFAULT_MAX_ITER = int(1e4)

_EPS_ONLY_FILENAME_METHODS = ("eot-t0", "eot-50")


def format_token(value: float | int | str | None) -> str:
    text = "none" if value is None else str(value)
    text = text.replace("-", "m").replace(".", "p")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in text)


def format_rel_reg(value: float) -> str:
    """Return the shortest decimal string that round-trips the regularization value.

    Full float precision matters for argv serialization and equality-based grid filters,
    including values such as 1/300.
    """
    return repr(float(value))


def resolve_output_base(output_dir: Path | None, dataset: str) -> Path:
    if output_dir is None:
        return ROOT / "data" / "benchmark_embeddings" / dataset
    path = Path(output_dir).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _resolved_n_neighbors(args: argparse.Namespace) -> int:
    """Return the neighborhood used by UMAP or MultiMAP, otherwise ``-1``."""
    uses_umap_neighbourhood = (
        str(getattr(args, "backend", DEFAULT_BACKEND)) == "umap"
        or getattr(args, "task_kind", None) == "multimap"
        or "multimap" in set(getattr(args, "methods", ()))
        or "multimap" in Path(getattr(args, "output_dir", "") or "").parts
    )
    if not uses_umap_neighbourhood:
        return -1
    requested = getattr(args, "n_neighbors", None)
    if requested is not None:
        return int(requested)
    return DEFAULT_UMAP_N_NEIGHBORS


def config_token(args: argparse.Namespace) -> str:
    max_cells = getattr(args, "max_cells_per_batch", None)
    max_cells0 = getattr(args, "max_cells_batch0", None)
    max_cells1 = getattr(args, "max_cells_batch1", None)
    if max_cells0 is not None or max_cells1 is not None:
        cap0 = max_cells if max_cells0 is None else max_cells0
        cap1 = max_cells if max_cells1 is None else max_cells1
        n_token = f"{'all' if cap0 is None else int(cap0)}_{'all' if cap1 is None else int(cap1)}"
    else:
        n_token = "all" if max_cells is None else str(int(max_cells))

    output_parts = set(Path(getattr(args, "output_dir", "") or "").parts)
    if "2d_outputs" in output_parts or "rctsne" in output_parts:
        return "__".join(
            [
                f"n_per_batch_{format_token(n_token)}",
                f"q_{format_token(getattr(args, 'embedding_dim', DEFAULT_EMBEDDING_DIM))}",
                "nn_na",
            ]
        )
    if "multimap" in output_parts:
        if bool(getattr(args, "multimap_native", False)):
            derived_n_neighbors = DEFAULT_UMAP_N_NEIGHBORS * 2
            return "__".join(
                [
                    f"n_per_batch_{format_token(n_token)}",
                    f"q_{format_token(getattr(args, 'embedding_dim', DEFAULT_EMBEDDING_DIM))}",
                    f"nn_{format_token(derived_n_neighbors)}",
                ]
            )
        return "__".join(
            [
                f"n_per_batch_{format_token(n_token)}",
                f"q_{format_token(getattr(args, 'embedding_dim', DEFAULT_EMBEDDING_DIM))}",
                f"nn_{format_token(_resolved_n_neighbors(args))}",
                f"min_dist_{format_token(DEFAULT_MULTIMAP_MIN_DIST)}",
                f"spread_{format_token(DEFAULT_MULTIMAP_SPREAD)}",
                f"epochs_{format_token(getattr(args, 'n_epochs', DEFAULT_MULTIMAP_N_EPOCHS))}",
                f"init_{DEFAULT_MULTIMAP_INIT}",
            ]
        )

    backend = str(getattr(args, "backend", DEFAULT_BACKEND))
    nn_token = "na" if backend == DEFAULT_BACKEND else format_token(_resolved_n_neighbors(args))
    return "__".join(
        [
            f"n_per_batch_{format_token(n_token)}",
            f"q_{format_token(getattr(args, 'embedding_dim', DEFAULT_EMBEDDING_DIM))}",
            f"epochs_{format_token(getattr(args, 'n_epochs', DEFAULT_N_EPOCHS))}",
            f"pca_{format_token(getattr(args, 'pca_components', DEFAULT_PCA_COMPONENTS))}",
            f"nn_{nn_token}",
        ]
    )


def resolve_task_root(args: argparse.Namespace) -> Path:
    return resolve_output_base(getattr(args, "output_dir", None), args.dataset) / config_token(args)


def assert_no_parameter_collision(output_path: Path, args: argparse.Namespace) -> None:
    """Reject incompatible task metadata in a shared output directory."""
    directory = output_path.parent
    if not directory.is_dir():
        return
    watched = {
        "perplexity": float(args.perplexity),
        "n_neighbors": _resolved_n_neighbors(args),
        "backend": str(getattr(args, "backend", DEFAULT_BACKEND)),
    }
    if str(getattr(args, "task_kind", "")) in OT_PLAN_METHODS:
        watched["device"] = str(getattr(args, "ot_device", "cpu"))
    if str(getattr(args, "task_kind", "")) in UOT_METHODS:
        watched["rel_reg_m"] = float(getattr(args, "rel_reg_m", DEFAULT_REL_REG_M))
    if str(getattr(args, "task_kind", "")) in SYMMETRIZATION_METHODS:
        watched["graph_symmetrization"] = str(
            getattr(args, "graph_symmetrization", DEFAULT_SYMMETRIZATION)
        )
    if str(getattr(args, "task_kind", "")) in KNNDIAG_SYMMETRIZATION_METHODS:
        watched["knndiag_symmetrization"] = str(
            getattr(args, "knndiag_symmetrization", DEFAULT_SYMMETRIZATION)
        )
    kind = str(getattr(args, "task_kind", ""))
    if kind in METHOD_SPECS:
        if not METHOD_SPECS[kind].backend_axis:
            watched.pop("backend")
        if not METHOD_SPECS[kind].uses_perplexity:
            watched.pop("perplexity")
    if kind in VISUALIZED_METHODS and str(getattr(args, "backend", DEFAULT_BACKEND)) == "tsne":
        watched["tsne_early_exaggeration_iter"] = int(
            getattr(args, "tsne_early_exaggeration_iter", DEFAULT_TSNE_EARLY_EXAGGERATION_ITER)
        )
    # Every sibling is checked, not only the first readable one.
    for sibling in sorted(directory.glob("*.npz")):
        if sibling == output_path or sibling.name.endswith("_benchmark_embeddings.npz"):
            continue
        try:
            with np.load(sibling, allow_pickle=True) as cached:
                legacy = _legacy_watched_values(cached)
                for key, value in watched.items():
                    if key not in cached.files:
                        if key not in legacy:
                            continue
                        other = legacy[key]
                    else:
                        other = cached[key].item() if cached[key].ndim == 0 else cached[key]
                    if key == "backend" and str(other) == NO_BACKEND:
                        # A direct 2D layout has no embedding backend to collide with.
                        continue
                    if isinstance(value, float):
                        matches = float(other) == value
                    elif isinstance(value, int):
                        matches = int(other) == value
                    else:
                        matches = str(other) == value
                    if not matches:
                        raise ValueError(
                            f"{directory} already holds {sibling.name} with {key}={other!r}, but "
                            f"this run has {key}={value!r}. The config token does not separate "
                            f"them, so one would overwrite the other. Use a different --tree, or "
                            f"add {key} to config_token."
                        )
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and "already holds" in str(exc):
                raise
            continue


def _legacy_watched_values(cached) -> dict[str, object]:
    """Values a sibling written before a setting was recorded must have used.

    Such files used openTSNE's 250 exaggerated iterations in the 50D tails, but a first-third
    split of ``n_epochs`` in OT layouts (recorded as -1). Which applies depends on the sibling's
    own method, not on the task now being written.
    """
    if "task_kind" not in cached:
        return {}
    kind = str(cached["task_kind"].item())
    return {
        "tsne_early_exaggeration_iter": (
            -1 if kind in OT_PLAN_METHODS else DEFAULT_TSNE_EARLY_EXAGGERATION_ITER
        )
    }


def assert_output_backend_matches(output_path: Path, backend: str) -> None:
    """Reject an explicit output directory belonging to another backend."""
    parts = set(output_path.resolve().parts)
    named = parts & set(BACKENDS)
    if not named or backend in named:
        return
    raise ValueError(
        f"backend={backend!r} would write into {sorted(named)[0]!r}'s tree: {output_path}. "
        f"Pass --output-dir under results/{backend}/ (or use --tree, which derives it)."
    )


def resolve_task_output_path(
    task_root: Path,
    *,
    task_kind: str,
    seed: int,
    rel_reg: float | None = None,
    effective_k: int | None = None,
    backend: str = DEFAULT_BACKEND,
) -> Path:
    """Return the task path for one sweep-grid point."""
    seed_token = format_token(seed)
    stem = method_stem(task_kind, backend)
    if task_kind in EPS_K_METHODS:
        return (
            task_root / f"{stem}_seed_{seed_token}_rel_reg_{format_token(format_rel_reg(rel_reg))}"
            f"_k_{format_token(effective_k)}.npz"
        )
    if task_kind in _EPS_ONLY_FILENAME_METHODS:
        return (
            task_root
            / f"{stem}_seed_{seed_token}_rel_reg_{format_token(format_rel_reg(rel_reg))}.npz"
        )
    if task_kind in TASK_KINDS:
        return task_root / f"{stem}_seed_{seed_token}.npz"
    raise ValueError(f"Unsupported task_kind={task_kind!r}.")


def resolve_aggregate_output_path(task_root: Path, dataset: str) -> Path:
    return task_root / f"{format_token(dataset)}_benchmark_embeddings.npz"


def __getattr__(name: str):
    """Retain the former ``config.validate_args`` compatibility export lazily."""
    if name == "validate_args":
        from ne_aligned_utils.benchmark.sweep.core import validate_args

        return validate_args
    raise AttributeError(name)


__all__ = [
    "DEFAULT_EFFECTIVE_KS",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_MAX_ITER",
    "DEFAULT_N_EPOCHS",
    "DEFAULT_PCA_COMPONENTS",
    "DEFAULT_PERPLEXITY",
    "DEFAULT_REL_REG",
    "DEFAULT_REL_REG_M",
    "DEFAULT_SYMMETRIZATION",
    "DEFAULT_TOL",
    "SYMMETRIZATIONS",
    "assert_no_parameter_collision",
    "assert_output_backend_matches",
    "config_token",
    "format_rel_reg",
    "format_token",
    "resolve_aggregate_output_path",
    "resolve_output_base",
    "resolve_task_output_path",
    "resolve_task_root",
    "validate_args",  # noqa: F822 - provided lazily by __getattr__
]
