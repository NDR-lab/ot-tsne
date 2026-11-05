"""Read saved task timing fields and require canonical algorithm totals.

It reads task files only; it never executes a benchmark method.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Fields that must agree when comparing runtime allocations.
RESOURCE_KEYS = ("threads_requested", "blas_name", "device")

RUNTIME_STAGE_COLUMNS: tuple[str, ...] = (
    # The canonical algorithm runtime. This is the field a runtime figure uses.
    "time_total",
    "time_pre_embedding",
    "time_embedding",
    "time_ot_cost",
    "time_ot_solve",
    "time_ot_postprocess",
    "time_ot_solve_self",
    "time_embedding_graph",
    "time_embedding_initialization",
    "time_embedding_optimization",
    "time_embedding_reference",
    "time_embedding_query",
    # Diagnostics. Not algorithm time; never summed into a total.
    "time_wall_outer",
    "time_excluded_overhead",
    "time_warmup",
)

RUNTIME_RESOURCE_COLUMNS: tuple[str, ...] = (
    "timing_schema",
    "threads_requested",
    "n_jobs",
    "blas_name",
    "device",
    "slurm_cpus_per_task",
    "env_fingerprint",
    "n_obs",
    "n_batches",
    "n_features_input",
)


class RuntimeComparabilityError(RuntimeError):
    """Raised when the selected rows' times do not measure the same amount of work."""


def require_canonical_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Refuse a frame that cannot supply ``time_total``.

    The legacy ``seconds`` column is always present, so a missing-column check is not enough: the
    failure mode to prevent is reading ``seconds`` *because* ``time_total`` was all NaN, which would
    silently mix two different definitions of "total" on one axis.
    """
    if "time_total" not in frame.columns:
        raise RuntimeComparabilityError(
            "this frame has no 'time_total' column, only the legacy 'seconds'. The runtime "
            "appendix uses the canonical algorithm runtime; re-run the sweep so the task files "
            "carry it, rather than falling back to 'seconds', which for the subprocess methods "
            "also contains interpreter startup and library import."
        )
    if not frame["time_total"].notna().any():
        raise RuntimeComparabilityError(
            "every row's 'time_total' is NaN, so these task files predate the timing schema (or "
            "are all --reuse-from runs). There is nothing here to report."
        )
    return frame


def scan_task_tree(root: Path) -> pd.DataFrame:
    """Read saved task fields, skipping aggregate embedding caches. Runs nothing."""
    root = Path(root)
    paths = sorted(
        path for path in root.rglob("*.npz") if not path.name.endswith("_benchmark_embeddings.npz")
    )
    if not paths:
        raise FileNotFoundError(f"no task .npz files under {root}")

    wanted = (
        "dataset",
        "task_kind",
        "method",
        "backend",
        "seed",
        "rel_reg",
        "effective_k",
        "perplexity",
        "n_epochs",
        "representation_reused",
        # Use multimap_native to distinguish profiles; stored n_neighbors alone is ambiguous.
        "multimap_native",
        *RUNTIME_STAGE_COLUMNS,
        *RUNTIME_RESOURCE_COLUMNS,
    )
    rows: list[dict[str, object]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as cache:
            row: dict[str, object] = {"task_path": str(path)}
            for name in wanted:
                if name not in cache.files:
                    row[name] = np.nan
                    continue
                value = np.asarray(cache[name])
                row[name] = value.item() if value.ndim == 0 else value.tolist()
            sizes = (
                np.asarray(cache["batch_sizes"], dtype=int)
                if "batch_sizes" in cache.files
                else None
            )
            row["n_cells"] = int(sizes.sum()) if sizes is not None else np.nan
            rows.append(row)

    frame = pd.DataFrame(rows)
    # Group on the backend-neutral task_kind; backend remains a separate column.
    frame["method_key"] = frame["task_kind"]
    frame["representation_reused"] = frame["representation_reused"].fillna(False).astype(bool)
    return require_canonical_totals(frame)
