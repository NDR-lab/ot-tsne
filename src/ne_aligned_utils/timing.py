"""Benchmark stage timing and comparable resource metadata.

``time_total`` excludes data loading, persistence, reporting, and scheduler overhead. Optional
stage fields measure natural method boundaries; they are not required to sum exactly. Missing
stages use NaN rather than zero. Temporary third-party instrumentation restores original callables
after each task.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

#: The canonical total, on every task.
TOTAL_FIELD = "time_total"

#: The coarse split, where it is meaningful.
STAGE_FIELDS: tuple[str, ...] = ("time_pre_embedding", "time_embedding")

#: The OT decomposition, nested inside ``time_pre_embedding``.
OT_FIELDS: tuple[str, ...] = ("time_ot_cost", "time_ot_solve", "time_ot_postprocess")

#: Optional diagnostics nested inside ``time_embedding``. Backend-neutral names: ``graph`` is
#: openTSNE's affinity build or UMAP's fuzzy simplicial set, whichever ran.
EMBEDDING_DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "time_embedding_graph",
    "time_embedding_initialization",
    "time_embedding_optimization",
)

#: Additional timings: self-OT cost/solve and full reference/query embedding calls.
#: Reference/query fields overlap the embedding diagnostics and are not additive to them.
EXTRA_FIELDS: tuple[str, ...] = (
    "time_ot_solve_self",
    "time_embedding_reference",
    "time_embedding_query",
)

#: Diagnostics that are *not* algorithm time and must never be mixed into ``time_total``.
DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    # The outer `_compute_task` measurement. Equal to time_total for the in-process methods;
    # larger for the subprocess ones, where it also holds interpreter startup and IO.
    "time_wall_outer",
    # time_wall_outer - time_total, for the methods whose child reports its own duration.
    "time_excluded_overhead",
    # The deliberate JIT warm-up, run outside the timed region. See `warmup_backend`.
    "time_warmup",
)

#: Every timing field a task may carry, in a stable order.
ALL_TIMING_FIELDS: tuple[str, ...] = (
    TOTAL_FIELD,
    *STAGE_FIELDS,
    *OT_FIELDS,
    *EMBEDDING_DIAGNOSTIC_FIELDS,
    *EXTRA_FIELDS,
    *DIAGNOSTIC_FIELDS,
)

#: Allowed fields by schema; unavailable stages remain NaN. OT keeps its decomposition
#: even without a backend; two_stage splits integration/layout; fused reports only a total.
SCHEMA_FIELDS: dict[str, frozenset[str]] = {
    "ot": frozenset({*STAGE_FIELDS, *OT_FIELDS, *EMBEDDING_DIAGNOSTIC_FIELDS, *EXTRA_FIELDS}),
    "two_stage": frozenset({*STAGE_FIELDS, *EMBEDDING_DIAGNOSTIC_FIELDS, *EXTRA_FIELDS}),
    "fused": frozenset(),
}

#: Explicit task-kind timing schemas; unknown methods must not inherit a guessed schema.
TASK_SCHEMAS: dict[str, str] = {
    # -- OT plan solved, then a layout: OT-tSNE variants, and EOT (a separate method) ---------
    "ot": "ot",
    "ot-knndiag": "ot",
    "ot-selfotmedian": "ot",
    "ot-gramdiag": "ot",
    "ot-uot-topk": "ot",
    "ot-uot-mass": "ot",
    "ot-zerodiag": "ot",
    "ot-zerodiag-split": "ot",
    "eot-50": "ot",
    # Closed-form EOT retains OT sub-timings even though it has no backend.
    "eot-t0": "ot",
    # -- correction or integration, then a backend --------------------------------------------
    "cca-50": "two_stage",
    "cca-50-uncentered": "two_stage",
    "harmony-50": "two_stage",
    "scvi-50": "two_stage",
    "scanorama-50": "two_stage",
    "seurat-50": "two_stage",
    "seurat3-50": "two_stage",
    "liger1-50": "two_stage",
    "liger-50": "two_stage",
    "joint": "two_stage",
    "ref": "two_stage",
    "ref-our": "two_stage",
    # -- one inseparable computation ----------------------------------------------------------
    "bc-tsne": "fused",
    "rc-tsne": "fused",
    "multimap": "fused",
    "joint-pca": "fused",
    "cca-2d": "fused",
    "cca-2d-uncentered": "fused",
    "harmony-2d": "fused",
    "scanorama-2d": "fused",
    "scvi-2d": "fused",
    "seurat-2d": "fused",
    "seurat3-2d": "fused",
    "liger1-2d": "fused",
    "liger-2d": "fused",
}

#: Methods with child-reported algorithm totals. rc-tsne instead uses parent-measured
#: binary duration because the binary exposes no algorithm-only boundary.
CHILD_TIMED_METHODS: tuple[str, ...] = (
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


def resolve_schema(task_kind: str) -> str:
    """The timing schema for ``task_kind``; raises rather than guessing at an unknown name."""
    try:
        return TASK_SCHEMAS[str(task_kind)]
    except KeyError:
        raise KeyError(
            f"{task_kind!r} has no timing schema. Add it to TASK_SCHEMAS with the shape that "
            "actually describes it, rather than letting it default to one that does not."
        ) from None


@dataclass
class StageTimer:
    """Accumulates elapsed seconds per named stage.

    Additive per name: a stage entered more than once sums, which is what ``ot-selfotmedian``'s
    three Sinkhorn solves and ``ref``'s two optimisations need. Nesting is allowed and is not
    subtracted -- ``time_ot_solve`` sits inside ``time_pre_embedding`` and both are real
    measurements of real boundaries.
    """

    seconds: dict[str, float] = field(default_factory=dict)

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - start)

    def record(self, name: str, elapsed: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + float(elapsed)

    def set(self, name: str, elapsed: float) -> None:
        """Overwrite rather than accumulate. For a duration reported by a child process."""
        self.seconds[name] = float(elapsed)

    def get(self, name: str) -> float:
        return self.seconds.get(name, math.nan)


def finalise(
    timer: StageTimer,
    *,
    schema: str,
    wall_outer: float,
    task_kind: str,
    representation_reused: bool = False,
    warmup: float | None = None,
) -> dict[str, float]:
    """Resolve persisted stage timings and canonical algorithm duration.

    In-process totals use parent wall time; child-reported totals exclude startup and
    imports, with the difference recorded as overhead. rc-tsne uses binary-call wall
    time, including launch and file interchange. Reused representations invalidate
    total and pre-embedding durations but retain measured embedding time.
    """
    allowed = SCHEMA_FIELDS[schema]
    out: dict[str, float] = {name: math.nan for name in ALL_TIMING_FIELDS}

    for name in allowed:
        if name in timer.seconds:
            out[name] = float(timer.seconds[name])

    out["time_wall_outer"] = float(wall_outer)
    if warmup is not None:
        out["time_warmup"] = float(warmup)

    if str(task_kind) in CHILD_TIMED_METHODS:
        child_total = _child_total(timer, schema)
        if not math.isnan(child_total):
            out[TOTAL_FIELD] = child_total
            out["time_excluded_overhead"] = max(float(wall_outer) - child_total, 0.0)
        else:
            # If child timing is absent, retain parent wall time and leave excluded overhead NaN.
            out[TOTAL_FIELD] = float(wall_outer)
    else:
        out[TOTAL_FIELD] = float(wall_outer)

    if representation_reused:
        out[TOTAL_FIELD] = math.nan
        out["time_pre_embedding"] = math.nan

    return out


def _child_total(timer: StageTimer, schema: str) -> float:
    """Combine child-reported integration and Python embedding durations when available.

    Fused children report time_total directly; direct R layouts may report their
    whole duration as time_pre_embedding because they share an integration script.
    """
    if schema == "fused":
        total = timer.seconds.get(TOTAL_FIELD, math.nan)
        if not math.isnan(total):
            return total
        return timer.seconds.get("time_pre_embedding", math.nan)
    pre = timer.seconds.get("time_pre_embedding", math.nan)
    embed = timer.seconds.get("time_embedding", math.nan)
    if math.isnan(pre) or math.isnan(embed):
        return math.nan
    return float(pre) + float(embed)


# Instrumenting the embedding backends


@contextlib.contextmanager
def instrument_embedders(timer: StageTimer | None) -> Iterator[None]:
    """Temporarily time backend graph building, initialization, and optimization.

    Preserve callable signatures, arguments, and results, restoring originals on exit.
    A None timer disables instrumentation. Graph-based OT times its own preparation;
    backend bookkeeping remains covered by the surrounding time_embedding stage.
    """
    if timer is None:
        yield
        return

    restore: list[tuple[Any, str, Any]] = []

    # Preserve signatures: openTSNE inspects to_new for perplexity before allowing transform.
    # functools.wraps exposes the original signature through __wrapped__.
    def _wrap_function(owner: Any, name: str, stage: str) -> None:
        original = getattr(owner, name)

        @functools.wraps(original)
        def timed(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timer.record(stage, time.perf_counter() - start)

        setattr(owner, name, timed)
        restore.append((owner, name, original))

    def _wrap_method(owner: Any, name: str, stage: str) -> None:
        original = getattr(owner, name)

        @functools.wraps(original)
        def timed(self: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                timer.record(stage, time.perf_counter() - start)

        setattr(owner, name, timed)
        restore.append((owner, name, original))

    try:
        from openTSNE import affinity as _affinity, tsne as _tsne

        # `prepare_initial` builds a MultiscaleMixture, not a PerplexityBasedNN, so this is the
        # class that actually runs. `to_new` is the query-side affinity `ref` needs.
        _wrap_method(_affinity.MultiscaleMixture, "__init__", "time_embedding_graph")
        _wrap_method(_affinity.MultiscaleMixture, "to_new", "time_embedding_graph")
        _wrap_function(_tsne.initialization_scheme, "pca", "time_embedding_initialization")
        _wrap_function(_tsne.initialization_scheme, "spectral", "time_embedding_initialization")
        # Patch both joint and partial optimizers; query projection uses a separate class.
        _wrap_method(_tsne.TSNEEmbedding, "optimize", "time_embedding_optimization")
        _wrap_method(_tsne.PartialTSNEEmbedding, "optimize", "time_embedding_optimization")

        from umap import umap_ as _umap

        _wrap_function(_umap, "nearest_neighbors", "time_embedding_graph")
        _wrap_function(_umap, "fuzzy_simplicial_set", "time_embedding_graph")
        _wrap_function(_umap, "spectral_layout", "time_embedding_initialization")
        _wrap_function(_umap, "optimize_layout_euclidean", "time_embedding_optimization")

        yield
    finally:
        for owner, name, original in reversed(restore):
            setattr(owner, name, original)


#: Lazy dependencies imported before algorithm timing and included in time_warmup.
LAZY_IMPORTS: dict[str, tuple[str, ...]] = {
    # `compute_ot_plan` does `import ot`; the kNN-diagonal variant additionally reaches for
    # sklearn's NearestNeighbors.
    "ot": ("ot", "sklearn.neighbors"),
    "ot-knndiag": ("ot", "sklearn.neighbors"),
    "ot-selfotmedian": ("ot", "sklearn.neighbors"),
    "ot-gramdiag": ("ot", "sklearn.neighbors"),
    "ot-uot-topk": ("ot",),
    "ot-uot-mass": ("ot",),
    # No sklearn.neighbors: there is no kNN step -- the within-batch blocks are zero.
    "ot-zerodiag": ("ot",),
    "ot-zerodiag-split": ("ot",),
    "eot-t0": ("ot",),
    "eot-50": ("ot",),
    "harmony-2d": ("harmonypy",),
    "harmony-50": ("harmonypy",),
    "scanorama-2d": ("scanorama",),
    "scanorama-50": ("scanorama",),
    "scvi-2d": ("scvi", "anndata"),
    "scvi-50": ("scvi", "anndata"),
}


def warmup(task_kind: str, backend: str, *, enabled: bool = True) -> float:
    """Run lazy imports and UMAP JIT compilation before algorithm timing.

    Return elapsed seconds, or NaN when no warmup applies. MultiMAP warms up in its
    own interpreter. Warmup failures are tolerated so the method can report its own
    dependency or execution error.
    """
    if not enabled:
        return math.nan

    modules = LAZY_IMPORTS.get(str(task_kind), ())
    if not modules and backend != "umap":
        return math.nan

    import importlib

    start = time.perf_counter()
    for name in modules:
        with contextlib.suppress(ImportError):
            importlib.import_module(name)

    if backend == "umap":
        with contextlib.suppress(Exception):
            import numpy as np
            import umap

            points = np.linspace(0.0, 1.0, 60, dtype=float).reshape(30, 2)
            umap.UMAP(
                n_components=2,
                n_neighbors=4,
                min_dist=0.1,
                init="spectral",
                metric="euclidean",
                random_state=0,
                n_jobs=1,
                n_epochs=2,
                verbose=False,
            ).fit_transform(points)
    return time.perf_counter() - start


# Resource metadata


def pin_torch_threads(threads: int | None) -> None:
    """Set intra-op threads only if torch is already loaded, avoiding imports for other methods."""
    if threads is None:
        return
    module = sys.modules.get("torch")
    if module is None:
        return
    with contextlib.suppress(Exception):
        module.set_num_threads(int(threads))


def blas_name() -> str:
    """Identify NumPy's BLAS for runtime resource comparisons.

    Apple Accelerate may ignore thread environment settings and expose no native pool;
    recording the library prevents pooling it with controlled OpenBLAS or MKL runs.
    """
    try:
        import numpy as np

        config = np.show_config(mode="dicts") or {}
        return str(config.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown"))
    except Exception:
        return "unknown"


def threadpool_summary() -> list[dict[str, Any]]:
    """The native thread pools actually loaded, when they can be introspected.

    Empty on Accelerate, which is itself informative rather than a failure.
    """
    try:
        import threadpoolctl

        return [
            {
                "user_api": entry.get("user_api"),
                "internal_api": entry.get("internal_api"),
                "num_threads": entry.get("num_threads"),
                "version": entry.get("version"),
            }
            for entry in threadpoolctl.threadpool_info()
        ]
    except Exception:
        return []


def _package_versions() -> dict[str, str]:
    names = (
        "numpy",
        "scipy",
        "scikit-learn",
        "openTSNE",
        "umap-learn",
        "pot",
        "numba",
        "scanpy",
        "anndata",
        "harmonypy",
        "scanorama",
        "scvi-tools",
        "torch",
    )
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
        except Exception:
            continue
    return out


def cpu_model() -> str:
    """The processor model name, or "" when the platform does not report one.

    BLAS libraries choose kernels by processor, so nodes with different CPUs can give
    slightly different floating-point results from one environment.
    """
    with contextlib.suppress(OSError):
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    if sys.platform == "darwin":
        import subprocess

        with contextlib.suppress(OSError, subprocess.SubprocessError):
            completed = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if completed.returncode == 0:
                return completed.stdout.strip()
    return platform.processor()


def environment_record(threads: int | None) -> dict[str, Any]:
    """Collect process, dependency, device, and thread provenance for content-addressed storage."""
    from ne_aligned_utils.config import observed_thread_env

    return {
        "threads_requested": -1 if threads is None else int(threads),
        "thread_env": observed_thread_env(),
        "threadpools": threadpool_summary(),
        "blas_name": blas_name(),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "packages": _package_versions(),
    }


def environment_fingerprint(record: dict[str, Any]) -> str:
    """A short, content-addressed id for one environment record."""
    payload = json.dumps(record, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def write_environment_record(directory: Path, record: dict[str, Any]) -> str:
    """Store environment provenance under its fingerprint and return that identifier.

    Repeated records share a file. Write failures do not fail the benchmark task.
    """
    fingerprint = environment_fingerprint(record)
    path = Path(directory) / f"environment_{fingerprint}.json"
    if not path.exists():
        with contextlib.suppress(OSError):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
    return fingerprint


__all__ = [
    "ALL_TIMING_FIELDS",
    "CHILD_TIMED_METHODS",
    "DIAGNOSTIC_FIELDS",
    "EMBEDDING_DIAGNOSTIC_FIELDS",
    "EXTRA_FIELDS",
    "OT_FIELDS",
    "SCHEMA_FIELDS",
    "STAGE_FIELDS",
    "StageTimer",
    "TASK_SCHEMAS",
    "TOTAL_FIELD",
    "blas_name",
    "environment_fingerprint",
    "environment_record",
    "finalise",
    "instrument_embedders",
    "pin_torch_threads",
    "resolve_schema",
    "threadpool_summary",
    "LAZY_IMPORTS",
    "warmup",
    "write_environment_record",
]
