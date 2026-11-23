"""Disk cache for the balanced OT plan and its SVD, shared by OT-tSNE tasks that differ only in k
or seed.

The plan and its SVD depend on the two inputs and the solver settings, not on k or the seed. A
file is keyed by a hash of the input bytes, the solver settings, the thread count and the numpy
and POT versions, so it is used only where an independent computation would give the same bits.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from ne_aligned_utils.config import DEFAULT_OT_SVD_DIRECTIONS, DEFAULT_OT_SVD_SOLVER
from ne_aligned_utils.methods.common import stage as _stage
from ne_aligned_utils.methods.ot_affinity import PlanSpectrum, resolve_plan_svd
from ne_aligned_utils.methods.ot_transport import CPU_DEVICE, compute_ot_plan

#: Solver outputs kept next to the plan; everything ``compute_ot_tsne_embedding_from_arrays``
#: reads from the solver besides the plan itself.
_SCALARS = (
    "epsilon",
    "median_squared_distance",
    "ot_device",
    "sinkhorn_niter",
    "sinkhorn_final_error",
    "sinkhorn_converged",
)


def cache_key(
    x0: np.ndarray,
    x1: np.ndarray,
    *,
    rel_reg: float,
    tol: float,
    max_iter: int,
    device: str,
    threads: int | None,
    svd_solver: str = "full",
    svd_directions: int | None = None,
) -> str:
    """Hex digest identifying one plan and SVD computation.

    A full SVD keeps the key it had before the solver became a setting, so existing cache
    files stay valid. Truncated solvers add the solver, the directions and scikit-learn's
    version to the key.
    """
    import ot

    digest = hashlib.sha256()
    for array in (x0, x1):
        array = np.ascontiguousarray(array, dtype=float)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    settings = {
        "transport": "balanced",
        "solver": "sinkhorn",
        "rel_reg": repr(float(rel_reg)),
        "tol": repr(float(tol)),
        "max_iter": int(max_iter),
        "device": str(device),
        # Thread count changes the SVD's last bits, so a cache never crosses it.
        "threads": None if threads is None else int(threads),
        "numpy": np.__version__,
        "pot": ot.__version__,
    }
    used, directions = resolve_plan_svd(
        svd_solver, svd_directions, np.shape(x0)[0], np.shape(x1)[0]
    )
    if used != "full":
        import sklearn

        settings.update(svd_solver=used, svd_directions=directions, sklearn=sklearn.__version__)
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


def cached_plan_and_spectrum(
    x0: np.ndarray,
    x1: np.ndarray,
    *,
    cache_dir: str | Path,
    rel_reg: float,
    tol: float,
    max_iter: int,
    device: str = CPU_DEVICE,
    threads: int | None = None,
    timer: object | None = None,
    svd_solver: str = DEFAULT_OT_SVD_SOLVER,
    svd_directions: int | None = DEFAULT_OT_SVD_DIRECTIONS,
) -> tuple[np.ndarray, dict[str, object], PlanSpectrum, bool]:
    """Return ``(plan, solver artifacts, spectrum, cache_hit)``.

    On a miss, compute both under the same timing stages the uncached path uses, and save them.
    On a hit, nothing is timed: the caller must not report this task's total runtime.
    """
    cache_dir = Path(cache_dir)
    key = cache_key(
        x0,
        x1,
        rel_reg=rel_reg,
        tol=tol,
        max_iter=max_iter,
        device=device,
        threads=threads,
        svd_solver=svd_solver,
        svd_directions=svd_directions,
    )
    path = cache_dir / f"plan_svd_{key[:16]}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["key"]) != key:
                raise ValueError(f"{path} holds key {cached['key']}, expected {key}.")
            plan = np.array(cached["plan"])
            spectrum = PlanSpectrum(
                plan.shape[0],
                plan.shape[1],
                np.array(cached["singular_values"]),
                np.array(cached["eot"]),
            )
            artifacts = {name: cached[name].item() for name in _SCALARS if name in cached.files}
        return plan, artifacts, spectrum, True

    with _stage(timer, "time_pre_embedding"):
        plan, solver = compute_ot_plan(
            x0,
            x1,
            rel_reg=rel_reg,
            tol=tol,
            max_iter=max_iter,
            device=device,
            return_artifacts=True,
            timer=timer,
        )
    # The uncached path takes the SVD inside these two stages.
    with _stage(timer, "time_pre_embedding"), _stage(timer, "time_ot_postprocess"):
        spectrum = PlanSpectrum.from_plan(plan, solver=svd_solver, n_directions=svd_directions)
    artifacts = {name: solver[name] for name in _SCALARS if name in solver}

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Write then rename, so a concurrent reader never sees a partial file.
    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".npz.tmp", delete=False) as handle:
        np.savez(
            handle,
            key=np.asarray(key),
            plan=plan,
            singular_values=spectrum.singular_values,
            eot=spectrum.eot,
            **{name: np.asarray(value) for name, value in artifacts.items()},
        )
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return plan, artifacts, spectrum, False


__all__ = ["cache_key", "cached_plan_and_spectrum"]
