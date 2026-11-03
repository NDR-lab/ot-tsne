"""Run MultiMAP in an isolated interpreter using NPZ input and output files.

Its NumPy 1.x dependency stack is separate from this package's NumPy 2.x stack.
Output rows follow batch 0 then batch 1. The subprocess produces the final layout.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ne_aligned_utils.config import DEFAULT_MULTIMAP_USE_NATIVE_SCALING
from ne_aligned_utils.methods import _split_joint_embedding, _validate_pair_arrays

PY_SCRIPTS_DIR = Path(__file__).resolve().parent / "py_scripts"
MULTIMAP_PYTHON_ENV_VAR = "NE_ALIGNED_MULTIMAP_PYTHON"
#: Conventional location of the isolated venv (see py_scripts/README.md); used
#: only when the environment variable above is unset.
DEFAULT_MULTIMAP_PYTHON = Path.home() / ".venvs" / "ne-aligned-multimap" / "bin" / "python"
DEFAULT_TIMEOUT_SECONDS = 3600
MULTIMAP_RUNNER = "multimap_runner.py"


class MultiMapSubprocessError(RuntimeError):
    """Raised when the MultiMAP interpreter cannot be located, run, or produce valid output."""


def multimap_python_command() -> list[str]:
    """Return the base command used to launch the isolated MultiMAP interpreter.

    Honours ``NE_ALIGNED_MULTIMAP_PYTHON`` (shell-tokenised, so a multi-token
    override like ``"python mock.py"`` works); otherwise falls back to
    :data:`DEFAULT_MULTIMAP_PYTHON` when that path exists.
    """
    override = os.environ.get(MULTIMAP_PYTHON_ENV_VAR)
    if override:
        tokens = shlex.split(override)
        if not tokens:
            raise MultiMapSubprocessError(f"{MULTIMAP_PYTHON_ENV_VAR} is set but empty.")
        return tokens
    if DEFAULT_MULTIMAP_PYTHON.exists():
        return [str(DEFAULT_MULTIMAP_PYTHON)]
    raise MultiMapSubprocessError(
        "No MultiMAP interpreter found. MultiMAP needs its own numpy<2 environment; "
        f"create one (see {PY_SCRIPTS_DIR / 'README.md'}) and point "
        f"{MULTIMAP_PYTHON_ENV_VAR} at its python, or install it at "
        f"{DEFAULT_MULTIMAP_PYTHON}."
    )


def multimap_available() -> bool:
    """Whether the isolated MultiMAP interpreter is reachable."""
    override = os.environ.get(MULTIMAP_PYTHON_ENV_VAR)
    if override:
        tokens = shlex.split(override)
        if not tokens:
            return False
        return shutil.which(tokens[0]) is not None or Path(tokens[0]).exists()
    return DEFAULT_MULTIMAP_PYTHON.exists()


def write_multimap_input(
    path: str | Path,
    x0: np.ndarray,
    x1: np.ndarray,
    batch_ids: np.ndarray,
    params: dict[str, object],
) -> None:
    """Serialise the two batches, batch ids, and parameters into an ``.npz`` input file."""
    np.savez_compressed(
        str(path),
        x0=np.ascontiguousarray(np.asarray(x0, dtype=np.float64)),
        x1=np.ascontiguousarray(np.asarray(x1, dtype=np.float64)),
        batch_ids=np.ascontiguousarray(np.asarray(batch_ids, dtype=np.int64)),
        params_json=np.asarray(json.dumps(params)),
    )


def read_multimap_output(path: str | Path, n_total: int, n_components: int) -> np.ndarray:
    """Read the ``embedding`` array from an ``.npz`` output file and validate its shape."""
    with np.load(str(path), allow_pickle=False) as handle:
        if "embedding" not in handle:
            raise MultiMapSubprocessError(
                f"MultiMAP output {path} is missing the 'embedding' array."
            )
        embedding = np.asarray(handle["embedding"], dtype=float)

    if embedding.ndim != 2:
        raise MultiMapSubprocessError(
            f"MultiMAP output embedding must be 2D, got shape {embedding.shape!r}."
        )
    if embedding.shape[0] != int(n_total):
        raise MultiMapSubprocessError(
            f"MultiMAP output embedding has {embedding.shape[0]} rows, expected {int(n_total)}."
        )
    if embedding.shape[1] != int(n_components):
        raise MultiMapSubprocessError(
            f"MultiMAP output embedding has width {embedding.shape[1]}, "
            f"expected {int(n_components)}."
        )
    if not np.isfinite(embedding).all():
        raise MultiMapSubprocessError("MultiMAP output embedding contains non-finite values.")
    return embedding


def run_multimap_subprocess(
    x0: np.ndarray,
    x1: np.ndarray,
    batch_ids: np.ndarray,
    params: dict[str, object],
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    timer: object | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run the bundled MultiMAP runner over the two batches and return its embedding.

    Raises :class:`MultiMapSubprocessError` with the captured stderr tail on any
    failure (missing interpreter, non-zero exit, absent/invalid output).
    """
    script_path = PY_SCRIPTS_DIR / MULTIMAP_RUNNER
    if not script_path.exists():
        raise MultiMapSubprocessError(f"MultiMAP runner {script_path} does not exist.")

    n_total = int(np.asarray(x0).shape[0] + np.asarray(x1).shape[0])
    n_components = int(params.get("n_components", 2))
    command = multimap_python_command()
    with tempfile.TemporaryDirectory(prefix="ne_aligned_multimap_") as tmp_dir:
        input_path = Path(tmp_dir) / "input.npz"
        output_path = Path(tmp_dir) / "output.npz"
        write_multimap_input(input_path, x0, x1, batch_ids, params)
        full_command = [*command, str(script_path), str(input_path), str(output_path)]
        # Use persistent temporary cache directories when home directories are restricted.
        # Per-call caches would discard Numba compilation between seeds.
        child_env = os.environ.copy()
        shared_cache_root = Path(tempfile.gettempdir()) / "ne_aligned_multimap_cache"
        numba_cache = shared_cache_root / "numba"
        matplotlib_cache = shared_cache_root / "matplotlib"
        numba_cache.mkdir(parents=True, exist_ok=True)
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        child_env.setdefault("NUMBA_CACHE_DIR", str(numba_cache))
        child_env.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        # Force a headless backend because the isolated environment may lack notebook-inline
        # support.
        child_env["MPLBACKEND"] = "Agg"
        try:
            completed = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=child_env,
            )
        except FileNotFoundError as exc:
            raise MultiMapSubprocessError(
                f"Failed to launch MultiMAP command {full_command!r}: {exc}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise MultiMapSubprocessError(f"MultiMAP runner timed out after {timeout}s.") from exc

        if completed.returncode != 0:
            raise MultiMapSubprocessError(
                f"MultiMAP runner exited with code {completed.returncode}.\n"
                f"--- stderr (tail) ---\n{completed.stderr[-4000:]}"
            )
        if not output_path.exists():
            raise MultiMapSubprocessError(
                "MultiMAP runner produced no output file.\n"
                f"--- stdout (tail) ---\n{completed.stdout[-2000:]}"
            )
        embedding = read_multimap_output(output_path, n_total=n_total, n_components=n_components)
        diagnostics = _read_feature_diagnostics(output_path)
        child_timings = read_multimap_timings(output_path)

    artifacts: dict[str, object] = {
        "multimap_runner": MULTIMAP_RUNNER,
        "multimap_python": command[0],
        "multimap_stdout_tail": completed.stdout[-2000:],
        **diagnostics,
        "multimap_timings": child_timings,
    }
    return embedding, artifacts


def _read_feature_diagnostics(path: str | Path) -> dict[str, int]:
    """Read scaling diagnostics written by the runner."""
    with np.load(str(path), allow_pickle=False) as handle:
        return {
            key: int(np.asarray(handle[key]))
            for key in (
                "n_constant_features",
                "n_constant_features_batch0",
                "n_constant_features_batch1",
                "n_features_used",
            )
            if key in handle
        }


def read_multimap_timings(path: str | Path) -> dict[str, float]:
    """Read the runner's algorithm-only seconds, excluding startup, imports, and JIT warmup.

    Missing keys return an empty dict and remain NaN downstream.
    """
    try:
        with np.load(str(path), allow_pickle=False) as handle:
            return {
                key: float(np.asarray(handle[key]))
                for key in ("time_total", "time_warmup")
                if key in handle
            }
    except (OSError, KeyError, ValueError):
        return {}


def compute_multimap_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    embd_dim: int = 2,
    random_state: int = 0,
    n_neighbors: int | None = None,
    min_dist: float | None = None,
    spread: float | None = None,
    n_epochs: int | None = None,
    init: str | None = None,
    scale: bool = True,
    center_batches: bool = True,
    use_native_scaling: bool = DEFAULT_MULTIMAP_USE_NATIVE_SCALING,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Integrate processed features with MultiMAP in an isolated NumPy 1.x interpreter.

    The output is a final layout; two dimensions are native to this method. Native
    scaling uses Scanpy per batch. ``use_native_scaling=False`` selects the guarded
    compatibility implementation.
    """
    x0, x1 = _validate_pair_arrays(x0, x1)
    if not np.isfinite(x0).all() or not np.isfinite(x1).all():
        raise ValueError("MultiMAP input contains non-finite values.")
    n0, n1 = x0.shape[0], x1.shape[0]
    embd_dim = int(embd_dim)
    if embd_dim < 1:
        raise ValueError(f"embd_dim must be a positive integer, got {embd_dim!r}.")
    # ``None`` on any layout parameter means "do not send it", which is how the native profile is
    # requested: MultiMAP then applies its own rule for that one parameter. Each is validated only
    # when a value was given, so a native run cannot be rejected for a value it never set.
    if n_neighbors is not None:
        n_neighbors = int(n_neighbors)
        if n_neighbors < 2:
            raise ValueError(f"n_neighbors must be at least 2, got {n_neighbors}.")
    if min_dist is not None:
        min_dist = float(min_dist)
        if min_dist < 0.0:
            raise ValueError(f"min_dist must be nonnegative, got {min_dist}.")
    if spread is not None:
        spread = float(spread)
        if spread <= 0.0:
            raise ValueError(f"spread must be positive, got {spread}.")
    if min_dist is not None and spread is not None and min_dist > spread:
        raise ValueError(f"min_dist ({min_dist}) must not exceed spread ({spread}).")
    if n_epochs is not None:
        n_epochs = int(n_epochs)
        if n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {n_epochs}.")
    if init is not None:
        init = str(init)
        if init not in ("spectral", "random"):
            raise ValueError(f"init must be 'spectral' or 'random', got {init!r}.")
    params: dict[str, object] = {
        "method": "multimap",
        "n_components": embd_dim,
        "seed": int(random_state),
        "scale": bool(scale),
        "center_batches": bool(center_batches),
        "use_native_scaling": bool(use_native_scaling),
    }
    if not center_batches and scale:
        raise ValueError(
            "center_batches=False requires scale=False because per-batch scaling centers data."
        )
    if use_native_scaling and not scale:
        raise ValueError("use_native_scaling=True requires scale=True.")
    # A parameter left as None is omitted rather than back-filled, which is how MultiMAP's native
    # profile is requested. The historical controlled profile remains available by supplying all
    # five values explicitly. Absent keys are recorded as native in the artifacts below.
    for key, value in (
        ("n_neighbors", n_neighbors),
        ("min_dist", min_dist),
        ("spread", spread),
        ("n_epochs", n_epochs),
        ("init", init),
    ):
        if value is not None:
            params[key] = value
    batch_ids = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
    embedding, subprocess_artifacts = run_multimap_subprocess(x0, x1, batch_ids, params)
    if timer is not None:
        # `set`, not `record`: these are absolute durations measured in another process, not
        # intervals to accumulate. MultiMAP is a fused method, so the only stage it reports is the
        # total; the warm-up is a diagnostic and is deliberately outside it.
        for name, seconds in dict(subprocess_artifacts.get("multimap_timings") or {}).items():
            timer.set(name, float(seconds))

    if not return_artifacts:
        return embedding
    left_embedding, right_embedding = _split_joint_embedding(embedding, n0)
    artifacts: dict[str, object] = {
        **subprocess_artifacts,
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "multimap_dim": int(embd_dim),
        "multimap_scale": bool(scale),
        "multimap_center_batches": bool(center_batches),
        "multimap_use_native_scaling": bool(use_native_scaling),
        "random_state": int(random_state),
    }
    # "native" records that we did not set it, which is a different statement from any number.
    artifacts["multimap_n_neighbors"] = "native" if n_neighbors is None else n_neighbors
    artifacts["multimap_min_dist"] = "native" if min_dist is None else min_dist
    artifacts["multimap_spread"] = "native" if spread is None else spread
    artifacts["multimap_n_epochs"] = "native" if n_epochs is None else n_epochs
    artifacts["multimap_init"] = "native" if init is None else init
    return embedding, artifacts


__all__ = [
    "DEFAULT_MULTIMAP_PYTHON",
    "MULTIMAP_PYTHON_ENV_VAR",
    "MultiMapSubprocessError",
    "PY_SCRIPTS_DIR",
    "compute_multimap_embedding_from_arrays",
    "multimap_available",
    "multimap_python_command",
    "read_multimap_output",
    "run_multimap_subprocess",
    "write_multimap_input",
]
