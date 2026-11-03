"""Run Seurat, LIGER, and BC-tSNE in isolated R subprocesses.

HDF5 inputs contain both batches, batch IDs, and parameters; output rows follow
batch 0 then batch 1. ``NE_ALIGNED_RSCRIPT`` names the required launcher.
LIGER 2 uses ``NE_ALIGNED_LIGER_RSCRIPT`` for its separate API environment.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ne_aligned_utils.config import (
    DEFAULT_BACKEND,
    DEFAULT_LIGER_PRESELECTED_FEATURES,
    DEFAULT_SEURAT_PRESELECTED_FEATURES,
    DEFAULT_SEURAT_WORKFLOW,
)
from ne_aligned_utils.methods import (
    _split_joint_embedding,
    _validate_pair_arrays,
    embed_2d,
    embedder_artifacts,
)

R_SCRIPTS_DIR = Path(__file__).resolve().parent / "r_scripts"
RSCRIPT_ENV_VAR = "NE_ALIGNED_RSCRIPT"
LIGER_RSCRIPT_ENV_VAR = "NE_ALIGNED_LIGER_RSCRIPT"
TIMEOUT_ENV_VAR = "NE_ALIGNED_R_TIMEOUT"
STANDARD_TIMEOUT_SECONDS = 14400

#: Maximum R subprocess duration in seconds; callers may override it for larger inputs.
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get(TIMEOUT_ENV_VAR, str(STANDARD_TIMEOUT_SECONDS)))


class RSubprocessError(RuntimeError):
    """Raised when an R method cannot be located, run, or produce valid output."""


_NO_LAUNCHER_HELP = f"""{RSCRIPT_ENV_VAR} is not set, so no R launcher has been named.

  For the environment this repository pins:
    export {RSCRIPT_ENV_VAR}="$HOME/.local/opt/ne-aligned-r/bin/micromamba run -n rverify Rscript"
    export MAMBA_ROOT_PREFIX="$HOME/.local/opt/ne-aligned-r/mmroot"

  To use the Rscript already on PATH, ask for it explicitly:
    export {RSCRIPT_ENV_VAR}=Rscript

  See reproduction/README.md and src/ne_aligned_utils/methods/external/r_scripts/README.md."""


def rscript_command(env_var: str = RSCRIPT_ENV_VAR) -> list[str]:
    """Return the shell-tokenized launcher from the requested environment variable.

    Require an explicit launcher to avoid unintended R versions. PATH lookup is allowed
    only when explicitly requested, for example ``NE_ALIGNED_RSCRIPT=Rscript``.
    """
    override = os.environ.get(env_var)
    if override is None:
        if env_var == RSCRIPT_ENV_VAR:
            raise RSubprocessError(_NO_LAUNCHER_HELP)
        raise RSubprocessError(
            f"{env_var} is not set. See reproduction/README.md "
            "for the separate current LIGER environment."
        )
    tokens = shlex.split(override)
    if not tokens:
        raise RSubprocessError(f"{env_var} is set but blank.")
    return tokens


def r_available(env_var: str = RSCRIPT_ENV_VAR) -> bool:
    """Return whether a launcher is configured, without probing multi-token commands."""
    return bool(shlex.split(os.environ.get(env_var, "")))


def write_r_input(
    path: str | Path,
    x0: np.ndarray,
    x1: np.ndarray,
    batch_ids: np.ndarray,
    params: dict[str, object],
    library_sizes: np.ndarray | None = None,
) -> None:
    """Serialise the two batches, batch ids, and parameters into an HDF5 input file.

    ``library_sizes`` (one per cell, batch 0 then batch 1) replaces each cell's total over the
    supplied genes as the divisor in the R script's count normalisation.
    """
    import h5py

    x0 = np.ascontiguousarray(np.asarray(x0, dtype=np.float64))
    x1 = np.ascontiguousarray(np.asarray(x1, dtype=np.float64))
    batch_ids = np.ascontiguousarray(np.asarray(batch_ids, dtype=np.int64))
    with h5py.File(str(path), "w") as handle:
        handle.create_dataset("x0", data=x0)
        handle.create_dataset("x1", data=x1)
        handle.create_dataset("batch_ids", data=batch_ids)
        if library_sizes is not None:
            handle.create_dataset(
                "library_size", data=np.ascontiguousarray(library_sizes, dtype=np.float64)
            )
        group = handle.create_group("params")
        # sizes are always written so the R side can verify / transpose matrices
        group.attrs["n0"] = int(x0.shape[0])
        group.attrs["n1"] = int(x1.shape[0])
        group.attrs["n_features"] = int(x0.shape[1])
        for key, value in params.items():
            group.attrs[key] = value


def read_r_timings(path: str | Path) -> dict[str, float]:
    """Read child-reported algorithm seconds from HDF5 timing attributes.

    Child timing excludes interpreter startup and imports. Missing timings return an
    empty dict and remain NaN downstream; malformed numeric attributes raise.
    """
    import h5py
    import numpy as _np

    try:
        with h5py.File(str(path), "r") as handle:
            group = handle.get("timings")
            if group is None:
                return {}
            attrs = dict(group.attrs)
    except OSError:
        # Ignore unreadable HDF5 files, but let malformed timing values raise.
        return {}
    # hdf5r represents scalar attributes as length-one arrays.
    return {
        str(key): float(_np.asarray(value, dtype=float).reshape(-1)[0])
        for key, value in attrs.items()
        if _np.asarray(value).size == 1
    }


def read_r_environment(path: str | Path) -> dict[str, object]:
    """Read the R version, BLAS/LAPACK and loaded package versions the R script recorded.

    Returns an empty dict for outputs that carry no record, such as the test suite's mock.
    """
    import h5py

    def text(value: object) -> str:
        value = np.asarray(value).reshape(-1)[0]
        return value.decode() if isinstance(value, bytes) else str(value)

    try:
        with h5py.File(str(path), "r") as handle:
            group = handle.get("environment")
            if group is None:
                return {}
            record: dict[str, object] = {key: text(v) for key, v in group.attrs.items()}
            packages = handle.get("r_packages")
            if packages is not None:
                record["packages"] = {key: text(v) for key, v in sorted(packages.attrs.items())}
    except OSError:
        return {}
    return record


def read_r_output(path: str | Path, n_total: int) -> np.ndarray:
    """Read the ``embedding`` dataset from an HDF5 output file, aligned to ``n_total`` rows.

    R (column-major) and HDF5 (row-major) can disagree on matrix orientation, so
    a transpose is applied when the row/column axes are swapped.
    """
    import h5py

    with h5py.File(str(path), "r") as handle:
        if "embedding" not in handle:
            raise RSubprocessError(f"R output {path} is missing the 'embedding' dataset.")
        embedding = np.asarray(handle["embedding"], dtype=float)

    if embedding.ndim != 2:
        raise RSubprocessError(f"R output embedding must be 2D, got shape {embedding.shape!r}.")
    if embedding.shape[0] != int(n_total) and embedding.shape[1] == int(n_total):
        embedding = np.ascontiguousarray(embedding.T)
    if embedding.shape[0] != int(n_total):
        raise RSubprocessError(
            f"R output embedding has {embedding.shape[0]} rows, expected {int(n_total)}."
        )
    if not np.isfinite(embedding).all():
        raise RSubprocessError("R output embedding contains non-finite values.")
    return embedding


def run_r_method(
    script: str,
    x0: np.ndarray,
    x1: np.ndarray,
    batch_ids: np.ndarray,
    params: dict[str, object],
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    launcher_env_var: str = RSCRIPT_ENV_VAR,
    library_sizes: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run a bundled R script over the two batches and return its ``(n0+n1) x D`` embedding.

    Raises :class:`RSubprocessError` with the captured R stderr tail on any
    failure (missing launcher, non-zero exit, absent/invalid output).
    """
    script_path = R_SCRIPTS_DIR / script
    if not script_path.exists():
        raise RSubprocessError(f"R script {script_path} does not exist.")
    n_total = int(np.asarray(x0).shape[0] + np.asarray(x1).shape[0])
    command = rscript_command(launcher_env_var)
    with tempfile.TemporaryDirectory(prefix="ne_aligned_r_") as tmp_dir:
        input_path = Path(tmp_dir) / "input.h5"
        output_path = Path(tmp_dir) / "output.h5"
        write_r_input(input_path, x0, x1, batch_ids, params, library_sizes=library_sizes)
        full_command = [*command, str(script_path), str(input_path), str(output_path)]
        try:
            completed = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise RSubprocessError(f"Failed to launch R command {full_command!r}: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RSubprocessError(f"R script {script} timed out after {timeout}s.") from exc

        if completed.returncode != 0:
            raise RSubprocessError(
                f"R script {script} exited with code {completed.returncode}.\n"
                f"--- R stderr (tail) ---\n{completed.stderr[-4000:]}"
            )
        if not output_path.exists():
            raise RSubprocessError(
                f"R script {script} produced no output file.\n"
                f"--- R stdout (tail) ---\n{completed.stdout[-2000:]}"
            )
        embedding = read_r_output(output_path, n_total=n_total)
        child_timings = read_r_timings(output_path)
        r_environment = read_r_environment(output_path)

    artifacts: dict[str, object] = {
        "r_script": script,
        "r_command": command[0],
        "r_stdout_tail": completed.stdout[-2000:],
        "r_timings": child_timings,
    }
    if r_environment:
        artifacts["r_environment"] = json.dumps(r_environment, sort_keys=True)
    return embedding, artifacts


def _validate_library_sizes(library_sizes: np.ndarray | None, n_total: int) -> np.ndarray | None:
    if library_sizes is None:
        return None
    library_sizes = np.asarray(library_sizes, dtype=float).reshape(-1)
    if library_sizes.shape[0] != n_total:
        raise ValueError(f"library_sizes has {library_sizes.shape[0]} entries, expected {n_total}.")
    if not np.all(np.isfinite(library_sizes)) or np.any(library_sizes <= 0):
        raise ValueError("library_sizes must be finite and positive.")
    return library_sizes


def _batch_ids(n0: int, n1: int) -> np.ndarray:
    return np.concatenate([np.zeros(int(n0), dtype=int), np.ones(int(n1), dtype=int)])


def _run_r_integration(
    script: str,
    method: str,
    x0: np.ndarray,
    x1: np.ndarray,
    dim: int,
    random_state: int,
    extra_params: dict[str, object] | None = None,
    timer: object | None = None,
    launcher_env_var: str = RSCRIPT_ENV_VAR,
    library_sizes: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object], int]:
    """Run R integration and record child timings as absolute stage durations.

    The task runner separately measures the whole method call, including subprocess overhead.
    """
    x0, x1 = _validate_pair_arrays(x0, x1)
    n0, n1 = x0.shape[0], x1.shape[0]
    library_sizes = _validate_library_sizes(library_sizes, n0 + n1)
    params: dict[str, object] = {"method": method, "dim": int(dim), "seed": int(random_state)}
    if extra_params:
        params.update(extra_params)
    representation, r_artifacts = run_r_method(
        script,
        x0,
        x1,
        _batch_ids(n0, n1),
        params,
        launcher_env_var=launcher_env_var,
        library_sizes=library_sizes,
    )
    r_artifacts["library_size_source"] = (
        "selected_genes" if library_sizes is None else "all_genes"
    )
    if timer is not None:
        for name, seconds in dict(r_artifacts.get("r_timings") or {}).items():
            timer.set(name, float(seconds))
    return representation, r_artifacts, n0


def compute_seurat_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    embd_dim: int = 2,
    random_state: int = 0,
    return_artifacts: bool = False,
    timer: object | None = None,
    preselected_features: bool = DEFAULT_SEURAT_PRESELECTED_FEATURES,
    workflow: str = DEFAULT_SEURAT_WORKFLOW,
    library_sizes: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Run Seurat CCA integration in ``embd_dim`` dimensions.

    ``workflow`` selects v5 layers or v3-style anchors. ``preselected_features=True``
    uses all supplied columns for anchors instead of selecting variable genes again.
    ``library_sizes`` (batch 0 then batch 1) replaces the supplied-gene totals in LogNormalize.
    """
    if workflow not in {"legacy", "v5_cca"}:
        raise ValueError("workflow must be 'legacy' or 'v5_cca'")
    representation, r_artifacts, n0 = _run_r_integration(
        "seurat.R",
        "seurat",
        x0,
        x1,
        embd_dim,
        random_state,
        extra_params={
            "preselected_features": bool(preselected_features),
            "workflow": workflow,
        },
        timer=timer,
        library_sizes=library_sizes,
    )
    if not return_artifacts:
        return representation
    left_embedding, right_embedding = _split_joint_embedding(representation, n0)
    artifacts: dict[str, object] = {
        **r_artifacts,
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "seurat_dim": int(embd_dim),
        "seurat_dim_used": int(representation.shape[1]),
        "preselected_features": bool(preselected_features),
        "workflow": workflow,
        "random_state": int(random_state),
    }
    return representation, artifacts


def compute_seurat_then_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    seurat_dim: int = 50,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_jobs: int = 1,
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    backend: str = DEFAULT_BACKEND,
    n_neighbors: int | None = None,
    return_artifacts: bool = False,
    timer: object | None = None,
    preselected_features: bool = DEFAULT_SEURAT_PRESELECTED_FEATURES,
    workflow: str = DEFAULT_SEURAT_WORKFLOW,
    library_sizes: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Integrate with Seurat in ``seurat_dim`` dimensions, then apply the selected 2D backend.

    Feature and workflow controls match ``compute_seurat_embedding_from_arrays``.
    """
    if workflow not in {"legacy", "v5_cca"}:
        raise ValueError("workflow must be 'legacy' or 'v5_cca'")
    result = _r_integration_then_tsne(
        "seurat.R",
        "seurat",
        x0,
        x1,
        seurat_dim,
        random_state,
        perplexity=perplexity,
        n_jobs=n_jobs,
        n_iter=n_iter,
        early_exaggeration_iter=early_exaggeration_iter,
        backend=backend,
        n_neighbors=n_neighbors,
        return_artifacts=return_artifacts,
        dim_key="seurat_dim",
        extra_params={
            "preselected_features": bool(preselected_features),
            "workflow": workflow,
        },
        timer=timer,
        library_sizes=library_sizes,
    )
    if not return_artifacts:
        return result
    embedding, artifacts = result
    artifacts["preselected_features"] = bool(preselected_features)
    artifacts["workflow"] = workflow
    return embedding, artifacts


def compute_seurat3_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    embd_dim: int = 2,
    random_state: int = 0,
    return_artifacts: bool = False,
    timer: object | None = None,
    preselected_features: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Historical Seurat v3-style anchor workflow (backs ``seurat3-2d``)."""
    return compute_seurat_embedding_from_arrays(
        x0,
        x1,
        embd_dim=embd_dim,
        random_state=random_state,
        return_artifacts=return_artifacts,
        timer=timer,
        preselected_features=preselected_features,
        workflow="legacy",
    )


def compute_seurat3_then_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    seurat_dim: int = 50,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_jobs: int = 1,
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    backend: str = DEFAULT_BACKEND,
    n_neighbors: int | None = None,
    return_artifacts: bool = False,
    timer: object | None = None,
    preselected_features: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Historical Seurat v3-style integration followed by the shared 2D backend."""
    return compute_seurat_then_tsne_embedding_from_arrays(
        x0,
        x1,
        seurat_dim=seurat_dim,
        random_state=random_state,
        perplexity=perplexity,
        n_jobs=n_jobs,
        n_iter=n_iter,
        early_exaggeration_iter=early_exaggeration_iter,
        backend=backend,
        n_neighbors=n_neighbors,
        return_artifacts=return_artifacts,
        timer=timer,
        preselected_features=preselected_features,
        workflow="legacy",
    )


def compute_liger1_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    embd_dim: int = 2,
    random_state: int = 0,
    return_artifacts: bool = False,
    timer: object | None = None,
    preselected_features: bool = False,
    requested_features: int | None = None,
    couple_quantile_seed: bool = True,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Run LIGER 1 iNMF with ``embd_dim`` factors and quantile alignment.

    The benchmark seed controls both steps. ``couple_quantile_seed=False`` leaves
    quantile alignment at rliger's default seed for compatibility.
    """
    if requested_features is not None and int(requested_features) < 1:
        raise ValueError("requested_features must be a positive integer or None")
    if preselected_features and requested_features is not None:
        raise ValueError("preselected_features and requested_features cannot both be set")
    representation, r_artifacts, n0 = _run_r_integration(
        "liger1.R",
        "liger1",
        x0,
        x1,
        embd_dim,
        random_state,
        extra_params={
            "preselected_features": bool(preselected_features),
            "requested_features": 0 if requested_features is None else int(requested_features),
            "couple_quantile_seed": bool(couple_quantile_seed),
        },
        timer=timer,
    )
    if not return_artifacts:
        return representation
    left_embedding, right_embedding = _split_joint_embedding(representation, n0)
    selected_match = re.search(
        r"NE_ALIGNED_LIGER_SELECTED_FEATURES=(\d+)", str(r_artifacts.get("r_stdout_tail", ""))
    )
    artifacts: dict[str, object] = {
        **r_artifacts,
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "liger1_k": int(embd_dim),
        "liger1_k_used": int(representation.shape[1]),
        "preselected_features": bool(preselected_features),
        "requested_features": requested_features,
        "selected_features": None if selected_match is None else int(selected_match.group(1)),
        "couple_quantile_seed": bool(couple_quantile_seed),
        "random_state": int(random_state),
    }
    return representation, artifacts


def compute_liger1_then_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    liger_dim: int = 50,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_jobs: int = 1,
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    backend: str = DEFAULT_BACKEND,
    n_neighbors: int | None = None,
    return_artifacts: bool = False,
    timer: object | None = None,
    preselected_features: bool = False,
    requested_features: int | None = None,
    couple_quantile_seed: bool = True,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Historical LIGER 1 iNMF followed by the shared 2D backend.

    The feature and quantile-seed controls have the same meanings as in
    :func:`compute_liger1_embedding_from_arrays`.
    """
    if requested_features is not None and int(requested_features) < 1:
        raise ValueError("requested_features must be a positive integer or None")
    if preselected_features and requested_features is not None:
        raise ValueError("preselected_features and requested_features cannot both be set")
    result = _r_integration_then_tsne(
        "liger1.R",
        "liger1",
        x0,
        x1,
        liger_dim,
        random_state,
        perplexity=perplexity,
        n_jobs=n_jobs,
        n_iter=n_iter,
        early_exaggeration_iter=early_exaggeration_iter,
        backend=backend,
        n_neighbors=n_neighbors,
        return_artifacts=return_artifacts,
        dim_key="liger1_k",
        extra_params={
            "preselected_features": bool(preselected_features),
            "requested_features": 0 if requested_features is None else int(requested_features),
            "couple_quantile_seed": bool(couple_quantile_seed),
        },
        timer=timer,
    )
    if not return_artifacts:
        return result
    embedding, artifacts = result
    selected_match = re.search(
        r"NE_ALIGNED_LIGER_SELECTED_FEATURES=(\d+)", str(artifacts.get("r_stdout_tail", ""))
    )
    artifacts["preselected_features"] = bool(preselected_features)
    artifacts["requested_features"] = requested_features
    artifacts["selected_features"] = (
        None if selected_match is None else int(selected_match.group(1))
    )
    artifacts["couple_quantile_seed"] = bool(couple_quantile_seed)
    return embedding, artifacts


def compute_liger_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    embd_dim: int = 2,
    random_state: int = 0,
    return_artifacts: bool = False,
    timer: object | None = None,
    preselected_features: bool = DEFAULT_LIGER_PRESELECTED_FEATURES,
    requested_features: int | None = None,
    library_sizes: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Current LIGER iNMF followed by its default quantile alignment.

    Current LIGER lives in a separate R environment because it cannot coexist with the pinned
    historical rliger 1.0.1 installation. By default all supplied columns are treated as the
    benchmark's preselected feature panel. Set ``preselected_features=False`` to let LIGER select
    features itself; ``requested_features`` optionally controls how many it selects.
    """
    if requested_features is not None and int(requested_features) < 1:
        raise ValueError("requested_features must be a positive integer or None")
    if preselected_features and requested_features is not None:
        raise ValueError("preselected_features and requested_features cannot both be set")
    representation, r_artifacts, n0 = _run_r_integration(
        "liger.R",
        "liger",
        x0,
        x1,
        embd_dim,
        random_state,
        extra_params={
            "preselected_features": bool(preselected_features),
            "requested_features": 0 if requested_features is None else int(requested_features),
        },
        timer=timer,
        launcher_env_var=LIGER_RSCRIPT_ENV_VAR,
        library_sizes=library_sizes,
    )
    if not return_artifacts:
        return representation
    left_embedding, right_embedding = _split_joint_embedding(representation, n0)
    selected_match = re.search(
        r"NE_ALIGNED_LIGER_SELECTED_FEATURES=(\d+)", str(r_artifacts.get("r_stdout_tail", ""))
    )
    return representation, {
        **r_artifacts,
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "liger_k": int(embd_dim),
        "liger_k_used": int(representation.shape[1]),
        "preselected_features": bool(preselected_features),
        "requested_features": requested_features,
        "selected_features": None if selected_match is None else int(selected_match.group(1)),
        "alignment": "quantileNorm",
        "random_state": int(random_state),
    }


def compute_liger_then_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    liger_dim: int = 50,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_jobs: int = 1,
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    backend: str = DEFAULT_BACKEND,
    n_neighbors: int | None = None,
    return_artifacts: bool = False,
    timer: object | None = None,
    preselected_features: bool = DEFAULT_LIGER_PRESELECTED_FEATURES,
    requested_features: int | None = None,
    library_sizes: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Current LIGER to ``liger_dim`` factors, followed by the shared 2D backend."""
    if requested_features is not None and int(requested_features) < 1:
        raise ValueError("requested_features must be a positive integer or None")
    if preselected_features and requested_features is not None:
        raise ValueError("preselected_features and requested_features cannot both be set")
    result = _r_integration_then_tsne(
        "liger.R",
        "liger",
        x0,
        x1,
        liger_dim,
        random_state,
        perplexity=perplexity,
        n_jobs=n_jobs,
        n_iter=n_iter,
        early_exaggeration_iter=early_exaggeration_iter,
        backend=backend,
        n_neighbors=n_neighbors,
        return_artifacts=return_artifacts,
        dim_key="liger_k",
        extra_params={
            "preselected_features": bool(preselected_features),
            "requested_features": 0 if requested_features is None else int(requested_features),
            "alignment": "quantileNorm",
        },
        timer=timer,
        launcher_env_var=LIGER_RSCRIPT_ENV_VAR,
        library_sizes=library_sizes,
    )
    if not return_artifacts:
        return result
    embedding, artifacts = result
    selected_match = re.search(
        r"NE_ALIGNED_LIGER_SELECTED_FEATURES=(\d+)", str(artifacts.get("r_stdout_tail", ""))
    )
    artifacts["selected_features"] = (
        None if selected_match is None else int(selected_match.group(1))
    )
    return embedding, artifacts


def compute_bc_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    random_state: int = 0,
    perplexity: float = 30.0,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """BC-tSNE (batch-balanced t-SNE, R) — a single-2D method (backs ``bc-tsne``).

    BC-tSNE is intrinsically a 2D embedder, so the R side returns the final
    ``(n0+n1) x 2`` embedding directly (no Python t-SNE tail).
    """
    representation, r_artifacts, n0 = _run_r_integration(
        "bc_tsne.R",
        "bc_tsne",
        x0,
        x1,
        2,
        random_state,
        extra_params={"perplexity": float(perplexity)},
        timer=timer,
    )
    if representation.shape[1] != 2:
        raise RSubprocessError(
            f"bc-tsne must return a 2D embedding, got width {representation.shape[1]}."
        )
    if not return_artifacts:
        return representation
    left_embedding, right_embedding = _split_joint_embedding(representation, n0)
    artifacts: dict[str, object] = {
        **r_artifacts,
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "perplexity": float(perplexity),
        "random_state": int(random_state),
    }
    return representation, artifacts


def _r_integration_then_tsne(
    script: str,
    method: str,
    x0: np.ndarray,
    x1: np.ndarray,
    dim: int,
    random_state: int,
    *,
    perplexity: float,
    n_jobs: int,
    n_iter: int | None,
    early_exaggeration_iter: int | None,
    backend: str = DEFAULT_BACKEND,
    n_neighbors: int | None = None,
    return_artifacts: bool,
    dim_key: str,
    extra_params: dict[str, object] | None = None,
    timer: object | None = None,
    launcher_env_var: str = RSCRIPT_ENV_VAR,
    library_sizes: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    representation, r_artifacts, n0 = _run_r_integration(
        script,
        method,
        x0,
        x1,
        dim,
        random_state,
        extra_params=extra_params,
        timer=timer,
        launcher_env_var=launcher_env_var,
        library_sizes=library_sizes,
    )
    if representation.shape[1] < 2:
        raise RSubprocessError(
            f"{method}+TSNE needs at least two integration dimensions before t-SNE, "
            f"but only {representation.shape[1]} were returned."
        )
    tsne_embedding, tsne_artifacts = embed_2d(
        representation,
        backend,
        random_state=random_state,
        perplexity=perplexity,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
        n_iter=n_iter,
        early_exaggeration_iter=early_exaggeration_iter,
        return_artifacts=True,
        timer=timer,
    )
    if not return_artifacts:
        return tsne_embedding
    left_embedding, right_embedding = _split_joint_embedding(tsne_embedding, n0)
    artifacts: dict[str, object] = {
        **r_artifacts,
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "integration_embedding": representation,
        dim_key: int(dim),
        f"{dim_key}_used": int(representation.shape[1]),
        **dict(extra_params or {}),
        **embedder_artifacts(backend, tsne_artifacts),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
    }
    if n_iter is not None:
        artifacts["n_iter"] = int(n_iter)
    if early_exaggeration_iter is not None:
        artifacts["early_exaggeration_iter"] = int(early_exaggeration_iter)
    return tsne_embedding, artifacts


__all__ = [
    "LIGER_RSCRIPT_ENV_VAR",
    "RSubprocessError",
    "R_SCRIPTS_DIR",
    "RSCRIPT_ENV_VAR",
    "compute_bc_tsne_embedding_from_arrays",
    "compute_liger1_embedding_from_arrays",
    "compute_liger1_then_tsne_embedding_from_arrays",
    "compute_liger_embedding_from_arrays",
    "compute_liger_then_tsne_embedding_from_arrays",
    "compute_seurat_embedding_from_arrays",
    "compute_seurat_then_tsne_embedding_from_arrays",
    "compute_seurat3_embedding_from_arrays",
    "compute_seurat3_then_tsne_embedding_from_arrays",
    "r_available",
    "read_r_output",
    "run_r_method",
    "rscript_command",
    "write_r_input",
]
