"""Run the vendored revised conditional t-SNE driver and compiled binary.

Apply joint PCA before conditioning out batch labels. Build the binary with
``scripts/setup/build_rctsne.sh`` or set ``NE_ALIGNED_RCTSNE_BIN``.
This API uses the paper's beta (same-batch suppression), passed as driver ``alpha``;
the driver reverses alpha/beta naming. See ``vendor/rctsne/VENDOR.md``.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import tempfile
from pathlib import Path

import numpy as np

from ne_aligned_utils.config import (
    DEFAULT_PERPLEXITY,
    DEFAULT_RC_TSNE_BETA,
    RC_TSNE_KEEP_NONZERO,
    RC_TSNE_MAX_ITER,
    RC_TSNE_STOP_EARLY_EXAG_ITER,
    RC_TSNE_THETA,
)
from ne_aligned_utils.methods import (
    _split_joint_embedding,
    _validate_pair_arrays,
    compute_pca_embedding,
)

VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "rctsne"
RCTSNE_BIN_ENV_VAR = "NE_ALIGNED_RCTSNE_BIN"
#: Where ``scripts/setup/build_rctsne.sh`` puts the binary; used only when the variable is unset.
DEFAULT_RCTSNE_BIN = VENDOR_DIR / "bin" / "fast_ctsne"
DEFAULT_TIMEOUT_SECONDS = 3600


class RcTsneSubprocessError(RuntimeError):
    """Raised when the rc-tsne binary cannot be located, run, or produce valid output."""


def rctsne_command() -> list[str]:
    """Return the binary command, shell-tokenizing ``NE_ALIGNED_RCTSNE_BIN`` when set."""
    override = os.environ.get(RCTSNE_BIN_ENV_VAR)
    if override:
        tokens = shlex.split(override)
        if not tokens:
            raise RcTsneSubprocessError(f"{RCTSNE_BIN_ENV_VAR} is set but empty.")
        return tokens
    return [str(DEFAULT_RCTSNE_BIN)]


def rctsne_available() -> bool:
    """Whether the binary is reachable (an env override is trusted, else the built path exists)."""
    override = os.environ.get(RCTSNE_BIN_ENV_VAR)
    if override:
        tokens = shlex.split(override)
        if not tokens:
            return False
        return shutil.which(tokens[0]) is not None or Path(tokens[0]).exists()
    return DEFAULT_RCTSNE_BIN.exists()


def _load_vendored_driver():
    """Import the vendored driver by path without exposing it as a package module."""
    path = VENDOR_DIR / "fast_ctsne.py"
    if not path.exists():
        raise RcTsneSubprocessError(
            f"The vendored rc-tsne driver is missing at {path}. "
            "The vendor tree is tracked in git; restore it rather than rebuilding it."
        )
    spec = importlib.util.spec_from_file_location("ne_aligned_utils._vendor_fast_ctsne", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_conditioning(labels: np.ndarray, beta: float) -> tuple[float, float, float]:
    """Return ``(beta, alpha, same_pair_ratio)`` using the paper's parameter names.

    Solve ``beta * ratio + alpha * (1 - ratio) = 1``. Per-row normalization means
    only ``alpha / beta`` affects affinities. The driver reverses these parameter names.
    """
    labels = np.asarray(labels)
    beta = float(beta)
    if not (0.0 < beta <= 1.0):
        raise ValueError(
            f"beta must be in (0, 1], got {beta}. Their driver asserts the same bound; "
            "beta = 1 is ordinary t-SNE and 0 is rejected upstream, not a limiting case."
        )
    counts = np.bincount(labels.astype(int))
    counts = counts[counts > 0]
    if counts.size < 2:
        raise ValueError(
            "Conditioning needs at least two label values; got "
            f"{counts.size}. With one batch there is nothing to factor out."
        )
    driver = _load_vendored_driver()
    n = int(labels.size)
    alpha = float(driver.get_beta(label_counts=list(counts), n=n, alpha=beta))
    ratio = float(sum(c * (c - 1) / (n * (n - 1)) for c in counts))
    return beta, alpha, ratio


def check_split_neighbour_budget(
    n0: int, n1: int, perplexity: float, keep_nonzero: float = RC_TSNE_KEEP_NONZERO
) -> None:
    """Require ``keep_nonzero / 2 * perplexity <= min(n0, n1) - 1``.

    Each batch must supply half the neighbor budget. Reject an infeasible request
    rather than silently lower perplexity.
    """
    per_side = float(keep_nonzero) / 2.0 * float(perplexity)
    smaller = min(int(n0), int(n1))
    if per_side > smaller - 1:
        raise ValueError(
            f"rc-tsne needs {per_side:g} same-batch neighbours per point at perplexity "
            f"{float(perplexity):g}, but the smaller batch has {smaller} rows "
            f"(batches are {int(n0)} and {int(n1)}). Lower the perplexity to at most "
            f"{(smaller - 1) / (float(keep_nonzero) / 2.0):.4g} for this input."
        )


def _joint_pca(
    joint: np.ndarray, pca_components: int | None, random_state: int
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply benchmark joint PCA with the standard dimension clamp and skip rule.

    This is benchmark preprocessing, separate from the rc-tSNE algorithm.
    """
    if pca_components is None:
        return joint, {"pca_components": None}
    if not float(pca_components).is_integer():
        raise ValueError(f"pca_components must be an integer or None, got {pca_components}.")
    pca_dim = min(int(pca_components), joint.shape[0] - 1, joint.shape[1])
    if pca_dim < 2 or pca_dim >= joint.shape[1]:
        return joint, {"pca_components": None}
    reduced, artifacts = compute_pca_embedding(
        joint, embd_dim=pca_dim, random_state=int(random_state), return_artifacts=True
    )
    artifacts["pca_components"] = int(pca_dim)
    return np.asarray(reduced, dtype=float), artifacts


def _validate_embedding(embedding, n_total: int) -> np.ndarray:
    """Shape and finiteness checks, mirroring ``read_multimap_output``."""
    if embedding is None:
        raise RcTsneSubprocessError(
            "The rc-tsne driver returned nothing. Upstream printed and returned None on a failed "
            "binary; vendor patch 3 makes that raise, so this means something else went wrong."
        )
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2:
        raise RcTsneSubprocessError(
            f"rc-tsne embedding must be 2D, got shape {embedding.shape!r}."
        )
    if embedding.shape[0] != int(n_total):
        raise RcTsneSubprocessError(
            f"rc-tsne embedding has {embedding.shape[0]} rows, expected {int(n_total)}."
        )
    if embedding.shape[1] != 2:
        raise RcTsneSubprocessError(
            f"rc-tsne embedding has width {embedding.shape[1]}, expected 2."
        )
    if not np.isfinite(embedding).all():
        raise RcTsneSubprocessError("rc-tsne embedding contains non-finite values.")
    return embedding


def run_rctsne(
    joint: np.ndarray,
    labels: np.ndarray,
    *,
    beta: float = DEFAULT_RC_TSNE_BETA,
    perplexity: float = DEFAULT_PERPLEXITY,
    random_state: int = 0,
    n_jobs: int = 1,
    return_loss: bool = True,
):
    """Run the vendored driver on a prepared joint feature matrix and conditioning labels."""
    driver = _load_vendored_driver()
    joint = np.ascontiguousarray(np.asarray(joint, dtype=float))
    labels = np.ascontiguousarray(np.asarray(labels).astype(np.int32))
    command = rctsne_command()
    if not rctsne_available():
        raise RcTsneSubprocessError(
            f"The rc-tsne binary is not available (looked for {command[0]!r}). "
            f"Build it with `bash scripts/setup/build_rctsne.sh`, then set "
            f"{RCTSNE_BIN_ENV_VAR} to the path it prints."
        )
    # Use a per-call temporary directory for the driver's input and output .dat files.
    with tempfile.TemporaryDirectory(prefix="ne_aligned_rctsne_") as work:
        try:
            result = driver.fast_ctsne(
                joint,
                labels,
                # Their `alpha` is the paper's beta. Do not "fix" this to beta=. See VENDOR.md.
                alpha=float(beta),
                perplexity=float(perplexity),
                theta=RC_TSNE_THETA,
                max_iter=RC_TSNE_MAX_ITER,
                stop_early_exag_iter=RC_TSNE_STOP_EARLY_EXAG_ITER,
                keep_nonzero=RC_TSNE_KEEP_NONZERO,
                initialization="pca",
                fixed_perp=True,
                seed=int(random_state),
                nthreads=int(n_jobs),
                return_loss=bool(return_loss),
                print_output=False,
                work_dir=work,
                binary_command=command,
            )
        except driver.FastCtsneError as exc:
            raise RcTsneSubprocessError(str(exc)) from exc
    if return_loss:
        embedding, loss = result
        return embedding, np.asarray(loss, dtype=float)
    return result, None


def compute_rc_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    random_state: int = 0,
    perplexity: float = DEFAULT_PERPLEXITY,
    beta: float = DEFAULT_RC_TSNE_BETA,
    pca_components: int | None = 50,
    n_jobs: int = 1,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Joint 2D embedding of two batches with revised conditional t-SNE.

    The batch id is the label conditioned out. There is no ``backend`` argument: the binary
    produces the final layout, so there is no separable visualizer to swap.
    """
    x0, x1 = _validate_pair_arrays(x0, x1)
    n0, n1 = x0.shape[0], x1.shape[0]
    check_split_neighbour_budget(n0, n1, perplexity)

    joint = np.concatenate([x0, x1], axis=0)
    labels = np.concatenate([np.zeros(n0, dtype=np.int32), np.ones(n1, dtype=np.int32)])
    beta, alpha, ratio = resolve_conditioning(labels, beta)

    reduced, pca_artifacts = _joint_pca(joint, pca_components, random_state)
    embedding, loss = run_rctsne(
        reduced,
        labels,
        beta=beta,
        perplexity=perplexity,
        random_state=random_state,
        n_jobs=n_jobs,
        return_loss=True,
    )
    embedding = _validate_embedding(embedding, n0 + n1)

    if not return_artifacts:
        return embedding
    left, right = _split_joint_embedding(embedding, n0)
    artifacts: dict[str, object] = {
        "left_embedding": left,
        "right_embedding": right,
        # Persist both multipliers in paper naming to expose parameter swaps.
        "rc_beta": float(beta),
        "rc_alpha": float(alpha),
        "rc_same_pair_ratio": float(ratio),
        "rc_perplexity": float(perplexity),
        "rc_keep_nonzero": int(RC_TSNE_KEEP_NONZERO),
        "rc_max_iter": int(RC_TSNE_MAX_ITER),
        "rc_stop_early_exag_iter": int(RC_TSNE_STOP_EARLY_EXAG_ITER),
        "rc_theta": float(RC_TSNE_THETA),
        # The driver records loss every 50 iterations, leaving other entries NaN.
        "rc_loss": loss,
        "embedding_backend": "rctsne",
        **pca_artifacts,
    }
    return embedding, artifacts
