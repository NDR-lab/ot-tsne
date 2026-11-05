"""Shared PCA, t-SNE, and UMAP embedding primitives."""

import numpy as np
import umap
from openTSNE import TSNE
from sklearn.decomposition import PCA

from ne_aligned_utils.config import DEFAULT_UMAP_N_NEIGHBORS
from ne_aligned_utils.methods.common import (
    stage as _stage,
    validate_embedding_dim as _validate_embedding_dim,
    validate_matrix as _validate_matrix,
)


def compute_pca_embedding(
    x: np.ndarray,
    embd_dim: int = 2,
    random_state: int = 0,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    x = _validate_matrix(x)
    max_dim = min(x.shape[0], x.shape[1])
    embd_dim = _validate_embedding_dim(embd_dim, max_dim, method_name="PCA")

    pca = PCA(n_components=embd_dim, random_state=int(random_state))
    embedding = np.asarray(pca.fit_transform(x), dtype=float)
    if not return_artifacts:
        return embedding

    artifacts: dict[str, object] = {
        "components": np.asarray(pca.components_, dtype=float),
        "explained_variance": np.asarray(pca.explained_variance_, dtype=float),
        "explained_variance_ratio": np.asarray(pca.explained_variance_ratio_, dtype=float),
        "mean": np.asarray(pca.mean_, dtype=float),
        "embd_dim": int(embd_dim),
    }
    return embedding, artifacts

def _safe_perplexity(n_samples: int, perplexity: float) -> float:
    n_samples = int(n_samples)
    if n_samples < 3:
        raise ValueError("t-SNE needs at least three samples.")
    return float(min(float(perplexity), max(1.0, (n_samples - 1) / 3.0)))

def compute_tsne_embedding(
    x: np.ndarray,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_jobs: int = 1,
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    x = _validate_matrix(x)
    tsne_kwargs: dict[str, object] = {
        "n_components": 2,
        "random_state": int(random_state),
        "perplexity": _safe_perplexity(x.shape[0], perplexity),
        "initialization": "pca",
        "n_jobs": int(n_jobs),
        "verbose": False,
    }
    if n_iter is not None:
        tsne_kwargs["n_iter"] = int(n_iter)
    if early_exaggeration_iter is not None:
        tsne_kwargs["early_exaggeration_iter"] = int(early_exaggeration_iter)

    tsne = TSNE(**tsne_kwargs)
    embedding = np.asarray(tsne.fit(x), dtype=float)
    if not return_artifacts:
        return embedding

    artifacts: dict[str, object] = {
        "perplexity": float(tsne_kwargs["perplexity"]),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
    }
    if n_iter is not None:
        artifacts["n_iter"] = int(n_iter)
    if early_exaggeration_iter is not None:
        artifacts["early_exaggeration_iter"] = int(early_exaggeration_iter)
    return embedding, artifacts

def compute_umap_embedding(
    x: np.ndarray,
    random_state: int = 0,
    n_neighbors: int | None = None,
    n_epochs: int | None = None,
    min_dist: float = 0.1,
    spread: float = 1.0,
    init: str = "spectral",
    n_jobs: int = 1,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Embed a feature matrix in 2D with UMAP and optional resolved-parameter artifacts.

    Resolve neighborhood size and epoch count before fitting so artifacts describe
    actual settings. Record ``min_dist``, ``spread``, and fitted kernel parameters;
    all determine the low-dimensional geometry. Seeded fitting uses one worker.
    """
    x = _validate_matrix(x)
    if n_neighbors is None:
        n_neighbors = DEFAULT_UMAP_N_NEIGHBORS
    n_neighbors = int(n_neighbors)
    if n_neighbors < 2:
        raise ValueError(f"n_neighbors must be at least 2, got {n_neighbors}.")
    # UMAP warns and clamps a too-large n_neighbors to n - 1. Doing it here keeps the recorded
    # value equal to the value used, and keeps that warning out of the logs.
    effective = min(n_neighbors, max(2, x.shape[0] - 1))
    # UMAP's own rule, applied here so the number is knowable from the artifacts.
    resolved_epochs = (
        int(n_epochs) if n_epochs is not None else (500 if x.shape[0] <= 10000 else 200)
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective,
        min_dist=float(min_dist),
        spread=float(spread),
        init=str(init),
        metric="euclidean",
        random_state=int(random_state),
        n_jobs=int(n_jobs),
        n_epochs=resolved_epochs,
        verbose=False,
    )
    embedding = np.asarray(reducer.fit_transform(x), dtype=float)
    if not return_artifacts:
        return embedding

    artifacts: dict[str, object] = {
        "n_neighbors": int(effective),
        "n_neighbors_requested": int(n_neighbors),
        "min_dist": float(min_dist),
        "spread": float(spread),
        "init": str(init),
        "n_epochs": resolved_epochs,
        "n_epochs_requested": -1 if n_epochs is None else int(n_epochs),
        # The fitted low-dimensional kernel, so the artifacts pin it rather than implying it.
        "a": float(getattr(reducer, "_a", float("nan"))),
        "b": float(getattr(reducer, "_b", float("nan"))),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
    }
    return embedding, artifacts

def embedder_artifacts(backend: str, artifacts: dict[str, object]) -> dict[str, object]:
    """Return backend-specific parameter artifacts for persistence."""
    if backend == "tsne":
        return {"embedding_backend": "tsne", "tsne_perplexity": artifacts["perplexity"]}
    if backend == "umap":
        return {
            "embedding_backend": "umap",
            "umap_n_neighbors": artifacts["n_neighbors"],
            "umap_n_neighbors_requested": artifacts["n_neighbors_requested"],
            "umap_min_dist": artifacts["min_dist"],
            "umap_spread": artifacts["spread"],
            "umap_init": artifacts["init"],
            "umap_n_epochs": artifacts["n_epochs"],
            "umap_a": artifacts["a"],
            "umap_b": artifacts["b"],
        }
    raise ValueError(f"backend must be 'tsne' or 'umap', got {backend!r}.")

def embed_2d(
    x: np.ndarray,
    backend: str = "tsne",
    *,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_neighbors: int | None = None,
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    n_jobs: int = 1,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Embed a representation in 2D and time the selected backend.

    UMAP neighborhood size is independent of t-SNE perplexity. ``n_iter`` maps to
    UMAP epochs, which are not equivalent to t-SNE iterations; artifacts record the
    resolved budget. Early exaggeration is supported only by t-SNE.
    """
    if backend == "tsne":
        with _stage(timer, "time_embedding"):
            return compute_tsne_embedding(
                x,
                random_state=random_state,
                perplexity=perplexity,
                n_jobs=n_jobs,
                n_iter=n_iter,
                early_exaggeration_iter=early_exaggeration_iter,
                return_artifacts=return_artifacts,
            )
    if backend == "umap":
        if early_exaggeration_iter is not None:
            # openTSNE's early-exaggeration phase has no UMAP counterpart. Silently dropping it
            # would make the two arms differ in a way the artifacts would not record.
            raise ValueError(
                "early_exaggeration_iter is a t-SNE parameter and has no UMAP equivalent; "
                "leave it unset when backend='umap'."
            )
        with _stage(timer, "time_embedding"):
            return compute_umap_embedding(
                x,
                random_state=random_state,
                n_neighbors=DEFAULT_UMAP_N_NEIGHBORS if n_neighbors is None else n_neighbors,
                n_epochs=n_iter,
                n_jobs=n_jobs,
                return_artifacts=return_artifacts,
            )
    raise ValueError(f"backend must be 'tsne' or 'umap', got {backend!r}.")
