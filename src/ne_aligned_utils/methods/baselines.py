"""Joint and reference baseline embedding methods."""

import time

import numpy as np

from ne_aligned_utils.config import DEFAULT_BACKEND
from ne_aligned_utils.methods.common import (
    validate_pair_arrays as _validate_pair_arrays,
)
from ne_aligned_utils.methods.embedders import (
    compute_pca_embedding,
    embed_2d,
)


def compute_pca_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    embd_dim: int = 2,
    random_state: int = 0,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    x0, x1 = _validate_pair_arrays(x0, x1)
    joint_data = np.concatenate([x0, x1], axis=0)
    joint_embedding, pca_artifacts = compute_pca_embedding(
        joint_data,
        embd_dim=embd_dim,
        random_state=random_state,
        return_artifacts=True,
    )

    if not return_artifacts:
        return joint_embedding

    artifacts: dict[str, object] = {
        **pca_artifacts,
        "left_embedding": joint_embedding[: x0.shape[0]],
        "right_embedding": joint_embedding[x0.shape[0] :],
    }
    return joint_embedding, artifacts

def compute_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_jobs: int = 1,
    pca_components: int | None = 50,
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    backend: str = DEFAULT_BACKEND,
    n_neighbors: int | None = None,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    x0, x1 = _validate_pair_arrays(x0, x1)
    joint_data = np.concatenate([x0, x1], axis=0)

    # Joint PCA belongs to this method's pre-embedding time; skipped reduction still has a measured
    # stage.
    _pre_start = time.perf_counter()
    pca_artifacts: dict[str, object] = {"pca_components": None}
    if pca_components is not None:
        if not float(pca_components).is_integer():
            raise ValueError(f"pca_components must be an integer or None, got {pca_components}.")
        pca_dim = min(int(pca_components), joint_data.shape[0] - 1, joint_data.shape[1])
        if pca_dim >= 2 and pca_dim < joint_data.shape[1]:
            tsne_input, pca_artifacts = compute_pca_embedding(
                joint_data,
                embd_dim=pca_dim,
                random_state=random_state,
                return_artifacts=True,
            )
            pca_artifacts["pca_components"] = int(pca_dim)
        else:
            tsne_input = joint_data
            pca_artifacts = {"pca_components": None}
    else:
        tsne_input = joint_data
    if timer is not None:
        timer.record("time_pre_embedding", time.perf_counter() - _pre_start)

    joint_embedding, tsne_artifacts = embed_2d(
        tsne_input,
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
        return joint_embedding

    artifacts: dict[str, object] = {
        # Retain the backend's unprefixed artifact keys and record embedding_backend.
        **tsne_artifacts,
        "embedding_backend": backend,
        **pca_artifacts,
        # Expose the exact backend input for comparisons; exclude it from persisted task artifacts.
        "pca_embedding": tsne_input,
        "left_embedding": joint_embedding[: x0.shape[0]],
        "right_embedding": joint_embedding[x0.shape[0] :],
    }
    return joint_embedding, artifacts
