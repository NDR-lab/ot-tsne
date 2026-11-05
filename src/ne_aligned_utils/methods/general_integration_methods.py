"""General-purpose integration methods and their 2D embedding wrappers."""

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ne_aligned_utils.config import (
    DEFAULT_BACKEND,
    DEFAULT_SCVI_BATCH_SIZE,
    DEFAULT_SCVI_MAX_EPOCHS,
)
from ne_aligned_utils.methods.common import (
    joint_data_and_batches as _joint_data_and_batches,
    split_joint_embedding as _split_joint_embedding,
    stage as _stage,
    synchronize_device as _synchronize_device,
    validate_matrix as _validate_matrix,
    validate_pair_arrays as _validate_pair_arrays,
    validate_positive_integer as _validate_positive_integer,
)
from ne_aligned_utils.methods.embedders import (
    compute_pca_embedding,
    embed_2d,
    embedder_artifacts,
)


def compute_cca_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    embd_dim: int = 2,
    return_artifacts: bool = False,
    center: bool = True,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Build a CCA-style embedding from singular vectors of ``x0 @ x1.T``.

    Center each batch independently unless ``center=False``. No within-batch covariance
    whitening is performed, so this is a simplified CCA baseline, separate from Seurat.
    ``embd_dim`` controls the retained shared dimensions.
    """
    x0, x1 = _validate_pair_arrays(x0, x1)
    embd_dim = _validate_positive_integer(embd_dim, "embd_dim")

    if center:
        # Per batch, not pooled: pooling would leave the between-batch mean difference in the
        # product, which is the offset this is meant to remove.
        x0 = x0 - x0.mean(axis=0, keepdims=True)
        x1 = x1 - x1.mean(axis=0, keepdims=True)

    cross_product = x0 @ x1.T
    u, singular_values, vt = np.linalg.svd(cross_product, full_matrices=False)
    q = min(embd_dim, u.shape[1], vt.shape[0])
    left_embedding = u[:, :q]
    right_embedding = vt.T[:, :q]
    joint_embedding = np.concatenate([left_embedding, right_embedding], axis=0)

    if not return_artifacts:
        return joint_embedding

    artifacts: dict[str, object] = {
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "singular_values": np.asarray(singular_values, dtype=float),
        "embd_dim": int(q),
        "embd_dim_requested": int(embd_dim),
        "centered": bool(center),
    }
    return joint_embedding, artifacts


def compute_cca_then_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    cca_dim: int = 50,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_jobs: int = 1,
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    backend: str = DEFAULT_BACKEND,
    n_neighbors: int | None = None,
    return_artifacts: bool = False,
    center: bool = True,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Integrate into ``cca_dim`` shared dimensions, then apply the selected 2D backend.

    ``center`` controls per-batch centering as in ``compute_cca_embedding_from_arrays``.
    """
    with _stage(timer, "time_pre_embedding"):
        cca_embedding, cca_artifacts = compute_cca_embedding_from_arrays(
            x0, x1, embd_dim=cca_dim, return_artifacts=True, center=center
        )
    if cca_embedding.shape[1] < 2:
        raise ValueError(
            "CCA+TSNE needs at least two CCA dimensions before t-SNE, "
            f"but only {cca_embedding.shape[1]} were available."
        )

    tsne_embedding, tsne_artifacts = embed_2d(
        cca_embedding,
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

    n0 = np.asarray(x0).shape[0]
    left_embedding, right_embedding = _split_joint_embedding(tsne_embedding, n0)
    artifacts: dict[str, object] = {
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "cca_embedding": cca_embedding,
        "cca_dim": int(cca_artifacts["embd_dim"]),
        "centered": bool(center),
        **embedder_artifacts(backend, tsne_artifacts),
        "n_jobs": int(n_jobs),
    }
    if n_iter is not None:
        artifacts["n_iter"] = int(n_iter)
    if early_exaggeration_iter is not None:
        artifacts["early_exaggeration_iter"] = int(early_exaggeration_iter)
    return tsne_embedding, artifacts


def _compute_clamped_pca_embedding(
    x: np.ndarray,
    requested_dim: int,
    random_state: int,
    method_name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    x = _validate_matrix(x)
    requested_dim = _validate_positive_integer(requested_dim, f"{method_name} dimension")
    max_dim = min(x.shape[0], x.shape[1])
    used_dim = min(requested_dim, max_dim)
    if used_dim < 2:
        raise ValueError(
            f"{method_name} needs at least two dimensions, but only {used_dim} are available."
        )

    embedding, artifacts = compute_pca_embedding(
        x,
        embd_dim=used_dim,
        random_state=random_state,
        return_artifacts=True,
    )
    artifacts["dim_requested"] = int(requested_dim)
    artifacts["dim_used"] = int(used_dim)
    return embedding, artifacts


def _run_harmony(
    embedding: np.ndarray,
    batch_ids: np.ndarray,
    batch_key: str,
    random_state: int,
    harmony_kwargs: dict[str, object] | None,
) -> np.ndarray:
    embedding = _validate_matrix(embedding, name="embedding")
    batch_ids = np.asarray(batch_ids)
    if batch_ids.shape[0] != embedding.shape[0]:
        raise ValueError(
            f"batch_ids length {batch_ids.shape[0]} does not match embedding rows "
            f"{embedding.shape[0]}."
        )

    metadata = pd.DataFrame({batch_key: batch_ids.astype(str)})
    kwargs = dict(harmony_kwargs or {})
    if "random_state" in kwargs and int(kwargs["random_state"]) != int(random_state):
        raise ValueError(
            "Pass the Harmony seed through random_state, not harmony_kwargs['random_state']."
        )
    kwargs["random_state"] = int(random_state)
    import harmonypy as hm

    harmony_out = hm.run_harmony(embedding, metadata, batch_key, **kwargs)
    corrected = np.asarray(harmony_out.Z_corr, dtype=float)
    if corrected.shape == embedding.shape:
        return corrected
    if corrected.T.shape == embedding.shape:
        return corrected.T
    raise ValueError(
        "Unexpected Harmony output shape: "
        f"got {corrected.shape}, expected {embedding.shape} or {embedding.T.shape}."
    )


def compute_harmony_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    embd_dim: int = 2,
    random_state: int = 0,
    batch_key: str = "batch",
    input_is_pca: bool = False,
    harmony_kwargs: dict[str, object] | None = None,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    joint_data, batch_ids, n0 = _joint_data_and_batches(x0, x1)
    if input_is_pca:
        used_dim = min(int(embd_dim), joint_data.shape[1])
        if used_dim < 2:
            raise ValueError("Harmony needs at least two precomputed PCA dimensions.")
        pca_embedding = np.asarray(joint_data[:, :used_dim], dtype=float)
        pca_artifacts = {
            "components": None,
            "explained_variance": np.asarray([], dtype=float),
            "explained_variance_ratio": np.asarray([], dtype=float),
            "mean": np.asarray([], dtype=float),
            "dim_used": used_dim,
        }
    else:
        pca_embedding, pca_artifacts = _compute_clamped_pca_embedding(
            joint_data,
            requested_dim=embd_dim,
            random_state=random_state,
            method_name="Harmony",
        )
    harmony_embedding = _run_harmony(
        pca_embedding,
        batch_ids=batch_ids,
        batch_key=batch_key,
        random_state=random_state,
        harmony_kwargs=harmony_kwargs,
    )

    if not return_artifacts:
        return harmony_embedding

    left_embedding, right_embedding = _split_joint_embedding(harmony_embedding, n0)
    pca_left_embedding, pca_right_embedding = _split_joint_embedding(pca_embedding, n0)
    artifacts: dict[str, object] = {
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "pca_embedding": pca_embedding,
        "pca_left_embedding": pca_left_embedding,
        "pca_right_embedding": pca_right_embedding,
        "batch_ids": batch_ids,
        "batch_key": str(batch_key),
        "harmony_dim": int(embd_dim),
        "harmony_dim_used": int(pca_artifacts["dim_used"]),
        "input_is_pca": bool(input_is_pca),
        "pca_components": pca_artifacts["components"],
        "pca_explained_variance": pca_artifacts["explained_variance"],
        "pca_explained_variance_ratio": pca_artifacts["explained_variance_ratio"],
        "pca_mean": pca_artifacts["mean"],
        "random_state": int(random_state),
    }
    return harmony_embedding, artifacts


def compute_harmony_then_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    harmony_dim: int = 50,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_jobs: int = 1,
    batch_key: str = "batch",
    input_is_pca: bool = False,
    harmony_kwargs: dict[str, object] | None = None,
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    backend: str = DEFAULT_BACKEND,
    n_neighbors: int | None = None,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    # Canonical sweeps pass the fixed preprocessing PCA and set input_is_pca=True. The fallback
    # remains useful to direct library callers, but it is not used for benchmark datasets.
    with _stage(timer, "time_pre_embedding"):
        harmony_embedding, harmony_artifacts = compute_harmony_embedding_from_arrays(
            x0,
            x1,
            embd_dim=harmony_dim,
            random_state=random_state,
            batch_key=batch_key,
            input_is_pca=input_is_pca,
            harmony_kwargs=harmony_kwargs,
            return_artifacts=True,
        )
    if harmony_embedding.shape[1] < 2:
        raise ValueError(
            "Harmony+TSNE needs at least two Harmony dimensions before t-SNE, "
            f"but only {harmony_embedding.shape[1]} were available."
        )

    tsne_embedding, tsne_artifacts = embed_2d(
        harmony_embedding,
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

    n0 = np.asarray(x0).shape[0]
    left_embedding, right_embedding = _split_joint_embedding(tsne_embedding, n0)
    artifacts: dict[str, object] = {
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "harmony_embedding": harmony_embedding,
        "harmony_left_embedding": harmony_artifacts["left_embedding"],
        "harmony_right_embedding": harmony_artifacts["right_embedding"],
        "pca_embedding": harmony_artifacts["pca_embedding"],
        "pca_left_embedding": harmony_artifacts["pca_left_embedding"],
        "pca_right_embedding": harmony_artifacts["pca_right_embedding"],
        "batch_ids": harmony_artifacts["batch_ids"],
        "batch_key": harmony_artifacts["batch_key"],
        "harmony_dim": int(harmony_dim),
        "harmony_dim_used": int(harmony_artifacts["harmony_dim_used"]),
        "input_is_pca": bool(harmony_artifacts["input_is_pca"]),
        "pca_components": harmony_artifacts["pca_components"],
        "pca_explained_variance": harmony_artifacts["pca_explained_variance"],
        "pca_explained_variance_ratio": harmony_artifacts["pca_explained_variance_ratio"],
        "pca_mean": harmony_artifacts["pca_mean"],
        **embedder_artifacts(backend, tsne_artifacts),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
    }
    if n_iter is not None:
        artifacts["n_iter"] = int(n_iter)
    if early_exaggeration_iter is not None:
        artifacts["early_exaggeration_iter"] = int(early_exaggeration_iter)
    return tsne_embedding, artifacts


def _resolve_torch_device(require_cpu: bool) -> tuple[str, object | None]:
    """Return the resolved scVI device and loaded torch module for synchronized timing.

    ``require_cpu`` rejects automatic accelerator placement to keep CPU runtimes comparable.
    """
    import torch

    if require_cpu:
        return "cpu", torch
    if torch.cuda.is_available():
        return "cuda", torch
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", torch
    return "cpu", torch


def _validate_scvi_counts(x: np.ndarray) -> np.ndarray:
    x = _validate_matrix(x)
    if np.any(~np.isfinite(x)):
        raise ValueError("scVI input contains non-finite values.")
    if np.any(x < 0.0):
        raise ValueError(
            "scVI expects nonnegative count-like input. "
            "Pass raw counts rather than centered, scaled, or PCA-transformed data."
        )
    return x


def compute_scvi_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    latent_dim: int = 2,
    random_state: int = 0,
    max_epochs: int | None = DEFAULT_SCVI_MAX_EPOCHS,
    batch_size: int = DEFAULT_SCVI_BATCH_SIZE,
    batch_key: str = "batch",
    accelerator: str | None = "auto",
    devices: str | int | None = "auto",
    setup_kwargs: dict[str, object] | None = None,
    model_kwargs: dict[str, object] | None = None,
    train_kwargs: dict[str, object] | None = None,
    return_artifacts: bool = False,
    require_cpu: bool = False,
    threads: int | None = None,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    joint_data, batch_ids, n0 = _joint_data_and_batches(x0, x1)
    joint_data = _validate_scvi_counts(joint_data)
    latent_dim = _validate_positive_integer(latent_dim, "latent_dim")

    np.random.seed(int(random_state))
    import scvi
    from anndata import AnnData

    from ne_aligned_utils.timing import pin_torch_threads

    # Set torch's worker count explicitly after its lazy import.
    device, torch_module = _resolve_torch_device(require_cpu)
    if require_cpu and device != "cpu":
        raise RuntimeError(
            f"--require-cpu was set but the resolved torch device is {device!r}. Runtime numbers "
            "from an accelerator are not comparable with the CPU-only benchmark."
        )
    pin_torch_threads(threads)
    if require_cpu:
        accelerator = "cpu"
        devices = "auto"

    try:
        scvi.settings.seed = int(random_state)
    except Exception:
        pass

    obs = pd.DataFrame({batch_key: batch_ids.astype(str)})
    adata = AnnData(X=joint_data, obs=obs)
    setup_options = dict(setup_kwargs or {})
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key, **setup_options)

    model_options = dict(model_kwargs or {})
    model = scvi.model.SCVI(adata, n_latent=latent_dim, **model_options)

    train_options = dict(train_kwargs or {})
    if max_epochs is not None and "max_epochs" not in train_options:
        train_options["max_epochs"] = int(max_epochs)
    if accelerator is not None and "accelerator" not in train_options:
        train_options["accelerator"] = accelerator
    if devices is not None and "devices" not in train_options:
        train_options["devices"] = devices
    if "batch_size" not in train_options:
        train_options["batch_size"] = int(batch_size)
    _synchronize_device(torch_module, device)
    model.train(**train_options)
    _synchronize_device(torch_module, device)

    latent_embedding = np.asarray(model.get_latent_representation(), dtype=float)
    _synchronize_device(torch_module, device)
    if not return_artifacts:
        return latent_embedding

    left_embedding, right_embedding = _split_joint_embedding(latent_embedding, n0)
    artifacts: dict[str, object] = {
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "batch_ids": batch_ids,
        "batch_key": str(batch_key),
        "latent_dim": int(latent_dim),
        "random_state": int(random_state),
        "max_epochs": None if max_epochs is None else int(max_epochs),
        "batch_size": int(train_options["batch_size"]),
        "train_kwargs": train_options,
        # Record the resolved device to avoid pooling incompatible runtime measurements.
        "torch_device": str(device),
    }
    return latent_embedding, artifacts


def compute_scvi_then_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    latent_dim: int = 50,
    random_state: int = 0,
    max_epochs: int | None = DEFAULT_SCVI_MAX_EPOCHS,
    batch_size: int = DEFAULT_SCVI_BATCH_SIZE,
    perplexity: float = 30.0,
    n_jobs: int = 1,
    batch_key: str = "batch",
    accelerator: str | None = "auto",
    devices: str | int | None = "auto",
    setup_kwargs: dict[str, object] | None = None,
    model_kwargs: dict[str, object] | None = None,
    train_kwargs: dict[str, object] | None = None,
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    backend: str = DEFAULT_BACKEND,
    n_neighbors: int | None = None,
    return_artifacts: bool = False,
    require_cpu: bool = False,
    threads: int | None = None,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    with _stage(timer, "time_pre_embedding"):
        scvi_embedding, scvi_artifacts = compute_scvi_embedding_from_arrays(
            x0,
            x1,
            latent_dim=latent_dim,
            random_state=random_state,
            max_epochs=max_epochs,
            batch_size=batch_size,
            batch_key=batch_key,
            accelerator=accelerator,
            devices=devices,
            setup_kwargs=setup_kwargs,
            model_kwargs=model_kwargs,
            train_kwargs=train_kwargs,
            return_artifacts=True,
            require_cpu=require_cpu,
            threads=threads,
        )
    if scvi_embedding.shape[1] < 2:
        raise ValueError(
            "scVI+TSNE needs at least two scVI latent dimensions before t-SNE, "
            f"but only {scvi_embedding.shape[1]} were available."
        )

    tsne_embedding, tsne_artifacts = embed_2d(
        scvi_embedding,
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

    n0 = np.asarray(x0).shape[0]
    left_embedding, right_embedding = _split_joint_embedding(tsne_embedding, n0)
    artifacts: dict[str, object] = {
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "scvi_embedding": scvi_embedding,
        "scvi_left_embedding": scvi_artifacts["left_embedding"],
        "scvi_right_embedding": scvi_artifacts["right_embedding"],
        "batch_ids": scvi_artifacts["batch_ids"],
        "batch_key": scvi_artifacts["batch_key"],
        "latent_dim": int(latent_dim),
        "random_state": int(random_state),
        "max_epochs": scvi_artifacts["max_epochs"],
        "batch_size": scvi_artifacts["batch_size"],
        "train_kwargs": scvi_artifacts["train_kwargs"],
        "torch_device": scvi_artifacts["torch_device"],
        **embedder_artifacts(backend, tsne_artifacts),
        "n_jobs": int(n_jobs),
    }
    if n_iter is not None:
        artifacts["n_iter"] = int(n_iter)
    if early_exaggeration_iter is not None:
        artifacts["early_exaggeration_iter"] = int(early_exaggeration_iter)
    return tsne_embedding, artifacts


def _run_scanorama(
    x0: np.ndarray,
    x1: np.ndarray,
    dimred: int,
    random_state: int,
    scanorama_kwargs: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate batches sharing a feature space into per-batch ``dimred`` coordinates.

    ``dimred`` also sets the alignment dimension; reducing it to two constrains integration.
    Use exact Euclidean neighbor search: the approximate Manhattan path can return
    truncated Annoy results under Python 3.14 and skip alignment.
    """
    x0 = _validate_matrix(x0)
    x1 = _validate_matrix(x1)
    import scanorama

    genes = [str(index) for index in range(x0.shape[1])]
    kwargs = dict(scanorama_kwargs or {})
    kwargs.setdefault("verbose", 0)
    kwargs.setdefault("approx", False)
    integrated, _ = scanorama.integrate(
        [x0, x1],
        [genes, genes],
        dimred=int(dimred),
        seed=int(random_state),
        **kwargs,
    )

    def _densify(block: object) -> np.ndarray:
        if sp.issparse(block):
            block = block.toarray()
        return np.asarray(block, dtype=float)

    return _densify(integrated[0]), _densify(integrated[1])


def compute_scanorama_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    embd_dim: int = 2,
    random_state: int = 0,
    scanorama_kwargs: dict[str, object] | None = None,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Integrate directly into ``embd_dim`` dimensions with Scanorama.

    Two dimensions also constrain Scanorama's internal alignment; use the two-stage
    wrapper for integration in a larger space before visualization.
    """
    x0, x1 = _validate_pair_arrays(x0, x1)
    embd_dim = _validate_positive_integer(embd_dim, "embd_dim")
    left_embedding, right_embedding = _run_scanorama(
        x0, x1, dimred=embd_dim, random_state=random_state, scanorama_kwargs=scanorama_kwargs
    )
    joint_embedding = np.concatenate([left_embedding, right_embedding], axis=0)

    if not return_artifacts:
        return joint_embedding

    artifacts: dict[str, object] = {
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "scanorama_dim": int(embd_dim),
        "scanorama_dim_used": int(left_embedding.shape[1]),
        "random_state": int(random_state),
    }
    return joint_embedding, artifacts


def compute_scanorama_then_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    scanorama_dim: int = 50,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_jobs: int = 1,
    scanorama_kwargs: dict[str, object] | None = None,
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    backend: str = DEFAULT_BACKEND,
    n_neighbors: int | None = None,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Integrate with Scanorama in ``scanorama_dim`` dimensions, then apply the selected 2D backend.
    """
    # Scanorama exposes integration as one call, so pre-embedding timing stays fused.
    with _stage(timer, "time_pre_embedding"):
        scanorama_embedding, scanorama_artifacts = compute_scanorama_embedding_from_arrays(
            x0,
            x1,
            embd_dim=scanorama_dim,
            random_state=random_state,
            scanorama_kwargs=scanorama_kwargs,
            return_artifacts=True,
        )
    if scanorama_embedding.shape[1] < 2:
        raise ValueError(
            "Scanorama+TSNE needs at least two Scanorama dimensions before t-SNE, "
            f"but only {scanorama_embedding.shape[1]} were available."
        )

    tsne_embedding, tsne_artifacts = embed_2d(
        scanorama_embedding,
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

    n0 = np.asarray(x0).shape[0]
    left_embedding, right_embedding = _split_joint_embedding(tsne_embedding, n0)
    artifacts: dict[str, object] = {
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "scanorama_embedding": scanorama_embedding,
        "scanorama_dim": int(scanorama_dim),
        "scanorama_dim_used": int(scanorama_artifacts["scanorama_dim_used"]),
        **embedder_artifacts(backend, tsne_artifacts),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
    }
    if n_iter is not None:
        artifacts["n_iter"] = int(n_iter)
    if early_exaggeration_iter is not None:
        artifacts["early_exaggeration_iter"] = int(early_exaggeration_iter)
    return tsne_embedding, artifacts
