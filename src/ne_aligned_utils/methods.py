from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from openTSNE import TSNE
from openTSNE.affinity import Affinities
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class EffectiveKSplit:
    effective_k: int
    batch0_k: int
    batch1_k: int


def _validate_pair_arrays(x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)

    if x0.ndim != 2 or x1.ndim != 2:
        raise ValueError(
            f"x0 and x1 must both be 2D arrays, got shapes {x0.shape} and {x1.shape}."
        )
    if x0.shape[0] == 0 or x1.shape[0] == 0:
        raise ValueError("x0 and x1 must both contain at least one sample.")
    if x0.shape[1] != x1.shape[1]:
        raise ValueError(
            "x0 and x1 must have the same number of features, "
            f"but got {x0.shape[1]} and {x1.shape[1]}."
        )

    return x0, x1


def _validate_matrix(x: np.ndarray, name: str = "x") -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {x.shape}.")
    if x.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if x.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature.")
    return x


def _validate_batch_sizes(n0: int, n1: int) -> tuple[int, int]:
    n0 = int(n0)
    n1 = int(n1)
    if n0 <= 0 or n1 <= 0:
        raise ValueError(f"n0 and n1 must be positive, got n0={n0}, n1={n1}.")
    return n0, n1


def resolve_effective_k_split(effective_k: int, n0: int, n1: int) -> EffectiveKSplit:
    """Split an exact neighbor budget proportionally across two batches."""
    n0, n1 = _validate_batch_sizes(n0, n1)
    if not float(effective_k).is_integer():
        raise ValueError(f"effective_k must be an integer, got {effective_k}.")
    effective_k = int(effective_k)
    if effective_k <= 0:
        raise ValueError(f"effective_k must be positive, got {effective_k}.")

    capacities = np.array([max(n0 - 1, 0), max(n1 - 1, 0)], dtype=int)
    max_effective_k = int(capacities.sum())
    if effective_k > max_effective_k:
        raise ValueError(
            "effective_k exceeds the available non-self neighbors under a shared "
            f"batch split: got {effective_k}, max is {max_effective_k} for "
            f"n0={n0}, n1={n1}."
        )

    sizes = np.array([n0, n1], dtype=float)
    quotas = effective_k * sizes / float(n0 + n1)
    counts = np.floor(quotas).astype(int)
    counts = np.minimum(counts, capacities)
    remainders = quotas - np.floor(quotas)

    remaining = effective_k - int(counts.sum())
    while remaining > 0:
        eligible = np.flatnonzero(counts < capacities)
        if eligible.size == 0:
            raise ValueError(
                "Unable to allocate effective_k exactly within the available "
                f"batch capacities for n0={n0}, n1={n1}."
            )

        best_remainder = np.max(remainders[eligible])
        tied = eligible[np.isclose(remainders[eligible], best_remainder)]
        if tied.size > 1:
            tied_sizes = np.array([n0, n1], dtype=int)[tied]
            smallest_size = np.min(tied_sizes)
            tied = tied[tied_sizes == smallest_size]
        chosen = int(np.min(tied))
        counts[chosen] += 1
        remaining -= 1

    return EffectiveKSplit(
        effective_k=effective_k,
        batch0_k=int(counts[0]),
        batch1_k=int(counts[1]),
    )


def compute_ot_plan(
    x0: np.ndarray,
    x1: np.ndarray,
    eps_times: float = 1.0,
    epsilon: float | None = None,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    x0, x1 = _validate_pair_arrays(x0, x1)
    import ot

    dist = ot.dist(x0, x1)
    median_squared_distance = float(np.median(dist))
    if epsilon is None:
        eps_times = float(eps_times)
        if eps_times <= 0.0:
            raise ValueError(f"eps_times must be positive, got {eps_times}.")
        epsilon = median_squared_distance / eps_times
    else:
        epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError(
            "The OT regularization epsilon is zero. Check that the two inputs are "
            "not degenerate, or use a smaller eps_times value."
        )

    n0, n1 = x0.shape[0], x1.shape[0]
    a = np.ones(n0, dtype=float) / n0
    b = np.ones(n1, dtype=float) / n1
    plan = ot.sinkhorn(
        a,
        b,
        dist,
        reg=epsilon,
        numItermax=int(max_iter),
        stopThr=float(tol),
    )
    plan = np.asarray(plan, dtype=float)
    if not return_artifacts:
        return plan
    return plan, {
        "cost_matrix": np.asarray(dist, dtype=float),
        "epsilon": float(epsilon),
        "median_squared_distance": float(median_squared_distance),
    }


def _validate_positive_integer(value: int, name: str) -> int:
    if not float(value).is_integer():
        raise ValueError(f"{name} must be an integer, got {value}.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _validate_embedding_dim(embd_dim: int, max_dim: int, method_name: str) -> int:
    embd_dim = _validate_positive_integer(embd_dim, "embd_dim")
    if embd_dim > int(max_dim):
        raise ValueError(f"{method_name} embd_dim={embd_dim} exceeds max dimension {max_dim}.")
    return embd_dim


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


def compute_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_jobs: int = 1,
    pca_components: int | None = 50,
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    x0, x1 = _validate_pair_arrays(x0, x1)
    joint_data = np.concatenate([x0, x1], axis=0)

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

    joint_embedding, tsne_artifacts = compute_tsne_embedding(
        tsne_input,
        random_state=random_state,
        perplexity=perplexity,
        n_jobs=n_jobs,
        n_iter=n_iter,
        early_exaggeration_iter=early_exaggeration_iter,
        return_artifacts=True,
    )

    if not return_artifacts:
        return joint_embedding

    artifacts: dict[str, object] = {
        **tsne_artifacts,
        **pca_artifacts,
        "left_embedding": joint_embedding[: x0.shape[0]],
        "right_embedding": joint_embedding[x0.shape[0] :],
    }
    return joint_embedding, artifacts


def compute_seurat_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    embd_dim: int = 2,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    x0, x1 = _validate_pair_arrays(x0, x1)
    embd_dim = _validate_positive_integer(embd_dim, "embd_dim")

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
    }
    return joint_embedding, artifacts


def _joint_data_and_batches(x0: np.ndarray, x1: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    x0, x1 = _validate_pair_arrays(x0, x1)
    joint_data = np.concatenate([x0, x1], axis=0)
    batch_ids = np.concatenate(
        [
            np.zeros(x0.shape[0], dtype=int),
            np.ones(x1.shape[0], dtype=int),
        ]
    )
    return joint_data, batch_ids, x0.shape[0]


def _split_joint_embedding(embedding: np.ndarray, n0: int) -> tuple[np.ndarray, np.ndarray]:
    embedding = np.asarray(embedding, dtype=float)
    n0 = int(n0)
    return embedding[:n0], embedding[n0:]


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
    harmony_kwargs: dict[str, object] | None,
) -> np.ndarray:
    embedding = _validate_matrix(embedding, name="embedding")
    batch_ids = np.asarray(batch_ids)
    if batch_ids.shape[0] != embedding.shape[0]:
        raise ValueError(
            f"batch_ids length {batch_ids.shape[0]} does not match embedding rows {embedding.shape[0]}."
        )

    metadata = pd.DataFrame({batch_key: batch_ids.astype(str)})
    kwargs = dict(harmony_kwargs or {})
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
    harmony_kwargs: dict[str, object] | None = None,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    joint_data, batch_ids, n0 = _joint_data_and_batches(x0, x1)
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
        "pca_components": pca_artifacts["components"],
        "pca_explained_variance": pca_artifacts["explained_variance"],
        "pca_explained_variance_ratio": pca_artifacts["explained_variance_ratio"],
        "pca_mean": pca_artifacts["mean"],
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
    harmony_kwargs: dict[str, object] | None = None,
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    harmony_embedding, harmony_artifacts = compute_harmony_embedding_from_arrays(
        x0,
        x1,
        embd_dim=harmony_dim,
        random_state=random_state,
        batch_key=batch_key,
        harmony_kwargs=harmony_kwargs,
        return_artifacts=True,
    )
    if harmony_embedding.shape[1] < 2:
        raise ValueError(
            "Harmony+TSNE needs at least two Harmony dimensions before t-SNE, "
            f"but only {harmony_embedding.shape[1]} were available."
        )

    tsne_embedding, tsne_artifacts = compute_tsne_embedding(
        harmony_embedding,
        random_state=random_state,
        perplexity=perplexity,
        n_jobs=n_jobs,
        n_iter=n_iter,
        early_exaggeration_iter=early_exaggeration_iter,
        return_artifacts=True,
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
        "pca_components": harmony_artifacts["pca_components"],
        "pca_explained_variance": harmony_artifacts["pca_explained_variance"],
        "pca_explained_variance_ratio": harmony_artifacts["pca_explained_variance_ratio"],
        "pca_mean": harmony_artifacts["pca_mean"],
        "tsne_perplexity": tsne_artifacts["perplexity"],
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
    }
    if n_iter is not None:
        artifacts["n_iter"] = int(n_iter)
    if early_exaggeration_iter is not None:
        artifacts["early_exaggeration_iter"] = int(early_exaggeration_iter)
    return tsne_embedding, artifacts


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
    max_epochs: int | None = 400,
    batch_key: str = "batch",
    accelerator: str | None = "auto",
    devices: str | int | None = "auto",
    setup_kwargs: dict[str, object] | None = None,
    model_kwargs: dict[str, object] | None = None,
    train_kwargs: dict[str, object] | None = None,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    joint_data, batch_ids, n0 = _joint_data_and_batches(x0, x1)
    joint_data = _validate_scvi_counts(joint_data)
    latent_dim = _validate_positive_integer(latent_dim, "latent_dim")

    np.random.seed(int(random_state))
    import scvi
    from anndata import AnnData

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
    model.train(**train_options)

    latent_embedding = np.asarray(model.get_latent_representation(), dtype=float)
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
        "train_kwargs": train_options,
    }
    return latent_embedding, artifacts


def compute_scvi_then_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    latent_dim: int = 50,
    random_state: int = 0,
    max_epochs: int | None = 400,
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
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    scvi_embedding, scvi_artifacts = compute_scvi_embedding_from_arrays(
        x0,
        x1,
        latent_dim=latent_dim,
        random_state=random_state,
        max_epochs=max_epochs,
        batch_key=batch_key,
        accelerator=accelerator,
        devices=devices,
        setup_kwargs=setup_kwargs,
        model_kwargs=model_kwargs,
        train_kwargs=train_kwargs,
        return_artifacts=True,
    )
    if scvi_embedding.shape[1] < 2:
        raise ValueError(
            "scVI+TSNE needs at least two scVI latent dimensions before t-SNE, "
            f"but only {scvi_embedding.shape[1]} were available."
        )

    tsne_embedding, tsne_artifacts = compute_tsne_embedding(
        scvi_embedding,
        random_state=random_state,
        perplexity=perplexity,
        n_jobs=n_jobs,
        n_iter=n_iter,
        early_exaggeration_iter=early_exaggeration_iter,
        return_artifacts=True,
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
        "train_kwargs": scvi_artifacts["train_kwargs"],
        "tsne_perplexity": tsne_artifacts["perplexity"],
        "n_jobs": int(n_jobs),
    }
    if n_iter is not None:
        artifacts["n_iter"] = int(n_iter)
    if early_exaggeration_iter is not None:
        artifacts["early_exaggeration_iter"] = int(early_exaggeration_iter)
    return tsne_embedding, artifacts


def compute_eot_embedding_from_plan(
    plan: np.ndarray,
    t: float = 0.0,
    embd_dim: int = 2,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2:
        raise ValueError(f"plan must be a 2D array, got shape {plan.shape!r}.")

    t = float(t)
    if not np.isfinite(t):
        raise ValueError(f"t must be finite, got {t}.")

    n0, n1 = plan.shape
    u, singular_values, vt = np.linalg.svd(plan, full_matrices=False)
    start = 1
    max_dim = min(u.shape[1], vt.shape[0]) - start
    if max_dim <= 0:
        raise ValueError(
            "EOT needs at least one non-trivial singular vector after dropping the "
            "leading transport singular mode."
        )

    embd_dim = _validate_positive_integer(embd_dim, "embd_dim")
    q = min(embd_dim, max_dim)
    stop = start + q
    weights = singular_values[start:stop] ** t
    left_embedding = u[:, start:stop] * weights * np.sqrt(n0)
    right_embedding = vt.T[:, start:stop] * weights * np.sqrt(n1)
    joint_embedding = np.concatenate([left_embedding, right_embedding], axis=0)

    if not return_artifacts:
        return joint_embedding

    artifacts: dict[str, object] = {
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        "singular_values": singular_values,
        "t": float(t),
        "embd_dim": int(q),
    }
    return joint_embedding, artifacts


def compute_eot_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    t: float = 0.0,
    eps_times: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    embd_dim: int = 2,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    x0, x1 = _validate_pair_arrays(x0, x1)
    plan, ot_artifacts = compute_ot_plan(
        x0,
        x1,
        eps_times=eps_times,
        tol=tol,
        max_iter=max_iter,
        return_artifacts=True,
    )
    joint_embedding, eot_artifacts = compute_eot_embedding_from_plan(
        plan,
        t=t,
        embd_dim=embd_dim,
        return_artifacts=True,
    )

    if not return_artifacts:
        return joint_embedding

    artifacts: dict[str, object] = {
        **eot_artifacts,
        "ot_plan": plan,
        "epsilon": ot_artifacts["epsilon"],
        "median_squared_distance": ot_artifacts["median_squared_distance"],
    }
    return joint_embedding, artifacts


def compute_eot_then_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    t: float = 0.0,
    eps_times: float = 1.0,
    eot_dim: int = 50,
    random_state: int = 0,
    perplexity: float = 30.0,
    n_jobs: int = 1,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    n_iter: int | None = None,
    early_exaggeration_iter: int | None = None,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    x0, x1 = _validate_pair_arrays(x0, x1)
    eot_embedding, eot_artifacts = compute_eot_embedding_from_arrays(
        x0,
        x1,
        t=t,
        eps_times=eps_times,
        tol=tol,
        max_iter=max_iter,
        embd_dim=eot_dim,
        return_artifacts=True,
    )
    if eot_embedding.shape[1] < 2:
        raise ValueError(
            "EOT+TSNE needs at least two non-trivial EOT dimensions before t-SNE, "
            f"but only {eot_embedding.shape[1]} were available."
        )

    tsne_embedding, tsne_artifacts = compute_tsne_embedding(
        eot_embedding,
        random_state=random_state,
        perplexity=perplexity,
        n_jobs=n_jobs,
        n_iter=n_iter,
        early_exaggeration_iter=early_exaggeration_iter,
        return_artifacts=True,
    )

    if not return_artifacts:
        return tsne_embedding

    artifacts: dict[str, object] = {
        "left_embedding": tsne_embedding[: x0.shape[0]],
        "right_embedding": tsne_embedding[x0.shape[0] :],
        "eot_embedding": eot_embedding,
        "eot_left_embedding": eot_artifacts["left_embedding"],
        "eot_right_embedding": eot_artifacts["right_embedding"],
        "ot_plan": eot_artifacts["ot_plan"],
        "singular_values": eot_artifacts["singular_values"],
        "epsilon": eot_artifacts["epsilon"],
        "median_squared_distance": eot_artifacts["median_squared_distance"],
        "t": eot_artifacts["t"],
        "eot_dim": int(eot_dim),
        "eot_dim_used": int(eot_artifacts["embd_dim"]),
        "tsne_perplexity": tsne_artifacts["perplexity"],
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
    }
    if n_iter is not None:
        artifacts["n_iter"] = int(n_iter)
    if early_exaggeration_iter is not None:
        artifacts["early_exaggeration_iter"] = int(early_exaggeration_iter)
    return tsne_embedding, artifacts


def build_ot_tsne_four_block_matrix(
    plan: np.ndarray,
    return_singular_values: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2:
        raise ValueError(f"plan must be a 2D array, got shape {plan.shape!r}.")

    n0, n1 = plan.shape
    u, singular_values, vt = np.linalg.svd(plan, full_matrices=False)
    eot = np.concatenate([u * np.sqrt(n0), vt.T * np.sqrt(n1)], axis=0)
    matrix = eot @ np.diag(singular_values) @ eot.T / np.sqrt(n0 * n1)
    matrix = np.asarray(matrix, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix[matrix < 0.0] = 0.0
    np.fill_diagonal(matrix, 0.0)
    if return_singular_values:
        return matrix, singular_values
    return matrix


def _row_or_col_topk_mask(
    block: np.ndarray,
    row_k: int,
    col_k: int,
    exclude_diagonal: bool = False,
) -> np.ndarray:
    block = np.asarray(block, dtype=float)
    n_rows, n_cols = block.shape
    row_k = int(row_k)
    col_k = int(col_k)
    if row_k < 0 or col_k < 0:
        raise ValueError(f"row_k and col_k must be nonnegative, got {row_k}, {col_k}.")

    row_capacity = n_cols - 1 if exclude_diagonal else n_cols
    col_capacity = n_rows - 1 if exclude_diagonal else n_rows
    if row_k > row_capacity:
        raise ValueError(f"row_k={row_k} exceeds block row capacity {row_capacity}.")
    if col_k > col_capacity:
        raise ValueError(f"col_k={col_k} exceeds block column capacity {col_capacity}.")

    mask = np.zeros_like(block, dtype=bool)
    scores = np.nan_to_num(block, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    if exclude_diagonal:
        scores = np.array(scores, copy=True)
        np.fill_diagonal(scores, -np.inf)

    if row_k > 0:
        if row_k == n_cols:
            mask[:, :] = True
        else:
            row_idx = np.argpartition(scores, n_cols - row_k, axis=1)[:, -row_k:]
            rows = np.arange(n_rows, dtype=int)[:, None]
            mask[rows, row_idx] = True

    if col_k > 0:
        if col_k == n_rows:
            mask[:, :] = True
        else:
            col_idx = np.argpartition(scores, n_rows - col_k, axis=0)[-col_k:, :]
            cols = np.arange(n_cols, dtype=int)[None, :]
            mask[col_idx, cols] = True

    if exclude_diagonal:
        np.fill_diagonal(mask, False)
    return mask


def sparsify_four_block_top_k(
    matrix: np.ndarray,
    n0: int,
    n1: int,
    effective_k: int,
    binarize: bool = False,
) -> tuple[np.ndarray, EffectiveKSplit]:
    n0, n1 = _validate_batch_sizes(n0, n1)
    matrix = np.asarray(matrix, dtype=float)
    expected_shape = (n0 + n1, n0 + n1)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Expected a square 4-block matrix with shape {expected_shape}, got {matrix.shape}."
        )

    split = resolve_effective_k_split(effective_k=effective_k, n0=n0, n1=n1)
    batch_ks = (split.batch0_k, split.batch1_k)
    slices = (slice(0, n0), slice(n0, n0 + n1))

    mask = np.zeros_like(matrix, dtype=bool)
    for row_batch, row_slice in enumerate(slices):
        for col_batch, col_slice in enumerate(slices):
            block = matrix[row_slice, col_slice]
            block_mask = _row_or_col_topk_mask(
                block,
                row_k=batch_ks[col_batch],
                col_k=batch_ks[row_batch],
                exclude_diagonal=row_batch == col_batch,
            )
            mask[row_slice, col_slice] = block_mask

    np.fill_diagonal(mask, False)
    sparse_matrix = np.where(mask, matrix, 0.0)
    sparse_matrix = np.nan_to_num(sparse_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    sparse_matrix[sparse_matrix < 0.0] = 0.0
    if binarize:
        sparse_matrix = mask.astype(float)
        np.fill_diagonal(sparse_matrix, 0.0)
    return sparse_matrix, split


def sparsify_four_block_top_percentage(
    matrix: np.ndarray,
    n0: int,
    n1: int,
    percentage: float,
    ensure_nonempty: bool = True,
    binarize: bool = False,
) -> np.ndarray:
    n0, n1 = _validate_batch_sizes(n0, n1)
    matrix = np.asarray(matrix, dtype=float)
    expected_shape = (n0 + n1, n0 + n1)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Expected a square 4-block matrix with shape {expected_shape}, got {matrix.shape}."
        )

    percentage = float(percentage)
    if not (0.0 < percentage <= 1.0):
        raise ValueError(f"percentage must be in (0, 1], got {percentage}.")

    mask = np.zeros_like(matrix, dtype=bool)
    slices = (slice(0, n0), slice(n0, n0 + n1))
    for row_slice in slices:
        for col_slice in slices:
            block = matrix[row_slice, col_slice]
            row_k = max(1, int(np.ceil(percentage * block.shape[1])))
            col_k = max(1, int(np.ceil(percentage * block.shape[0])))
            block_mask = _row_or_col_topk_mask(block, row_k=row_k, col_k=col_k)
            if ensure_nonempty:
                empty_rows = np.where(~block_mask.any(axis=1))[0]
                if empty_rows.size:
                    block_mask[empty_rows, np.argmax(block[empty_rows], axis=1)] = True
                empty_cols = np.where(~block_mask.any(axis=0))[0]
                if empty_cols.size:
                    block_mask[np.argmax(block[:, empty_cols], axis=0), empty_cols] = True
            mask[row_slice, col_slice] = block_mask

    np.fill_diagonal(mask, False)
    sparse_matrix = np.where(mask, matrix, 0.0)
    sparse_matrix = np.nan_to_num(sparse_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    sparse_matrix[sparse_matrix < 0.0] = 0.0
    if binarize:
        sparse_matrix = mask.astype(float)
        np.fill_diagonal(sparse_matrix, 0.0)
    return sparse_matrix


def normalize_ot_tsne_graph(matrix: np.ndarray) -> np.ndarray:
    graph = np.asarray(matrix, dtype=float)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError(f"Expected a square 2D graph matrix, got shape {graph.shape}.")

    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph[graph < 0.0] = 0.0
    graph = (graph > 0.0).astype(float)
    graph = np.maximum(graph, graph.T)
    np.fill_diagonal(graph, 0.0)

    total_mass = float(graph.sum())
    if total_mass <= 0.0:
        raise ValueError("Graph has zero total mass after normalization.")
    return graph / total_mass


def run_tsne_from_graph(
    graph: np.ndarray,
    random_state: int = 0,
    n_epochs: int = 200,
    init_scale: float = 10000.0,
) -> np.ndarray:
    graph = np.asarray(graph, dtype=float)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError(f"Expected a square 2D graph matrix, got shape {graph.shape}.")

    n_epochs = int(n_epochs)
    if n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive, got {n_epochs}.")

    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph[graph < 0.0] = 0.0
    np.fill_diagonal(graph, 0.0)

    total_mass = float(graph.sum())
    if total_mass <= 0.0:
        raise ValueError("Graph has zero total mass; cannot run t-SNE.")
    graph = graph / total_mass

    n_points = graph.shape[0]
    init = np.random.RandomState(int(random_state)).randn(n_points, 2)
    init_std = float(init[:, 0].std())
    init = init / (max(init_std, 1e-12) * float(init_scale))
    n_early = n_epochs // 3
    n_late = n_epochs - n_early

    affinities = Affinities()
    affinities.P = sp.csr_matrix(graph)
    embedder = TSNE(
        n_components=2,
        random_state=int(random_state),
        verbose=False,
        early_exaggeration_iter=n_early,
        n_iter=n_late,
    )
    return np.asarray(embedder.fit(affinities=affinities, initialization=init))


def compute_ot_tsne_embedding_from_plan(
    plan: np.ndarray,
    effective_k: int | None = 15,
    percentage: float | None = None,
    binarize: bool = True,
    ensure_nonempty: bool = True,
    random_state: int = 0,
    n_epochs: int = 200,
    init_scale: float = 10000.0,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2:
        raise ValueError(f"plan must be a 2D array, got shape {plan.shape!r}.")

    four_block_matrix, singular_values = build_ot_tsne_four_block_matrix(
        plan,
        return_singular_values=True,
    )
    if percentage is not None:
        if effective_k is not None:
            raise ValueError("Specify either effective_k or percentage, not both.")
        sparse_four_block_matrix = sparsify_four_block_top_percentage(
            four_block_matrix,
            n0=plan.shape[0],
            n1=plan.shape[1],
            percentage=percentage,
            ensure_nonempty=ensure_nonempty,
            binarize=binarize,
        )
        split = None
    else:
        if effective_k is None:
            raise ValueError("effective_k is required when percentage is not provided.")
        sparse_four_block_matrix, split = sparsify_four_block_top_k(
            four_block_matrix,
            n0=plan.shape[0],
            n1=plan.shape[1],
            effective_k=effective_k,
            binarize=binarize,
        )
    graph = normalize_ot_tsne_graph(sparse_four_block_matrix)
    embedding = run_tsne_from_graph(
        graph,
        random_state=random_state,
        n_epochs=n_epochs,
        init_scale=init_scale,
    )
    if not return_artifacts:
        return embedding

    artifacts: dict[str, object] = {
        "four_block_matrix": four_block_matrix,
        "sparse_four_block_matrix": sparse_four_block_matrix,
        "graph": graph,
        "singular_values": singular_values,
        "percentage": None if percentage is None else float(percentage),
    }
    if split is not None:
        artifacts.update(
            effective_k=int(split.effective_k),
            batch0_k=int(split.batch0_k),
            batch1_k=int(split.batch1_k),
        )
    return embedding, artifacts


def compute_ot_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    eps_times: float = 1.0,
    effective_k: int | None = 15,
    percentage: float | None = None,
    binarize: bool = True,
    ensure_nonempty: bool = True,
    random_state: int = 0,
    n_epochs: int = 200,
    tol: float = 1e-12,
    max_iter: int = int(1e4),
    init_scale: float = 10000.0,
    return_artifacts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    x0, x1 = _validate_pair_arrays(x0, x1)
    plan, ot_artifacts = compute_ot_plan(
        x0,
        x1,
        eps_times=eps_times,
        tol=tol,
        max_iter=max_iter,
        return_artifacts=True,
    )
    embedding, method_artifacts = compute_ot_tsne_embedding_from_plan(
        plan,
        effective_k=effective_k,
        percentage=percentage,
        binarize=binarize,
        ensure_nonempty=ensure_nonempty,
        random_state=random_state,
        n_epochs=n_epochs,
        init_scale=init_scale,
        return_artifacts=True,
    )

    if not return_artifacts:
        return embedding

    artifacts: dict[str, object] = {
        **method_artifacts,
        "ot_plan": plan,
        "epsilon": ot_artifacts["epsilon"],
        "median_squared_distance": ot_artifacts["median_squared_distance"],
    }
    return embedding, artifacts


__all__ = [
    "EffectiveKSplit",
    "build_ot_tsne_four_block_matrix",
    "compute_eot_embedding_from_arrays",
    "compute_eot_embedding_from_plan",
    "compute_eot_then_tsne_embedding_from_arrays",
    "compute_harmony_embedding_from_arrays",
    "compute_harmony_then_tsne_embedding_from_arrays",
    "compute_ot_plan",
    "compute_ot_tsne_embedding_from_plan",
    "compute_ot_tsne_embedding_from_arrays",
    "compute_pca_embedding",
    "compute_pca_embedding_from_arrays",
    "compute_scvi_embedding_from_arrays",
    "compute_scvi_then_tsne_embedding_from_arrays",
    "compute_seurat_embedding_from_arrays",
    "compute_tsne_embedding",
    "compute_tsne_embedding_from_arrays",
    "normalize_ot_tsne_graph",
    "resolve_effective_k_split",
    "run_tsne_from_graph",
    "sparsify_four_block_top_k",
    "sparsify_four_block_top_percentage",
]
