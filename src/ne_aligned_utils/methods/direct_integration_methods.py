"""Direct integration methods that map one batch into a reference embedding."""

import time

import numpy as np
import umap
from openTSNE import TSNE, TSNEEmbedding, affinity, initialization
from sklearn.decomposition import PCA

from ne_aligned_utils.config import DEFAULT_BACKEND, DEFAULT_UMAP_N_NEIGHBORS
from ne_aligned_utils.methods.common import (
    stage as _stage,
    validate_pair_arrays as _validate_pair_arrays,
)
from ne_aligned_utils.methods.embedders import _safe_perplexity


def _reference_pca(
    x0: np.ndarray,
    x1: np.ndarray,
    pca_components: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit PCA on ``x0`` and project both reference and query into that basis.

    Return both arrays and PCA artifacts. Skip PCA when the clamped dimension does
    not reduce the feature space.
    """
    if pca_components is None:
        return x0, x1, {"pca_components": None}
    if not float(pca_components).is_integer():
        raise ValueError(f"pca_components must be an integer or None, got {pca_components}.")
    n0, n_features = x0.shape
    pca_dim = min(int(pca_components), n0 - 1, n_features)
    if pca_dim < 2 or pca_dim >= n_features:
        return x0, x1, {"pca_components": None}
    pca = PCA(n_components=pca_dim, random_state=int(random_state))
    reference = np.asarray(pca.fit_transform(x0), dtype=float)
    query = np.asarray(pca.transform(x1), dtype=float)
    return reference, query, {"pca_components": int(pca_dim)}


def compute_ref_tsne_embedding_from_arrays(
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
    """Embed batch 0 as a reference and map batch 1 with openTSNE out-of-sample projection.

    The query cannot move reference coordinates. Fit preprocessing PCA on the reference
    and apply its basis to both batches so projection uses the same feature space.
    """
    x0, x1 = _validate_pair_arrays(x0, x1)
    # Fit reference-only PCA within pre-embedding timing to avoid learning a query-derived basis.
    with _stage(timer, "time_pre_embedding"):
        reference_input, query_input, pca_artifacts = _reference_pca(
            x0, x1, pca_components, random_state
        )

    # Time reference fitting and query projection separately, both within time_embedding.
    embedder_info: dict[str, object]
    _embed_start = time.perf_counter()
    if backend == "tsne":
        perplexity_used = _safe_perplexity(reference_input.shape[0], perplexity)
        tsne_kwargs: dict[str, object] = {
            "n_components": 2,
            "random_state": int(random_state),
            "perplexity": perplexity_used,
            "initialization": "pca",
            "n_jobs": int(n_jobs),
            "verbose": False,
        }
        if n_iter is not None:
            tsne_kwargs["n_iter"] = int(n_iter)
        if early_exaggeration_iter is not None:
            tsne_kwargs["early_exaggeration_iter"] = int(early_exaggeration_iter)

        with _stage(timer, "time_embedding_reference"):
            reference_embedding = TSNE(**tsne_kwargs).fit(reference_input)
        with _stage(timer, "time_embedding_query"):
            query_embedding = reference_embedding.transform(query_input)
        embedder_info = {"perplexity": float(perplexity_used)}
    elif backend == "umap":
        if early_exaggeration_iter is not None:
            raise ValueError(
                "early_exaggeration_iter is a t-SNE parameter and has no UMAP equivalent; "
                "leave it unset when backend='umap'."
            )
        # Fit UMAP only on the reference, then transform the query into its frozen space.
        # Small reference sets can use UMAP's exact-distance transform path.
        requested = DEFAULT_UMAP_N_NEIGHBORS if n_neighbors is None else int(n_neighbors)
        effective = min(requested, max(2, reference_input.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=effective,
            min_dist=0.1,
            metric="euclidean",
            random_state=int(random_state),
            n_jobs=int(n_jobs),
            verbose=False,
            **({"n_epochs": int(n_iter)} if n_iter is not None else {}),
        )
        with _stage(timer, "time_embedding_reference"):
            reference_embedding = reducer.fit_transform(reference_input)
        with _stage(timer, "time_embedding_query"):
            query_embedding = reducer.transform(query_input)
        embedder_info = {
            "umap_n_neighbors": int(effective),
            "umap_n_neighbors_requested": int(requested),
            "umap_min_dist": 0.1,
        }
    else:
        raise ValueError(f"backend must be 'tsne' or 'umap', got {backend!r}.")
    if timer is not None:
        timer.record("time_embedding", time.perf_counter() - _embed_start)

    left_embedding = np.asarray(reference_embedding, dtype=float)
    right_embedding = np.asarray(query_embedding, dtype=float)
    joint_embedding = np.concatenate([left_embedding, right_embedding], axis=0)

    if not return_artifacts:
        return joint_embedding

    artifacts: dict[str, object] = {
        "left_embedding": left_embedding,
        "right_embedding": right_embedding,
        # The reference-space input both arms are handed. Never persisted (it is in
        # UNPERSISTED_ARTIFACT_KEYS), so it costs nothing on disk and lets the
        # controlled-variable test compare the representation byte for byte.
        "reference_input": reference_input,
        "query_input": query_input,
        "embedding_backend": backend,
        **embedder_info,
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "pca_components": pca_artifacts["pca_components"],
    }
    if n_iter is not None:
        artifacts["n_iter"] = int(n_iter)
    if early_exaggeration_iter is not None:
        artifacts["early_exaggeration_iter"] = int(early_exaggeration_iter)
    return joint_embedding, artifacts


def _policar_gene_mask(
    counts: np.ndarray,
    n_genes: int,
    *,
    threshold: float = 0.0,
    at_least: int = 10,
) -> np.ndarray:
    """Poličar et al.'s zero-rate/mean-expression gene selector."""
    counts = np.asarray(counts, dtype=float)
    detected = counts > threshold
    detection_rate = detected.mean(axis=0)
    eligible = detected.sum(axis=0) >= int(at_least)
    if int(eligible.sum()) < int(n_genes):
        raise ValueError(
            f"Poličar selection needs {n_genes} genes detected in at least {at_least} "
            f"reference cells, but only {int(eligible.sum())} qualify."
        )

    mean_expression = np.full(counts.shape[1], np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = np.where(detected, np.log2(counts), np.nan)
        mean_expression[eligible] = np.nanmean(logged[:, eligible], axis=0)
    zero_rate = 1.0 - detection_rate
    zero_rate[~eligible] = np.nan

    offset, lower, upper = 5.0, 0.0, 10.0
    selected = np.zeros(counts.shape[1], dtype=bool)
    for _ in range(100):
        selected[eligible] = zero_rate[eligible] > (
            np.exp(-(mean_expression[eligible] - offset)) + 0.02
        )
        selected_count = int(selected.sum())
        if selected_count == int(n_genes):
            break
        if selected_count < int(n_genes):
            upper = offset
        else:
            lower = offset
        offset = (lower + upper) / 2.0
    return selected


def _policar_reference_pca(
    raw_reference: np.ndarray,
    gene_mask: np.ndarray,
    n_components: int = 50,
    library_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Authors' reference preprocessing: CPM, log1p, z-score, then SVD.

    ``library_sizes`` replaces each cell's total over the supplied genes as the CPM divisor.
    """
    if library_sizes is None:
        library_sizes = raw_reference.sum(axis=1)
    if np.any(library_sizes <= 0):
        raise ValueError("Poličar reference preprocessing requires every cell to have counts.")
    normalized = raw_reference / library_sizes[:, None] * 1_000_000.0
    normalized = np.log1p(normalized)
    means = normalized.mean(axis=0)
    scales = normalized.std(axis=0, ddof=0)
    selected_scales = scales[gene_mask]
    if np.any(selected_scales == 0):
        raise ValueError("Poličar reference genes must have non-zero variance after log1p.")
    selected = (normalized[:, gene_mask] - means[gene_mask]) / selected_scales

    u, singular_values, components = np.linalg.svd(selected, full_matrices=False)
    u[:, components.sum(axis=1) < 0] *= -1
    dimension = min(int(n_components), u.shape[1])
    return np.asarray(u[:, :dimension] * singular_values[:dimension], dtype=float)


def _policar_nonconstant_reference_genes(
    raw_reference: np.ndarray, library_sizes: np.ndarray | None = None
) -> np.ndarray:
    """Genes usable by reference PCA, determined from the reference batch only."""
    if library_sizes is None:
        library_sizes = raw_reference.sum(axis=1)
    if np.any(library_sizes <= 0):
        raise ValueError("Poličar reference preprocessing requires every cell to have counts.")
    normalized = np.log1p(raw_reference / library_sizes[:, None] * 1_000_000.0)
    return normalized.std(axis=0, ddof=0) > 0


def compute_policar_ref_tsne_embedding_from_arrays(
    x0: np.ndarray,
    x1: np.ndarray,
    random_state: int = 3,
    n_jobs: int = 1,
    preselected_features: bool = True,
    return_artifacts: bool = False,
    timer: object | None = None,
    reference_library_sizes: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Poličar et al.'s reference t-SNE procedure.

    ``x0`` and ``x1`` are aligned raw-count matrices. With preselected features, every supplied
    column is used for both the reference PCA and query mapping. Set
    ``preselected_features=False`` for the authors' separate 3,000-gene reference and 1,000-gene
    mapping selections, both learned from ``x0``. ``reference_library_sizes`` (one per ``x0``
    cell) replaces the supplied-gene totals as the reference CPM divisor. Query mapping uses
    cosine distances on raw counts, which no per-cell divisor changes.
    """
    raw_reference, raw_query = _validate_pair_arrays(x0, x1)
    if np.any(raw_reference < 0) or np.any(raw_query < 0):
        raise ValueError("Poličar reference t-SNE requires non-negative raw counts.")
    if reference_library_sizes is not None:
        reference_library_sizes = np.asarray(reference_library_sizes, dtype=float).reshape(-1)
        if reference_library_sizes.shape[0] != raw_reference.shape[0]:
            raise ValueError("reference_library_sizes must have one entry per x0 cell.")

    with _stage(timer, "time_pre_embedding"):
        if preselected_features:
            reference_mask = np.ones(raw_reference.shape[1], dtype=bool)
            mapping_mask = reference_mask.copy()
        else:
            reference_mask = _policar_gene_mask(raw_reference, 3000)
            mapping_mask = _policar_gene_mask(raw_reference, 1000)
        # A common cross-batch HVG panel may contain genes absent or constant in the reference
        # batch. They cannot enter its z-scored PCA. Resolve that from the reference alone and
        # apply the identical usable panel to query mapping, preserving reference semantics.
        nonconstant = _policar_nonconstant_reference_genes(raw_reference, reference_library_sizes)
        dropped_zero_variance = int((reference_mask & ~nonconstant).sum())
        reference_mask &= nonconstant
        mapping_mask &= nonconstant
        reference_pca = _policar_reference_pca(
            raw_reference, reference_mask, library_sizes=reference_library_sizes
        )

    multiscale = [
        _safe_perplexity(raw_reference.shape[0], 50.0),
        _safe_perplexity(raw_reference.shape[0], 500.0),
    ]
    multiscale = sorted(set(multiscale))
    embed_start = time.perf_counter()
    with _stage(timer, "time_embedding_reference"):
        reference_affinities = affinity.Multiscale(
            reference_pca,
            perplexities=multiscale,
            metric="cosine",
            n_jobs=int(n_jobs),
            random_state=int(random_state),
            verbose=False,
        )
        reference_init = initialization.pca(reference_pca, random_state=int(random_state))
        reference_embedding = TSNEEmbedding(
            reference_init,
            reference_affinities,
            negative_gradient_method="fft",
            n_jobs=int(n_jobs),
            random_state=int(random_state),
            verbose=False,
        )
        reference_embedding.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
        reference_embedding.optimize(n_iter=500, exaggeration=1, momentum=0.8, inplace=True)

    with _stage(timer, "time_embedding_query"):
        mapping_perplexity = _safe_perplexity(raw_reference.shape[0], 30.0)
        mapping_affinities = affinity.PerplexityBasedNN(
            raw_reference[:, mapping_mask],
            perplexity=mapping_perplexity,
            metric="cosine",
            n_jobs=int(n_jobs),
            random_state=int(random_state),
            verbose=False,
        )
        mapping_reference = TSNEEmbedding(
            np.asarray(reference_embedding),
            mapping_affinities,
            negative_gradient_method="fft",
            n_jobs=int(n_jobs),
            random_state=int(random_state),
            verbose=False,
        )
        query_embedding = mapping_reference.prepare_partial(
            raw_query[:, mapping_mask], initialization="median", k=10
        )
        query_embedding.optimize(250, learning_rate=0.1, momentum=0.8, inplace=True)
    if timer is not None:
        timer.record("time_embedding", time.perf_counter() - embed_start)

    left = np.asarray(reference_embedding, dtype=float)
    right = np.asarray(query_embedding, dtype=float)
    joint = np.concatenate([left, right], axis=0)
    if not return_artifacts:
        return joint
    return joint, {
        "left_embedding": left,
        "right_embedding": right,
        "preselected_features": bool(preselected_features),
        "reference_gene_count": int(reference_mask.sum()),
        "mapping_gene_count": int(mapping_mask.sum()),
        "dropped_zero_variance_gene_count": dropped_zero_variance,
        "reference_pca_components": int(reference_pca.shape[1]),
        "reference_perplexities": np.asarray(multiscale, dtype=float),
        "mapping_perplexity": float(mapping_perplexity),
        "mapping_initialization": "median",
        "mapping_k": 10,
        "random_state": int(random_state),
        "initialization_random_state": int(random_state),
        "n_jobs": int(n_jobs),
        "library_size_source": "selected_genes" if reference_library_sizes is None else "all_genes",
    }
