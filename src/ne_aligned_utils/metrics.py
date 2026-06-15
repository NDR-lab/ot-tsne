import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from scipy.stats import chi2

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples


# performance metrics


def _validate_positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    try:
        value_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.") from exc
    if value_int != value or value_int < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value_int


def _compute_self_knn_indices(embd, metadata, k, metadata_name="metadata"):
    """
    Validate generic kNN metric inputs and return self-excluded neighbor indices.

    The helper requests ``effective_k + 1`` neighbors so each point can drop its own
    index explicitly before scoring. If fewer than two samples are provided, it returns
    an empty neighbor matrix and effective_k = 0.
    """
    embd = np.asarray(embd, dtype=float)
    metadata = np.asarray(metadata)

    if embd.ndim != 2:
        raise ValueError(f"embd must be a 2D array, got shape {embd.shape!r}.")
    if metadata.ndim != 1:
        raise ValueError(f"{metadata_name} must be a 1D array, got shape {metadata.shape!r}.")
    if embd.shape[0] != metadata.shape[0]:
        raise ValueError(
            f"embd and {metadata_name} must have the same number of samples, "
            f"got {embd.shape[0]} and {metadata.shape[0]}."
        )

    k = _validate_positive_integer(k, "k")

    n_samples = embd.shape[0]
    effective_k = min(k, max(n_samples - 1, 0))
    if effective_k == 0:
        return metadata, np.empty((n_samples, 0), dtype=int), effective_k

    n_neighbors = min(n_samples, effective_k + 1)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(embd)
    candidate_indices = nbrs.kneighbors(embd, return_distance=False)

    neighbor_indices = np.empty((n_samples, effective_k), dtype=int)
    for row_idx, row_neighbors in enumerate(candidate_indices):
        filtered_neighbors = row_neighbors[row_neighbors != row_idx]
        if filtered_neighbors.shape[0] < effective_k:
            raise RuntimeError(
                "Could not exclude the query point from its own neighborhood. "
                f"Row {row_idx} only retained {filtered_neighbors.shape[0]} neighbors."
            )
        neighbor_indices[row_idx] = filtered_neighbors[:effective_k]

    return metadata, neighbor_indices, effective_k


def _validate_embedding_and_labels(embd, labels, labels_name="labels"):
    embd = np.asarray(embd, dtype=float)
    labels = np.asarray(labels)

    if embd.ndim != 2:
        raise ValueError(f"embd must be a 2D array, got shape {embd.shape!r}.")
    if labels.ndim != 1:
        raise ValueError(f"{labels_name} must be a 1D array, got shape {labels.shape!r}.")
    if embd.shape[0] != labels.shape[0]:
        raise ValueError(
            f"embd and {labels_name} must have the same number of samples, "
            f"got {embd.shape[0]} and {labels.shape[0]}."
        )

    return embd, labels


def _compute_neighbor_purity_scores(embd, labels, k):
    """
    Compute per-label fixed-k neighborhood purity scores.

    Labels with fewer than 5 cells are skipped as query labels. For each remaining label, the
    neighborhood size is ``min(label_size - 1, 15)`` and excludes the query cell. The returned
    scores are label-level means, intended to be macro-averaged by ``compute_neighbor_purity``.
    """
    embd, labels = _validate_embedding_and_labels(embd, labels, labels_name="labels")
    _validate_positive_integer(k, "k")

    n_samples = embd.shape[0]
    if n_samples == 0:
        return np.empty(0, dtype=float)

    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    eligible_label_ids = np.flatnonzero(counts >= 5)
    if eligible_label_ids.size == 0:
        return np.empty(0, dtype=float)

    max_neighbors = int(min(15, counts[eligible_label_ids].max(), n_samples))
    knn = NearestNeighbors(
        n_neighbors=min(n_samples, max_neighbors + 1),
        algorithm="auto",
    ).fit(embd)
    candidate_indices = knn.kneighbors(embd, return_distance=False)

    label_scores = np.empty(eligible_label_ids.size, dtype=float)
    for label_pos, label_id in enumerate(eligible_label_ids):
        query_indices = np.flatnonzero(inverse == label_id)
        label_k = int(min(query_indices.size - 1, 15))
        cell_scores = np.empty(query_indices.size, dtype=float)

        for cell_pos, query_idx in enumerate(query_indices):
            other_indices = candidate_indices[query_idx][
                candidate_indices[query_idx] != query_idx
            ]
            if other_indices.size < label_k:
                raise RuntimeError(
                    "Could not build a fixed-k self-excluded neighborhood. "
                    f"Row {query_idx} only retained {other_indices.size} other neighbors."
                )
            neighbor_indices = other_indices[:label_k]
            cell_scores[cell_pos] = float(np.mean(inverse[neighbor_indices] == label_id))

        label_scores[label_pos] = float(np.mean(cell_scores))

    return label_scores


def compute_neighbor_purity(embd, labels, k=50):
    """
    Compute macro-averaged fixed-k neighborhood purity across labels.

    Labels with fewer than 5 cells are skipped. For each remaining label, every cell is scored by
    the fraction of its ``min(label_size - 1, 15)`` nearest other points that share its
    label. The final score is the unweighted mean of those per-label means. The ``k`` argument is
    retained for backward compatibility and validation, but the neighborhood size is fixed by the
    label size and cap above.

    :param embd: embedding coordinates with shape (n_samples, n_dims)
    :param labels: 1D cluster or cell-type label vector with length n_samples
    :param k: deprecated; retained for backward compatibility
    :return: Macro-averaged neighbor purity score in [0, 1]
    """
    purity_scores = _compute_neighbor_purity_scores(embd, labels, k)
    if purity_scores.size == 0:
        return 0.0
    return float(np.mean(purity_scores))


def compute_knn_same_cell_type_fraction(embd, labels, k=50):
    """
    Backward-compatible alias for neighbor purity using cell-type labels.

    Despite the historical function name, this now matches ``compute_neighbor_purity``: labels with
    fewer than 5 cells are skipped, neighborhoods are self-excluded fixed-k with a cap of 15, and
    the final mean is macro-averaged over labels.

    :param embd: embedding coordinates with shape (n_samples, n_dims)
    :param labels: 1D cell-type label vector with length n_samples
    :param k: deprecated; retained for backward compatibility
    :return: Macro-averaged neighbor purity score in [0, 1]
    """
    return compute_neighbor_purity(embd, labels, k=k)


def compute_silhouette_index(embd, labels):
    """
    Compute the mean Silhouette index with respect to the provided labels.

    The score is averaged across all samples, matching the per-embedding quantity used in the
    paper before any aggregation across embedding dimensions.

    :param embd: embedding coordinates with shape (n_samples, n_dims)
    :param labels: 1D cluster or cell-type label vector with length n_samples
    :return: Mean Silhouette index in [-1, 1]
    """
    embd, labels = _validate_embedding_and_labels(embd, labels, labels_name="labels")

    n_samples = embd.shape[0]
    if n_samples < 2:
        return 0.0

    unique_labels = np.unique(labels)
    if unique_labels.size < 2 or unique_labels.size >= n_samples:
        return 0.0

    scores = silhouette_samples(embd, labels, metric="euclidean")
    return float(np.mean(scores))


def compute_silhouette_score(embd, labels):
    """
    Compute the benchmark paper's cell-type average silhouette width (ASW).

    The paper first computes the classical mean silhouette width with respect to
    cell identity labels, then rescales it from ``[-1, 1]`` to ``[0, 1]`` via
    ``(ASW + 1) / 2`` so that larger values indicate better label conservation.

    :param embd: embedding coordinates with shape (n_samples, n_dims)
    :param labels: 1D cell identity label vector with length n_samples
    :return: Scaled cell-type ASW in [0, 1]
    """
    return float((compute_silhouette_index(embd, labels) + 1.0) / 2.0)


def _validate_embedding_batch_labels(embd, batch_ids, labels=None):
    embd = np.asarray(embd, dtype=float)
    batch_ids = np.asarray(batch_ids)

    if embd.ndim != 2:
        raise ValueError(f"embd must be a 2D array, got shape {embd.shape!r}.")
    if batch_ids.ndim != 1:
        raise ValueError(f"batch_ids must be a 1D array, got shape {batch_ids.shape!r}.")
    if embd.shape[0] != batch_ids.shape[0]:
        raise ValueError(
            "embd and batch_ids must have the same number of samples, "
            f"got {embd.shape[0]} and {batch_ids.shape[0]}."
        )

    if labels is None:
        labels = np.zeros(embd.shape[0], dtype=int)
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be a 1D array, got shape {labels.shape!r}.")
    if embd.shape[0] != labels.shape[0]:
        raise ValueError(
            "embd and labels must have the same number of samples, "
            f"got {embd.shape[0]} and {labels.shape[0]}."
        )

    return embd, batch_ids, labels


def _encode_labels(values):
    _, encoded = np.unique(np.asarray(values), return_inverse=True)
    return encoded.astype(int)


def _self_excluded_knn(embd, k, return_distance=False):
    embd = np.asarray(embd, dtype=float)
    n_samples = embd.shape[0]
    effective_k = min(int(k), max(n_samples - 1, 0))
    if effective_k == 0:
        empty_idx = np.empty((n_samples, 0), dtype=int)
        if return_distance:
            return np.empty((n_samples, 0), dtype=float), empty_idx, effective_k
        return empty_idx, effective_k

    nbrs = NearestNeighbors(n_neighbors=effective_k + 1, algorithm="auto").fit(embd)
    distances, candidate_indices = nbrs.kneighbors(embd, return_distance=True)

    neighbor_indices = np.empty((n_samples, effective_k), dtype=int)
    neighbor_distances = np.empty((n_samples, effective_k), dtype=float)
    for row_idx, (row_distances, row_neighbors) in enumerate(
        zip(distances, candidate_indices)
    ):
        keep = row_neighbors != row_idx
        filtered_neighbors = row_neighbors[keep]
        filtered_distances = row_distances[keep]
        if filtered_neighbors.shape[0] < effective_k:
            raise RuntimeError(
                "Could not exclude the query point from its own neighborhood. "
                f"Row {row_idx} only retained {filtered_neighbors.shape[0]} neighbors."
            )
        neighbor_indices[row_idx] = filtered_neighbors[:effective_k]
        neighbor_distances[row_idx] = filtered_distances[:effective_k]

    if return_distance:
        return neighbor_distances, neighbor_indices, effective_k
    return neighbor_indices, effective_k


def _lisi_probabilities(distances, perplexity, tol=1e-5, max_iter=50):
    distances = np.asarray(distances, dtype=float)
    beta = 1.0
    beta_min = -np.inf
    beta_max = np.inf
    log_perplexity = np.log(float(perplexity))

    for _ in range(max_iter + 1):
        weights = np.exp(-distances * beta)
        weight_sum = float(weights.sum())
        if weight_sum == 0.0:
            return np.zeros_like(distances), 0.0

        probabilities = weights / weight_sum
        entropy = (
            np.log(weight_sum)
            + beta * float(np.dot(distances, weights)) / weight_sum
        )
        entropy_diff = entropy - log_perplexity
        if abs(entropy_diff) <= tol:
            return probabilities, entropy

        if entropy_diff > 0:
            beta_min = beta
            beta = beta * 2.0 if beta_max == np.inf else (beta + beta_max) / 2.0
        else:
            beta_max = beta
            beta = beta / 2.0 if beta_min == -np.inf else (beta + beta_min) / 2.0

    return probabilities, entropy


def compute_ilisi(embd, batch_ids, k=90, perplexity=None, scale=True):
    """
    Compute the benchmark paper's integration LISI (iLISI) score.

    This follows the paper's embedding-output setup: use the 90 nearest neighbors
    by default, set perplexity to one third of the neighborhood size, compute the
    inverse Simpson index for each cell's batch composition, take the median
    across cells, and rescale with ``(iLISI - 1) / (B - 1)`` for ``B`` batches.

    :param embd: embedding coordinates with shape (n_samples, n_dims)
    :param batch_ids: 1D batch-id vector with length n_samples
    :param k: number of nearest neighbors; the paper uses k=90
    :param perplexity: effective neighborhood size; defaults to floor(k / 3)
    :param scale: whether to rescale the median iLISI to [0, 1]
    :return: Median iLISI, scaled to [0, 1] by default
    """
    embd, batch_ids, _ = _validate_embedding_batch_labels(embd, batch_ids)
    k = _validate_positive_integer(k, "k")

    batch_codes = _encode_labels(batch_ids)
    n_batches = int(np.unique(batch_codes).size)
    if n_batches < 2:
        return float("nan")

    distances, neighbor_indices, effective_k = _self_excluded_knn(
        embd,
        k,
        return_distance=True,
    )
    if effective_k == 0:
        return float("nan")

    if perplexity is None:
        perplexity = max(1.0, float(np.floor(effective_k / 3.0)))
    else:
        perplexity = float(perplexity)
        if not np.isfinite(perplexity) or perplexity <= 0:
            raise ValueError(f"perplexity must be positive, got {perplexity!r}.")
    perplexity = min(perplexity, float(effective_k))

    lisi_scores = np.empty(embd.shape[0], dtype=float)
    for row_idx in range(embd.shape[0]):
        probabilities, entropy = _lisi_probabilities(distances[row_idx], perplexity)
        if entropy == 0.0:
            lisi_scores[row_idx] = np.nan
            continue
        batch_probabilities = np.bincount(
            batch_codes[neighbor_indices[row_idx]],
            weights=probabilities,
            minlength=n_batches,
        )
        simpson = float(np.dot(batch_probabilities, batch_probabilities))
        lisi_scores[row_idx] = 1.0 / simpson if simpson > 0.0 else np.nan

    ilisi = float(np.nanmedian(lisi_scores))
    if scale:
        ilisi = (ilisi - 1.0) / (n_batches - 1.0)
        ilisi = float(np.clip(ilisi, 0.0, 1.0))
    return ilisi


def _kbet_acceptance_rate(
    neighbor_batch_codes,
    expected_batch_codes,
    n_batches,
    alpha,
):
    expected_counts = np.bincount(expected_batch_codes, minlength=n_batches).astype(
        float
    )
    expected_freq = expected_counts / expected_counts.sum()
    expected_neighbor_counts = expected_freq * neighbor_batch_codes.shape[1]

    valid = expected_neighbor_counts > 0
    if valid.sum() < 2:
        return float("nan")

    observed = np.vstack(
        [
            np.bincount(row, minlength=n_batches).astype(float)
            for row in neighbor_batch_codes
        ]
    )
    statistics = np.sum(
        (observed[:, valid] - expected_neighbor_counts[valid]) ** 2
        / expected_neighbor_counts[valid],
        axis=1,
    )
    p_values = chi2.sf(statistics, df=int(valid.sum() - 1))
    return float(np.mean(p_values >= alpha))


def compute_kbet(
    embd,
    batch_ids,
    labels=None,
    alpha=0.05,
    k=15,
    min_cells_per_batch=45,
    min_label_size=10,
):
    """
    Compute a label-wise kBET batch-removal score.

    For each cell identity label, the batch composition in each cell's local
    neighborhood is compared with the label's global batch composition using the
    kBET chi-square test. A fixed neighborhood size is used for all labels.
    Labels are scored only when every batch has at least ``min_cells_per_batch``
    cells for that label. The label score is the kBET acceptance rate. The final
    score is the mean label score, so larger values indicate better batch mixing.

    :param embd: embedding coordinates with shape (n_samples, n_dims)
    :param batch_ids: 1D batch-id vector with length n_samples
    :param labels: 1D cell identity label vector; if omitted, all cells are one label
    :param alpha: chi-square test significance level
    :param k: fixed number of nearest neighbors used for each retained label
    :param min_cells_per_batch: labels with fewer cells in any batch are skipped
    :param min_label_size: labels smaller than this are skipped; scIB uses 10
    :return: Mean label-wise kBET acceptance score in [0, 1]
    """
    embd, batch_ids, labels = _validate_embedding_batch_labels(embd, batch_ids, labels)
    k = _validate_positive_integer(k, "k")
    min_cells_per_batch = _validate_positive_integer(
        min_cells_per_batch,
        "min_cells_per_batch",
    )
    min_label_size = _validate_positive_integer(min_label_size, "min_label_size")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}.")

    batch_codes = _encode_labels(batch_ids)
    n_batches = int(np.unique(batch_codes).size)
    if n_batches < 2:
        return float("nan")

    label_scores = []
    for label in np.unique(labels):
        label_idx = np.flatnonzero(labels == label)
        n_label = int(label_idx.size)
        label_batch_codes = batch_codes[label_idx]

        label_batch_counts = np.bincount(label_batch_codes, minlength=n_batches)
        if (
            n_label < min_label_size
            or np.any(label_batch_counts < min_cells_per_batch)
        ):
            continue

        _, label_neighbors, _ = _self_excluded_knn(
            embd[label_idx],
            min(k, n_label - 1),
            return_distance=True,
        )
        neighbor_batch_codes = label_batch_codes[label_neighbors]
        label_scores.append(
            _kbet_acceptance_rate(
                neighbor_batch_codes,
                label_batch_codes,
                n_batches=n_batches,
                alpha=float(alpha),
            )
        )

    if not label_scores:
        return float("nan")
    return float(np.nanmean(label_scores))


def _center_other_batch_fraction_score(q, p):
    q = np.asarray(q, dtype=float)
    if q.ndim != 1:
        raise ValueError(f"q must be a 1D array, got shape {q.shape!r}.")
    if not (0.0 < float(p) < 1.0):
        raise ValueError(f"p must lie in (0, 1), got {p!r}.")

    score = np.empty_like(q, dtype=float)
    lower = q <= p
    score[lower] = 0.5 * (q[lower] / p)
    score[~lower] = 0.5 + 0.5 * ((q[~lower] - p) / (1.0 - p))
    return np.clip(score, 0.0, 1.0)


def compute_knn_other_batch_fraction(
    embd,
    batch_ids,
    labels,
    k=50,
    min_cells_per_batch=10,
):
    """
    Score batch mixing within shared cell types, accounting for both under- and over-mixing.

    For each cell type represented in both batches with at least ``min_cells_per_batch`` cells
    per batch, the metric builds a kNN graph inside that cell type only. Each cell's fraction of
    cross-batch neighbors is compared against the expected fraction under ideal overlap given the
    batch sizes within that cell type. The per-cell score is mapped to ``[0, 1]`` with:

    - ``0`` meaning complete under-mixing,
    - ``0.5`` meaning the expected amount of mixing,
    - ``1`` meaning maximal over-mixing.

    The final score averages over batches within each shared cell type, then averages over the
    retained shared cell types.

    :param embd: embedding coordinates with shape (n_samples, n_dims)
    :param batch_ids: 1D batch-id vector with length n_samples
    :param labels: 1D cell-type label vector with length n_samples
    :param k: maximum number of nearest neighbors considered per retained cell type
    :param min_cells_per_batch: minimum support required in each batch for a cell type to be scored
    :return: Mean shared-cell-type batch-mixing score in [0, 1], or NaN if no shared cell type
        passes the support threshold
    """
    embd = np.asarray(embd, dtype=float)
    batch_ids = np.asarray(batch_ids)
    labels = np.asarray(labels)

    if embd.ndim != 2:
        raise ValueError(f"embd must be a 2D array, got shape {embd.shape!r}.")
    if batch_ids.ndim != 1:
        raise ValueError(f"batch_ids must be a 1D array, got shape {batch_ids.shape!r}.")
    if labels.ndim != 1:
        raise ValueError(f"labels must be a 1D array, got shape {labels.shape!r}.")
    if embd.shape[0] != batch_ids.shape[0] or embd.shape[0] != labels.shape[0]:
        raise ValueError(
            "embd, batch_ids, and labels must have the same number of samples, "
            f"got {embd.shape[0]}, {batch_ids.shape[0]}, and {labels.shape[0]}."
        )

    k = _validate_positive_integer(k, "k")
    min_cells_per_batch = _validate_positive_integer(
        min_cells_per_batch,
        "min_cells_per_batch",
    )

    unique_batches = np.unique(batch_ids)
    if unique_batches.shape[0] != 2:
        raise ValueError(
            "compute_knn_other_batch_fraction expects exactly two batches, "
            f"got {unique_batches.shape[0]} unique batch ids."
        )
    batch0, batch1 = unique_batches

    shared_label_scores = []

    for label in np.unique(labels):
        label_mask = labels == label
        idx = np.flatnonzero(label_mask)
        batch_ids_label = batch_ids[idx]

        idx0 = idx[batch_ids_label == batch0]
        idx1 = idx[batch_ids_label == batch1]
        n0_c = int(idx0.size)
        n1_c = int(idx1.size)

        if min(n0_c, n1_c) < min_cells_per_batch:
            continue

        idx_c = np.concatenate([idx0, idx1])
        embd_c = embd[idx_c]
        batch_c = batch_ids[idx_c]

        k_c = min(int(k), embd_c.shape[0] - 1)
        if k_c < 1:
            continue

        nbrs = NearestNeighbors(n_neighbors=k_c + 1, algorithm="auto").fit(embd_c)
        candidate_indices = nbrs.kneighbors(embd_c, return_distance=False)
        neighbor_indices = np.empty((embd_c.shape[0], k_c), dtype=int)
        for row_idx, row_neighbors in enumerate(candidate_indices):
            filtered_neighbors = row_neighbors[row_neighbors != row_idx]
            if filtered_neighbors.shape[0] < k_c:
                raise RuntimeError(
                    "Could not exclude the query point from its own neighborhood. "
                    f"Row {row_idx} only retained {filtered_neighbors.shape[0]} neighbors."
                )
            neighbor_indices[row_idx] = filtered_neighbors[:k_c]

        other_batch_fraction = (
            batch_c[neighbor_indices] != batch_c[:, None]
        ).mean(axis=1)

        denom = n0_c + n1_c - 1
        p0_c = n1_c / denom
        p1_c = n0_c / denom

        batch0_mask = batch_c == batch0
        batch1_mask = batch_c == batch1
        batch0_score = _center_other_batch_fraction_score(
            other_batch_fraction[batch0_mask],
            p0_c,
        ).mean()
        batch1_score = _center_other_batch_fraction_score(
            other_batch_fraction[batch1_mask],
            p1_c,
        ).mean()

        shared_label_scores.append(0.5 * (batch0_score + batch1_score))

    if not shared_label_scores:
        return float("nan")

    return float(np.mean(shared_label_scores))



def intersection_over_union(graphs):
    """
    Computes the intersection over union (IoU) metric between two sparse matrices. IoU is taken row-wise and averaged
     over all rows.
    :param graphs: list of two sparse matrices
    :return: IoU metric
    """
    assert len(graphs) == 2, "Only two graphs supported"
    intersection = graphs[0].multiply(graphs[1])
    intersection = np.array(intersection.sum(1)).flatten()

    union = graphs[0].maximum(graphs[1])
    union = np.array(union.sum(1)).flatten()
    union[union == 0] = 1  # avoid division by zero

    iou = intersection / union
    return iou.mean()

def compute_concordance(x, y, k=50):
    """
    Computes the concordance metric between two datasets x and y. The concordance is defined as the share of k nearest neighbors of each point in x that are also among the k nearest neighbors of the corresponding point in y.
    :param x: array of positions for first (clearn / highdimensiona) dataset
    :param y: array of positions for second (embedding) dataset
    :param k: number of nearest neighbors considered
    :return: Share of x's k nearest neighbors that are also y's k nearest neighbors
    """
    assert x.shape[0] == y.shape[0], "x and y must have the same number of points"
    nbrs_x = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x)
    _, indices_x = nbrs_x.kneighbors(x)
    nbrs_y = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(y)
    _, indices_y = nbrs_y.kneighbors(y)

    # create sparse knn graph matrices
    n = x.shape[0]
    data = np.ones(n * k)
    knn_graphs = []
    rows = np.arange(n).repeat(k)
    for indices in [indices_x, indices_y]:
        cols = indices.flatten()
        knn_graphs.append(coo_matrix((data, (rows, cols)), shape=(n, n)))

    return intersection_over_union(knn_graphs)


def compute_cross_concordance(x0, x1, y0, y1, k=50):
    """
    Computes the cross-concordance metric between two datasets (x0, x1) and (y0, y1). The cross-concordance is defined
    as the share of k nearest neighbors in x1 of each point in x0 and vice versa that are also among the k nearest
    neighbors in y1 of the corresponding point in y0, and vice versa. The denominator is the total number of neighbors
    considered in x and y jointly (between k and 2k many).
    :param x0: array of positions of the first part of the first (original / highdimensional) dataset
    :param x1: array of positions of the second part of the first (original / highdimensional) dataset
    :param y0: array of positions of the first part of the second (embedded) dataset
    :param y1: array of positions of the second part of the second (embedded) dataset
    :param k: number of nearest neighbors considered
    :return: Share of x's k nearest neighbors that are also y's k nearest neighbors
    """
    assert x0.shape[0] == y0.shape[0], f"x0 and y0 must have the same number of points, but have {x0.shape[0]} and {y0.shape[0]}"
    assert x1.shape[0] == y1.shape[0], f"x1 and y1 must have the same number of points, but have {x1.shape[0]} and {y1.shape[0]}"
    assert x0.shape[1] == x1.shape[1], f"x0 and x1 must have the same number of dimensions, but have {x0.shape[1]} and {x1.shape[1]}"
    assert y0.shape[1] == y1.shape[1], f"y0 and y1 must have the same number of dimensions, but have {y0.shape[1]} and {y1.shape[1]}"

    # create sparse knn graph matrices
    n0 = x0.shape[0]
    n1 = x1.shape[0]

    nbrs_x0 = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x1)
    _, indices_x0 = nbrs_x0.kneighbors(x0)

    nbrs_x1 = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x0)
    _, indices_x1 = nbrs_x1.kneighbors(x1)

    indices_x = np.concatenate((indices_x0+n0, indices_x1), axis=0)

    nbrs_y0 = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(y1)
    _, indices_y0 = nbrs_y0.kneighbors(y0)

    nbrs_y1 = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(y0)
    _, indices_y1 = nbrs_y1.kneighbors(y1)

    indices_y = np.concatenate((indices_y0+n0, indices_y1), axis=0)


    data = np.ones(n0 * k+ n1 * k)

    knn_graphs = []
    rows = np.concatenate([np.arange(n0).repeat(k), np.arange(n1).repeat(k)+n0])
    for indices in [indices_x, indices_y]:
        cols = indices.flatten()
        knn_graphs.append(coo_matrix((data, (rows, cols)), shape=(n0 + n1, n0 + n1)))

    return intersection_over_union(knn_graphs)


def compute_foscttm(x0_ref, x1_ref, y0, y1):
    """
    Computes FOSCTTM (Fraction of Samples Closer Than the True Match)
    with nearest-neighbor matching.
    For each point in one modality, the true match in the other modality
    is defined as nearest neighbor in latent space. The score is computed
    in embedding space in both directions and averaged symmetrically.

    :param x0_ref: latent/clean positions of modality 0 (n0 x d0)
    :param x1_ref: latent/clean positions of modality 1 (n1 x d0)
    :param y0: embedded positions of modality 0 (n0 x d1)
    :param y1: embedded positions of modality 1 (n1 x d1)
    :return: Symmetric FOSCTTM score (lower is better)
    """

    scores = []
    for xa, xb, ya, yb in [(x0_ref, x1_ref, y0, y1), (x1_ref, x0_ref, y1, y0)]:
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(xb)
        true_idx = nn.kneighbors(xa, return_distance=False).flatten()

        dist_to_match = np.linalg.norm(ya - yb[true_idx], axis=1)
        dists = np.linalg.norm(ya[:, None, :] - yb[None, :, :], axis=-1)

        denom = max(yb.shape[0] - 1, 1)
        frac = (dists < dist_to_match[:, None]).sum(axis=1) / denom
        scores.append(frac.mean())

    return float(np.mean(scores))


def compute_rand_index(labels_true, labels_pred):
    """
    Computes the Rand index between two partitions, matching the definition used in the paper:

        Rand = (a + b) / binom(n, 2),

    where a is the number of pairs assigned to the same cluster in both partitions and b is the
    number of pairs assigned to different clusters in both partitions.

    :param labels_true: ground-truth cluster labels
    :param labels_pred: predicted cluster labels
    :return: Rand index in [0, 1]
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError(
            "labels_true and labels_pred must have the same number of samples, "
            f"got {labels_true.shape[0]} and {labels_pred.shape[0]}"
        )

    n_samples = labels_true.shape[0]
    if n_samples < 2:
        return 1.0

    _, true_inverse = np.unique(labels_true, return_inverse=True)
    _, pred_inverse = np.unique(labels_pred, return_inverse=True)

    n_true = int(true_inverse.max()) + 1
    n_pred = int(pred_inverse.max()) + 1
    contingency = np.zeros((n_true, n_pred), dtype=np.int64)
    np.add.at(contingency, (true_inverse, pred_inverse), 1)

    def comb2(counts):
        counts = np.asarray(counts, dtype=np.int64)
        return np.sum(counts * (counts - 1) // 2, dtype=np.int64)

    total_pairs = n_samples * (n_samples - 1) // 2
    same_in_both = comb2(contingency)
    same_in_true = comb2(contingency.sum(axis=1))
    same_in_pred = comb2(contingency.sum(axis=0))
    different_in_both = total_pairs - same_in_true - same_in_pred + same_in_both

    return float((same_in_both + different_in_both) / total_pairs)


def compute_kmeans_rand(embd, gt, n_clusters=6, seed=0):
    """Compute adjusted Rand index after fitting k-means to an embedding."""
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    pred = kmeans.fit_predict(embd)
    return adjusted_rand_score(gt, pred)


def compute_kmeans_cluster_scores(embd, labels, n_clusters=None, seed=0):
    """Return Rand and adjusted Rand scores for k-means clusters."""
    embd, labels = _validate_embedding_and_labels(embd, labels)
    if n_clusters is None:
        n_clusters = int(np.unique(labels).size)
    n_clusters = _validate_positive_integer(n_clusters, "n_clusters")
    if n_clusters > embd.shape[0]:
        raise ValueError(
            f"n_clusters={n_clusters} exceeds the number of samples {embd.shape[0]}."
        )
    predicted = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(embd)
    return {
        "rand_index": compute_rand_index(labels, predicted),
        "adjusted_rand_index": float(adjusted_rand_score(labels, predicted)),
    }
