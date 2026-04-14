import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import os
from ne_aligned_utils.data import make_fig1_data, make_joint_gmm
import re


# performance metrics


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

    if isinstance(k, (bool, np.bool_)) or not np.isscalar(k):
        raise ValueError(f"k must be a positive integer, got {k!r}.")
    try:
        k_int = int(k)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"k must be a positive integer, got {k!r}.") from exc
    if k_int != k or k_int < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}.")
    k = k_int

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


def compute_knn_same_cell_type_fraction(embd, labels, k=50):
    """
    Compute the mean fraction of each point's k nearest neighbors that share its label.

    :param embd: embedding coordinates with shape (n_samples, n_dims)
    :param labels: 1D cell-type label vector with length n_samples
    :param k: number of nearest neighbors considered per point
    :return: Mean same-cell-type kNN fraction in [0, 1]
    """
    labels, neighbor_indices, effective_k = _compute_self_knn_indices(
        embd,
        labels,
        k,
        metadata_name="labels",
    )
    if effective_k == 0:
        return 0.0

    same_label = labels[neighbor_indices] == labels[:, None]
    return float(same_label.mean(axis=1).mean())


def compute_knn_other_batch_fraction(embd, batch_ids, k=50):
    """
    Compute the mean fraction of each point's k nearest neighbors that come from another batch.

    This metric is generic on the cells passed in. For datasets where some cell types appear in
    only one batch, filter to a shared-label subset before calling this function if you want a
    batch-mixing score that is not penalized by missing cell types.

    :param embd: embedding coordinates with shape (n_samples, n_dims)
    :param batch_ids: 1D batch-id vector with length n_samples
    :param k: number of nearest neighbors considered per point
    :return: Mean other-batch kNN fraction in [0, 1]
    """
    batch_ids, neighbor_indices, effective_k = _compute_self_knn_indices(
        embd,
        batch_ids,
        k,
        metadata_name="batch_ids",
    )
    if effective_k == 0:
        return 0.0

    other_batch = batch_ids[neighbor_indices] != batch_ids[:, None]
    return float(other_batch.mean(axis=1).mean())



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
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    pred = kmeans.fit_predict(embd)
    return adjusted_rand_score(gt, pred)


def run_all_metrics(file_name_parts, res_path, dataset, seeds, theta=1.0):
    concordances = []
    cross_concordances = []
    foscttms = []
    rands = []
    for seed in seeds:
        filename = f"seed_{seed}_".join(file_name_parts)

        filepath = os.path.join(res_path, filename)
        a = np.load(filepath)
        embd = a["embd"]

        # concordance
        if dataset == "fig1":
            clean0, clean1 = make_fig1_data(seed=seed, clean=True)
        elif dataset == "gmm":
            clean0, clean1, cluster0, cluster1 = make_joint_gmm(seed=seed, theta=theta, clean=True)

        clean = np.concatenate([clean0, clean1], axis=0)
        embd0 = embd[:len(clean0)]
        embd1 = embd[len(clean0):]

        concordances.append(compute_concordance(embd, clean))
        cross_concordances.append(compute_cross_concordance(clean0, clean1, embd0, embd1))

        foscttms.append(compute_foscttm(clean0, clean1, embd0, embd1))

        if dataset == "gmm":
            # also compute rand
            gt_clusters = np.concatenate([cluster0, cluster1])
            rands.append(compute_kmeans_rand(embd, gt_clusters, n_clusters=len(np.unique(gt_clusters)), seed=seed))

    metric_filename =  re.sub(r"_seed_\d+", "", filename)[:-4] + "_metrics.npz"
    metric_filepath = os.path.join(res_path, metric_filename)

    if dataset == "gmm":
        np.savez(metric_filepath,
                 concordance=np.array(concordances),
                 cross_concordance=np.array(cross_concordances),
                 foscttm=np.array(foscttms),
                 rand=np.array(rands))
    else:
        np.savez(metric_filepath,
                 concordance=np.array(concordances),
                 cross_concordance=np.array(cross_concordances),
                 foscttm=np.array(foscttms))

    print(f"Saved metrics to {metric_filepath} for all seeds.")
