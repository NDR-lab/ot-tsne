import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import os
from ne_aligned_utils.data import make_fig1_data, make_joint_gmm
import re


# performance metrics



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


def compute_kmeans_rand(embd, gt, n_clusters=6, seed=0):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    pred = kmeans.fit_predict(embd)
    ari = adjusted_rand_score(gt, pred)
    return ari


def run_all_metrics(file_name_parts, res_path, dataset, seeds, theta=1.0):
    concordances = []
    cross_concordances = []
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

        concordances.append(compute_concordance(embd, clean))
        cross_concordances.append(compute_cross_concordance(clean0, clean1, embd[:len(clean0)], embd[len(clean0):]))

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
                 rand=np.array(rands))
    else:
        np.savez(metric_filepath,
                 concordance=np.array(concordances),
                 cross_concordance=np.array(cross_concordances))

    print(f"Saved metrics to {metric_filepath} for all seeds.")
