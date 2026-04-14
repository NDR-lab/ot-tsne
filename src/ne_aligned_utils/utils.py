import ot
import scipy.sparse as sp
import numpy as np
from importlib.resources import files
from ne_aligned_utils.transport_plan_sparsify import sparsify_transport_plan

# utilities

def get_path(path_type):
    # util for reading in paths from file
    with files(__package__).joinpath("paths").open("rb") as file:
        lines = file.readlines()

    lines = [line.decode('ascii').split(" ") for line in lines]
    path_dict = {line[0]: " ".join(line[1:]).strip("\n") for line in lines}

    if path_type == "data":
        try:
            return path_dict["data_path"]
        except KeyError:
            print("There is no path 'data_path'.")

    elif path_type == "figures":
        try:
            return path_dict["fig_path"]
        except KeyError:
            print("There is no path 'fig_path'.")



def compute_ot_plan(x0, x1, eps_times=1, tol=1e-12, maxIter=int(1e4)):
    dist = ot.dist(x0, x1)  # this is already the squared Euclidean distance
    eps = np.median(dist)  # square of median and median of squares are the same

    eps /= eps_times

    n0 = len(x0)
    n1 = len(x1)

    a = np.ones((n0,)) / n0
    b = np.ones((n1,)) / n1

    P = ot.sinkhorn(a, b, dist, reg=eps, numItermax=maxIter, stopThr=tol)
    return P

def _plan_stats(P_sparse, method, percentage=None):
    nnz = int(np.count_nonzero(P_sparse))
    return {
        "nnz": nnz,
        "density": float(nnz / P_sparse.size),
        "retained_mass_before_renorm": float(P_sparse.sum()),
        "method": method,
        "percentage": None if percentage is None else float(percentage),
    }


def sparsify_plan(
    P,
    mode=None,
    k=15,
    percentage=None,
    ensure_nonempty=True,
    renormalize=True,
    return_stats=False,
    below_top_scale_exponent=2.0,
):
    if mode is None:
        P_sparse = P

    elif mode == "topk":
        P_sparse = np.zeros_like(P)
        for i in range(P.shape[0]):
            idx = np.argpartition(P[i], -k)[-k:]
            P_sparse[i, idx] = 1
        for j in range(P.shape[1]):
            idx = np.argpartition(P[:, j], -k)[-k:]
            P_sparse[idx, j] = 1

    elif mode == "weighted_topk":
        P_sparse = np.zeros_like(P)
        for i in range(P.shape[0]):
            idx = np.argpartition(P[i], -k)[-k:]
            P_sparse[i, idx] = P[i, idx]
        for j in range(P.shape[1]):
            idx = np.argpartition(P[:, j], -k)[-k:]
            P_sparse[idx, j] = P[idx, j]

    elif mode in {
        "percentage_neighbors",
        "percentage_mass",
        "percentage_below_top",
    }:
        if percentage is None:
            raise ValueError(f"percentage must be provided when mode='{mode}'.")
        return sparsify_transport_plan(
            P,
            method=mode,
            percentage=percentage,
            ensure_nonempty=ensure_nonempty,
            renormalize=renormalize,
            return_stats=return_stats,
            below_top_scale_exponent=below_top_scale_exponent,
        )
    else:

        raise NotImplementedError

    if return_stats:
        return P_sparse, _plan_stats(P_sparse, method=mode, percentage=percentage)
    return P_sparse

def get_eot(P, t=0, returnS=False):
    U, S, V = np.linalg.svd(P, full_matrices=False)
    n0 = P.shape[0]
    n1 = P.shape[1]
    eot = np.concatenate([U * S**t * np.sqrt(n0),
                          V.T *S**t * np.sqrt(n1)], axis=0)
    if returnS:
        return eot, S
    else:
        return eot

def get_bipartite_graph(P, diag=False):
    n0, n1 = P.shape

    if diag:
        eot, S = get_eot(P, t=0, returnS=True)
        sim = eot @ np.diag(S) @ eot.T / np.sqrt(n0 * n1)
        sim[sim<0] = 0
        np.fill_diagonal(sim, 0)
        G_sp = sp.csr_matrix(sim)
    else:
        P_sp = sp.csr_matrix(P)
        Z1 = sp.csr_matrix((n0, n0))
        Z2 = sp.csr_matrix((n1, n1))
        G_sp = sp.bmat([[Z1, P_sp],
                        [P_sp.T, Z2]],
                       format='csr')

    return G_sp / G_sp.sum()


def sparsify_affinities(a, n0, n1, k=15, symmetric=True):

    idx_upper_diag = (np.arange(n0, dtype=int)[:, None], a[:n0, :n0].argsort(axis=1)[:, -k:])
    idx_upper_right = (np.arange(n0,dtype=int)[:, None], a[:n0, n0:].argsort(axis=1)[:, -int(k*n1/n0):])

    idx_lower_left = (np.arange(n1,dtype=int)[:, None], a[n0:, :n0].argsort(axis=1)[:, -k:])
    idx_lower_diag = (np.arange(n1,dtype=int)[:, None], a[n0:, n0:].argsort(axis=1)[:, -int(k*n1/n0):])

    a_sparse = np.zeros_like(a)

    a_sparse[:n0, :n0][idx_upper_diag] = 1
    a_sparse[:n0, n0:][idx_upper_right] = 1
    a_sparse[n0:, :n0][idx_lower_left] = 1
    a_sparse[n0:, n0:][idx_lower_diag] = 1

    if symmetric:
        a_sparse = np.maximum(a_sparse, a_sparse.T)

    np.fill_diagonal(a_sparse, 0)

    a_sparse /= a_sparse.sum()
    return a_sparse
