import os
import pickle
from importlib.resources import files
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors

from ne_aligned_utils.methods import build_ot_tsne_four_block_matrix
from ne_aligned_utils.transport_plan_sparsify import sparsify_transport_plan


_REPO_ROOT = Path(__file__).resolve().parents[2]

# utilities

def get_path(path_type):
    # util for reading in paths from file
    with files(__package__).joinpath("paths").open("rb") as file:
        lines = file.readlines()

    lines = [line.decode('ascii').split(" ") for line in lines]
    path_dict = {line[0]: " ".join(line[1:]).strip("\n") for line in lines}

    if path_type == "data":
        try:
            return _resolve_configured_path(path_dict["data_path"])
        except KeyError:
            print("There is no path 'data_path'.")

    elif path_type == "figures":
        try:
            return _resolve_configured_path(path_dict["fig_path"])
        except KeyError:
            print("There is no path 'fig_path'.")


def _resolve_configured_path(path_value):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((_REPO_ROOT / path).resolve())


def save_dict(data, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(file_path):
    with Path(file_path).open("rb") as file:
        return pickle.load(file)


def kNN_graph(x, k, metric="euclidean"):
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    neighbors = NearestNeighbors(n_neighbors=min(k + 1, len(x)), metric=metric)
    neighbors.fit(x)
    raw_indices = neighbors.kneighbors(x, return_distance=False)
    row_ids = np.arange(len(x))[:, None]
    indices = raw_indices[raw_indices != row_ids].reshape(len(x), -1)

    if indices.shape[1] < k:
        raise ValueError(
            f"Unable to compute {k} non-self neighbors for {len(x)} samples."
        )

    indices = indices[:, :k]

    return torch.as_tensor(indices, dtype=torch.long)



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

def get_bipartite_graph(P, diag=False):
    n0, n1 = P.shape

    if diag:
        sim = build_ot_tsne_four_block_matrix(P)
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
