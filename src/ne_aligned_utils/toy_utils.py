# implementing basic functionalities first by myself, to better understand the project

import numpy as np
import scipy.sparse as sp
from openTSNE import TSNE
from openTSNE.affinity import Affinities
from sklearn.neighbors import NearestNeighbors

############# metrics #############

def compute_foscttm(x0, x1, y0, y1):
    """
    Computes FOSCTTM (Fraction of Samples Closer Than the True Match).
    For each point in x0, the true match in x1 is its nearest neighbor in
    clean/latent space. FOSCTTM is the fraction of x1 points that are closer
    to the embedded point in y0 than its true match in y1. Computed
    symmetrically and averaged.

    :param x0: clean positions of dataset 0 (n0 x d)
    :param x1: clean positions of dataset 1 (n1 x d)
    :param y0: embedded positions of dataset 0 (n0 x d')
    :param y1: embedded positions of dataset 1 (n1 x d')
    :return: FOSCTTM score (lower is better: 0 = perfect, 0.5 = random)
    """
    scores = []
    for xa, xb, ya, yb in [(x0, x1, y0, y1), (x1, x0, y1, y0)]:
        # find true match of each xa point in xb (nearest neighbor in clean space)
        true_match_idx = NearestNeighbors(n_neighbors=1).fit(xb).kneighbors(xa)[1].flatten()

        # distance to true match in embedding space
        dist_to_match = np.linalg.norm(ya - yb[true_match_idx], axis=1)  # (na,)

        # pairwise distances from ya to all yb points
        dists = np.linalg.norm(ya[:, None, :] - yb[None, :, :], axis=-1)  # (na, nb)

        # fraction of yb points closer than the true match
        scores.append((dists < dist_to_match[:, None]).mean(axis=1).mean())

    return np.mean(scores)


############# optimal transport utilities #############

def sinkhorn(M, r, c, reg, n_iter=100):
    """
    M: cost matrix (n x m)
    r: source distribution (n,)
    c: target distribution (m,)
    reg: regularization
    n_iter: number of iterations
    """
    K = np.exp(-M / reg)
    u = np.ones(len(r))

    for _ in range(n_iter):
        u = r / (K @ (c / (K.T @ u)))

    v = c / (K.T @ u)
    T = u[:, None] * K * v[None, :]
    return T


############# data generation #############

def make_circle_data(n0=1000, n1=5000, seed=0):
    np.random.seed(seed)

    theta0 = np.sort(2 * np.pi * np.random.rand(n0))
    theta1 = np.sort(2 * np.pi * np.random.rand(n1))

    r0 = 1 + 0.3 * np.random.rand(n0)
    r1 = 1 + 0.3 * np.random.rand(n1)

    # dataset 0
    x0 = 0.5 * r0 * np.cos(theta0) - 0.5
    y0 = 0.5 * r0 * np.sin(theta0) - 1
    z0 = np.zeros(n0)

    # dataset 1
    x1 = r1 * np.cos(theta1) + 3
    y1 = r1 * np.sin(theta1)
    z1 = np.random.rand(n1) - 0.5

    return np.stack([x0, y0, z0], axis=-1), np.stack([x1, y1, z1], axis=-1)


############# ot plan #############

def compute_ot_plan(x0, x1, eps_times=1, n_iter=10000):
    diff = x0[:, None, :] - x1[None, :, :]
    M = (diff ** 2).sum(-1)

    eps = np.median(M) / eps_times

    n0, n1 = len(x0), len(x1)
    a = np.ones(n0) / n0
    b = np.ones(n1) / n1

    return sinkhorn(M, a, b, reg=eps, n_iter=n_iter)


############# diagonal graph from ot plan #############

def get_diag_graph(P):
    n0, n1 = P.shape
    U, S, V = np.linalg.svd(P, full_matrices=False)

    eot = np.concatenate([U * np.sqrt(n0), V.T * np.sqrt(n1)], axis=0)

    A = eot @ np.diag(S) @ eot.T / np.sqrt(n0 * n1)

    A[A < 0] = 0
    np.fill_diagonal(A, 0)

    return A


############# sparsify + symmetrize #############

def sparsify_sym(A, n0, n1, k=15):
    n = n0 + n1
    A_sparse = np.zeros_like(A)

    rows0 = np.arange(n0)
    rows1 = np.arange(n1)

    idx = A[:n0, :n0].argsort(axis=1)[:, -k:]
    A_sparse[:n0, :n0][rows0[:, None], idx] = 1

    idx = A[:n0, n0:].argsort(axis=1)[:, -k:]
    A_sparse[:n0, n0:][rows0[:, None], idx] = 1

    idx = A[n0:, :n0].argsort(axis=1)[:, -k:]
    A_sparse[n0:, :n0][rows1[:, None], idx] = 1

    idx = A[n0:, n0:].argsort(axis=1)[:, -k:]
    A_sparse[n0:, n0:][rows1[:, None], idx] = 1

    A_sparse = np.maximum(A_sparse, A_sparse.T)
    np.fill_diagonal(A_sparse, 0)

    return sp.csr_matrix(A_sparse / A_sparse.sum())


############# inter-data laplacian eigenmaps (t=0) #############

def inter_data_le(P, q=2):
    """
    P: OT plan (n0 x n1), rows sum to 1/n0, cols to 1/n1
    q: embedding dimension
    X embedded as sqrt(n0) * U[:, 1:q+1]
    Y embedded as sqrt(n1) * V[1:q+1, :].T
    """
    n0, n1 = P.shape
    U, S, V = np.linalg.svd(P, full_matrices=False)

    embd0 = np.sqrt(n0) * U[:, 1:q+1]   
    embd1 = np.sqrt(n1) * V[1:q+1, :].T  

    return embd0, embd1


############# aligned tsne #############

def aligned_tsne(graph, n_epochs=200, seed=0):
    n = graph.shape[0]
    init = np.random.randn(n, 2) * 1e-4

    n_early = n_epochs // 3
    n_late = n_epochs - n_early

    embedder = TSNE(n_components=2,
                    verbose=True,
                    random_state=seed,
                    early_exaggeration_iter=n_early,
                    n_iter=n_late)

    affinities = Affinities()
    affinities.P = graph if sp.issparse(graph) else sp.csr_matrix(graph)

    return embedder.fit(affinities=affinities, initialization=init)
