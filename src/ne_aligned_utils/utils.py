import ot
import scipy.sparse as sp
import numpy as np


def compute_ot_plan(x0, x1, eps_times=1, tol=1e-12, maxIter=int(1e4)):
    dist = ot.dist(x0, x1)  # this is already the squared Euclidean distance
    eps = np.median(dist)  # square of median and median of squares are the same

    eps *= eps_times

    n0 = len(x0)
    n1 = len(x1)

    a = np.ones((n0,)) / n0
    b = np.ones((n1,)) / n1

    P = ot.sinkhorn(a, b, dist, reg=eps, numItermax=maxIter, stopThr=tol)
    return P

def sparsify_plan(P, mode=None, k=15):
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
            P_sparse[i, idx] += P[i, idx]
        for j in range(P.shape[1]):
            idx = np.argpartition(P[:, j], -k)[-k:]
            P_sparse[idx, j] += P[idx, j]
    else:

        raise NotImplementedError

    return P_sparse

def get_bipartite_graph(P):
    n0, n1 = P.shape
    P_sp = sp.csr_matrix(P)
    Z1 = sp.csr_matrix((n0, n0))
    Z2 = sp.csr_matrix((n1, n1))

    G_sp = sp.bmat([[Z1, P_sp],
                     [P_sp.T, Z2]],
                    format='csr')
    return G_sp
