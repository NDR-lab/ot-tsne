from ne_aligned_utils.data import make_fig1_data, make_joint_gmm, load_nCoV
from ne_aligned_utils.utils import compute_ot_plan, sparsify_plan, get_bipartite_graph, sparsify_affinities
import os
from openTSNE import TSNE
from openTSNE.affinity import Affinities
import numpy as np
import ot
import scipy.sparse as sp
import time


########################
# hyperparameter start
########################

seeds = [0]#, 1, 2, 3 ,4]
dataset = "nCoV" # "fig1" or "gmm" or "nCoV"
theta = 3

tol = 1e-12
maxIter = int(1e4)

eps_times_list = [1]#, 10, 100]
sparsify = None # None # "topk" # None # "weighted_topk"
sparsify_aff = True
norm = "whole"  # "per_point" # "whole"
embd_dim = 2
diag=False


init_str = "random"  # "pca" or "random"
init_scale = 10000
n_epochs = 200

res_path = os.path.join("/gpfs01/berens/user/sdamrich/data/ne_aligned", dataset)

########################
# hyperparameter end
########################

if dataset == "gmm":
    dataset_str = dataset+f"_theta_{theta}"
else:
    dataset_str = dataset

for seed in seeds:
    for eps_times in eps_times_list:
        filename = f"{dataset_str}_seed_{seed}_epstimes_{eps_times}_diag_{diag}_sparsify_{sparsify}_norm_{norm}_sparsify_aff_{sparsify_aff}_dim_{embd_dim}_init_{init_str}" + \
                                f"_scale_{init_scale}_epochs_{n_epochs}_tsne.npz"


        filepath = os.path.join(res_path, filename)

        try:
            data = np.load(filepath)
            print(f"File {filepath} already exists, skipping computation.")
            continue
        except FileNotFoundError:
            print(f"Starting with {filename}")

            # make data
            if dataset == "fig1":
                x0, x1 = make_fig1_data(seed=seed)
            elif dataset == "gmm":
                x0, x1 = make_joint_gmm(seed=seed, theta=theta)
            elif dataset == "nCoV":
                x0, x1 = load_nCoV(res_path=res_path, return_meta=False)
            else:
                raise NotImplementedError

            np.random.seed(seed)

            # get OT plan and turn into graph
            before_ot = time.time()
            P = compute_ot_plan(x0, x1, eps_times=eps_times, tol=tol, maxIter=maxIter)
            after_ot = time.time()

            P_sparse = sparsify_plan(P, mode=sparsify, k=15)

            graph = get_bipartite_graph(P_sparse, diag=diag)

            if sparsify_aff:
                graph = sparsify_affinities(graph.toarray(), n0=len(x0), n1=len(x1), k=15)
                graph = sp.csr_matrix(graph)

            if norm == "whole":
                graph.sum_duplicates()
                graph /= graph.sum()
            elif norm == "per_point":
                graph.sum_duplicates()
                graph /= graph.sum(1)
                graph = (graph.T + graph) / (2*graph.shape[0])  # make symmetric
                assert np.allclose(graph.sum(), 1)
            else:
                raise NotImplementedError("Only 'whole' normalization is implemented.")


            # compute initialization
            if init_str == "random":
                init = np.random.randn(len(x0) + len(x1), embd_dim)
            else:
                raise NotImplementedError

            init /= init[:, 0].std() * init_scale

            # compute embedding
            n_early_epochs = n_epochs // 3
            n_late_epochs = n_epochs - n_early_epochs
            embedder = TSNE(n_components=embd_dim,
                            verbose=True,
                            random_state=seed,
                            early_exaggeration_iter=n_early_epochs,
                            n_iter=n_late_epochs)
            affinities = Affinities()
            affinities.P = graph.tocsr()

            before_fit = time.time()
            embd = embedder.fit(affinities=affinities, initialization=init)
            after_fit = time.time()


            # save results
            np.savez(filepath,
                        P_sparse=P_sparse,
                        sparsify_aff = sparsify_aff,
                        embd=embd,
                        time_ot=after_ot - before_ot,
                        time_fit=after_fit - before_fit,
                        eps_times=eps_times,
                        embd_dim=embd_dim,
                        init_str=init_str,
                        init_scale=init_scale,
                        norm=norm,
                        n_epochs=n_epochs,
                        diag=diag
                     )
            print(f"Saved results to {filepath}")








