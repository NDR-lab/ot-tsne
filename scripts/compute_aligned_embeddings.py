from ne_aligned_utils.data import make_fig1_data
from ne_aligned_utils.utils import compute_ot_plan, sparsify_plan, get_bipartite_graph
import os
import cne
import numpy as np
import ot
import time

seeds = [0, 1, 2, 3 ,4]
dataset = "fig1"

tol = 1e-12
maxIter = int(1e4)

eps_times_list = [10, 100]

embd_dim = 2
init_str = "random"  # "pca" or "random"
init_scale = 1

batch_size = 2**14
negative_samples = 5
n_epochs = 200
loss_mode = "infonce"
alignment = True
sample_pos = "per_point"
sparsify = None

res_path = os.path.join("/gpfs01/berens/user/sdamrich/data/ne_aligned", dataset)

for seed in seeds:
    for eps_times in eps_times_list:
        filename = f"{dataset}_seed_{seed}_epstimes_{eps_times}_sparsify_{sparsify}_dim_{embd_dim}_init_{init_str}" + \
                                f"_scale_{init_scale}_bs_{batch_size}_m_{negative_samples}_epochs_{n_epochs}" + \
                                f"_loss_{loss_mode}_align_{alignment}_sample_pos_{sample_pos}.npz"


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
            else:
                raise NotImplementedError

            np.random.seed(seed)

            # get OT plan and turn into graph
            before_ot = time.time()
            P = compute_ot_plan(x0, x1, eps_times=eps_times, tol=tol, maxIter=maxIter)
            after_ot = time.time()

            P_sparse = sparsify_plan(P, mode=sparsify, k=15)

            graph = get_bipartite_graph(P_sparse)


            # compute initialization
            if init_str == "random":
                init = np.random.randn(len(x0) + len(x1), embd_dim)
            else:
                raise NotImplementedError

            init /= init[:, 0].std() * init_scale

            # compute embedding
            embedder = cne.CNE(negative_samples=negative_samples,
                               batch_size=batch_size,
                               embd_dim=embd_dim,
                               n_epochs=n_epochs,
                               loss_mode=loss_mode,
                               alignment=alignment,
                               data_on_gpu=True,
                               sample_pos=None)
            before_fit = time.time()
            embd = embedder.fit_transform(np.concatenate([x0, x1], axis=0), graph=graph, init=init)
            after_fit = time.time()


            # save results
            np.savez(filepath,
                        P_sparse=P_sparse,
                        embd=embd,
                        time_ot=after_ot - before_ot,
                        time_fit=after_fit - before_fit,
                        eps_times=eps_times,
                        embd_dim=embd_dim,
                        init_str=init_str,
                        init_scale=init_scale,
                        batch_size=batch_size,
                        negative_samples=negative_samples,
                        n_epochs=n_epochs,
                        loss_mode=loss_mode,
                        alignment=alignment,
                        sample_pos=sample_pos
                     )
            print(f"Saved results to {filepath}")








