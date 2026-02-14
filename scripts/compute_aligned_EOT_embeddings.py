from ne_aligned_utils.data import make_fig1_data, make_joint_gmm
from ne_aligned_utils.utils import compute_ot_plan, sparsify_plan, get_bipartite_graph
import os
import numpy as np
import ot
import time

########################
# hyperparameter start
########################

seeds = [0, 1, 2, 3 ,4]
dataset = "gmm"
theta = 3.0

tol = 1e-12
maxIter = int(1e4)

eps_times_list = [1, 10, 100]
embd_dim = 6
sparsify = None # "topk" # None # "topk" # None
ts = [0, 1]

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
        for t in ts:
            filename = f"{dataset_str}_seed_{seed}_epstimes_{eps_times}_sparsify_{sparsify}_dim_{embd_dim}_EOT_t_{t}.npz"

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
                else:
                    raise NotImplementedError

                np.random.seed(seed)

                # get OT plan and turn into graph
                before_ot = time.time()
                P = compute_ot_plan(x0, x1, eps_times=eps_times, tol=tol, maxIter=maxIter)
                after_ot = time.time()

                P_sparse = sparsify_plan(P, mode=sparsify, k=15)


                before_fit = time.time()
                U, S, V = np.linalg.svd(P_sparse, full_matrices=False)
                n0 = x0.shape[0]
                n1 = x1.shape[0]
                embd = np.concatenate([U[:, 1:1+embd_dim] * S[1:1+embd_dim]**t * np.sqrt(n0),
                                       V.T[:, 1:1+embd_dim] *S[1:1+embd_dim]**t * np.sqrt(n1)], axis=0)
                after_fit = time.time()


                # save results
                np.savez(filepath,
                            P_sparse=P_sparse,
                            embd=embd,
                            time_ot=after_ot - before_ot,
                            time_fit=after_fit - before_fit,
                            eps_times=eps_times,
                            embd_dim=embd_dim,
                            t=t
                         )
                print(f"Saved results to {filepath}")








