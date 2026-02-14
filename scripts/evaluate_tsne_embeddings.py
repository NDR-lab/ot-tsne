from ne_aligned_utils.data import make_fig1_data, make_joint_gmm
from ne_aligned_utils.metrics import run_all_metrics
import os
import ot

# change hyperparameters here

seeds = [0, 1, 2, 3 ,4]
dataset = "fig1"
theta = 1.0

tol = 1e-12
maxIter = int(1e4)

eps_times_list = [1, 10, 100]
sparsify = None # None # "topk" # None # "weighted_topk"
norm = "per_point"
embd_dim = 2


init_str = "random"  # "pca" or "random"
init_scale = 10000
n_epochs = 200

res_path = os.path.join("/gpfs01/berens/user/sdamrich/data/ne_aligned", dataset)
# do not change anything below this line


if dataset == "gmm":
    dataset_str = dataset+f"_theta_{theta}"
else:
    dataset_str = dataset

for eps_times in eps_times_list:
    file_name_parts = [f"{dataset_str}_",
                       f"epstimes_{eps_times}_sparsify_{sparsify}_norm_{norm}_dim_{embd_dim}_init_{init_str}" + \
                       f"_scale_{init_scale}_epochs_{n_epochs}_tsne.npz"]

    run_all_metrics(file_name_parts=file_name_parts,
                    res_path=res_path,
                    dataset=dataset,
                    seeds=seeds,
                    theta=theta)







