from ne_aligned_utils.data import make_fig1_data, make_joint_gmm
from ne_aligned_utils.metrics import run_all_metrics
import os
import ot

# change hyperparameters here

seeds = [0, 1, 2, 3 ,4]
dataset = "fig1"
theta = 3.0

tol = 1e-12
maxIter = int(1e4)

eps_times_list = [1, 10, 100]
embd_dim = 2
sparsify =  "topk"  # # None "topk"

# EOT hyperparams
init_str = "random"  # "pca" or "random"
init_scale = 1

batch_size = 2**14
negative_samples = 5
n_epochs = 200
loss_mode = "infonce"
alignment = False
sample_pos = None #"per_point" #"per_point" #None #

res_path = os.path.join("/gpfs01/berens/user/sdamrich/data/ne_aligned", dataset)

# do not change anything below this line


if dataset == "gmm":
    dataset_str = dataset+f"_theta_{theta}"
else:
    dataset_str = dataset

for eps_times in eps_times_list:
    file_name_parts = [f"{dataset_str}_",
                       f"epstimes_{eps_times}_sparsify_{sparsify}_dim_{embd_dim}_init_{init_str}" + \
                       f"_scale_{init_scale}_bs_{batch_size}_m_{negative_samples}_epochs_{n_epochs}" + \
                       f"_loss_{loss_mode}_align_{alignment}_sample_pos_{sample_pos}.npz"]

    run_all_metrics(file_name_parts=file_name_parts,
                    res_path=res_path,
                    dataset=dataset,
                    seeds=seeds,
                    theta=theta)








