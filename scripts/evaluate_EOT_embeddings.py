from ne_aligned_utils.metrics import run_all_metrics
import os

# change hyperparameters here
seeds = [0, 1, 2, 3 ,4]
dataset = "fig1"
theta = 1.0

tol = 1e-12
maxIter = int(1e4)

eps_times_list = [1, 10, 100]
embd_dim = 2
sparsify = None # "topk" # None # "topk" # None

# EOT hyperparams
ts = [0, 1]


res_path = os.path.join("/gpfs01/berens/user/sdamrich/data/ne_aligned", dataset)
# do not change anything below this line


if dataset == "gmm":
    dataset_str = dataset+f"_theta_{theta}"
else:
    dataset_str = dataset


for eps_times in eps_times_list:
    for t in ts:
        file_name_parts = [f"{dataset_str}_", f"epstimes_{eps_times}_sparsify_{sparsify}_dim_{embd_dim}_EOT_t_{t}.npz"]

        run_all_metrics(file_name_parts=file_name_parts,
                        res_path=res_path,
                        dataset=dataset,
                        seeds=seeds,
                        theta=theta)








