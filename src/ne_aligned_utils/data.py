import numpy as np
import os
import pandas as pd

# generation / loading of data

def make_fig1_data(n0=1000, n1=5000, seed=0, clean=False):
    np.random.seed(seed)

    # Deformed circle data
    theta1 = np.sort(2 * np.pi * np.random.rand(n0))
    theta2 = np.sort(2 * np.pi * np.random.rand(n1))

    x1_clean = np.cos(theta1)
    y1_clean = np.sin(theta1)

    x2_clean = np.cos(theta2)
    y2_clean = np.sin(theta2)

    r1 = 1 + 0.3 * np.random.rand(n0)
    r2 = 1 + 0.3 * np.random.rand(n1)

    x1_clean *= r1
    y1_clean *= r1

    x2_clean *= r2
    y2_clean *= r2

    if clean:
        return np.stack([x1_clean, y1_clean], axis=-1), np.stack([x2_clean, y2_clean], axis=-1)

    else:
        x1_clean *= 0.5
        y1_clean *= 0.5

        x1 = x1_clean - 0.5
        y1 = y1_clean - 1
        z1 = np.zeros(n0)

        x2 = x2_clean + 3
        y2 = y2_clean
        z2 = np.random.rand(n1) - 0.5

        return np.stack([x1, y1, z1], axis=-1), np.stack([x2, y2, z2], axis=-1)


def make_gmm(n=600, mean_scale=5.0, std=1.0, q=6):
    # sample from GMM

    # Means: mu_j = mean_scale * e_j
    means = mean_scale * np.eye(q)  # row j is mu_j


    # Mixture weights: uniform
    weights = np.ones(q) / q

    # Sample component indices
    component_indices = np.random.choice(q, size=n, p=weights)

    # Sample points
    samples = means[component_indices] + std * np.random.randn(n, q)

    return samples, component_indices

def make_heteroscedastic_noise(n=600, r=6, std=1):
    # sample heteroscedastic noise
    eta = np.random.randn(n, r) * std

    ind_k = np.tile(np.arange(r), reps=(n, 1))
    ind_n = np.tile(np.arange(n).reshape(-1, 1), reps=(1, r))

    mask1 = (1 <= ind_k) & (ind_n <= n//3)
    mask2 = (n//3 < ind_n ) & (ind_n <= 2*n//3)

    assert not np.any(mask1 & mask2)
    eta[mask1] * 10
    eta[mask2] * 5

    return eta

def make_joint_gmm(n0=600, n1=600, p=1000, q=6, seed=0, theta=1.0, clean=False):
    np.random.seed(seed)

    # sample from GMM

    clean0, clusters0 = make_gmm(n=n0, mean_scale=5.0, std=1.0, q=q)
    clean1, clusters1 = make_gmm(n=n1, mean_scale=5.0, std=1.0, q=q)

    if clean:
        return clean0, clean1, clusters0, clusters1

    # add noise
    eta1 = np.random.randn(n0, p)
    eta2 = make_heteroscedastic_noise(n=n1, r=q, std=1.0)
    eta2 = np.concatenate([eta2, np.zeros(shape=(n1, p-q))], axis=-1)

    v1 = np.concatenate([15*np.ones(2), np.zeros(p-2 )])

    z2 = np.random.rand(n1, p-q) * 1.5*theta + 0.5*theta


    data0 = v1 + theta * np.concatenate([clean0, np.zeros((n0, p-q))], axis=-1) + 0 + eta1
    data1 = 0  + theta * np.concatenate([clean1, np.zeros((n1, p-q))], axis=-1)  + eta2

    if p > q:
        data1 += z2 @ np.eye(p)[:, q:].T

    return data0, data1

def load_atac(res_path, return_meta=False):
    if res_path.endswith("atac"):
        data_path = res_path
    else:
        data_path = os.path.join(res_path, "atac")

    data0 = np.load(os.path.join(data_path, "data1.npy"))
    data1 = np.load(os.path.join(data_path, "data2.npy"))
    if not return_meta:
        return data0, data1

    cell_type_names = np.load(os.path.join(data_path, "cell_type_names.npy"), allow_pickle=True)
    labels0 = np.load(os.path.join(data_path, "labels1.npy"))
    labels1 = np.load(os.path.join(data_path, "labels2.npy"))
    dataset_names = np.load(os.path.join(data_path, "dataset_names.npy"), allow_pickle=True)

    return data0, data1, cell_type_names, labels0, labels1, dataset_names

def load_nCoV(res_path, return_meta=False):
    if res_path.endswith("nCoV"):
        data_path = res_path
    else:
        data_path = os.path.join(res_path, "nCoV")

    data0 = np.load(os.path.join(data_path, "X1.npy"))
    data1 = np.load(os.path.join(data_path, "X2.npy"))
    if not return_meta:
        return data0, data1

    cell_type_names = np.load(os.path.join(data_path, "cell_type_names.npy"), allow_pickle=True)
    labels0 = np.load(os.path.join(data_path, "labels1.npy"))
    labels1 = np.load(os.path.join(data_path, "labels2.npy"))
    dataset_names = np.load(os.path.join(data_path, "dataset_names.npy"), allow_pickle=True)

    return data0, data1, cell_type_names, labels0, labels1, dataset_names


def load_lupus(res_path, return_meta=False):
    if res_path.endswith("lupus"):
        data_path = res_path
    else:
        data_path = os.path.join(res_path, "lupus")

    data0 = np.load(os.path.join(data_path, "X1.npy"))
    data1 = np.load(os.path.join(data_path, "X2.npy"))
    if not return_meta:
        return data0, data1

    cell_type_names = np.load(os.path.join(data_path, "cell_type_names.npy"), allow_pickle=True)
    labels0 = np.load(os.path.join(data_path, "labels1.npy"))
    labels1 = np.load(os.path.join(data_path, "labels2.npy"))
    dataset_names = np.load(os.path.join(data_path, "dataset_names.npy"), allow_pickle=True)

    return data0, data1, cell_type_names, labels0, labels1, dataset_names


def load_yao_10x_purple(res_path, seed=0, subsample_size=5000):
    res_path_purple = os.path.join(res_path, "yao_10x_purple")

    meta = pd.read_csv(os.path.join(res_path, "yao_10x", "10x_metadata.csv"))


    np.random.seed(seed)
    meta = pd.read_csv(os.path.join(res_path, "10x_metadata.csv"))



    # subsample
    #perm0 = np.random.permutation(data0.shape[0])[:subsample_size]
    #perm1 = np.random.permutation(data1.shape[0])[:subsample_size]

    #return data0[perm0], data1[perm1]


