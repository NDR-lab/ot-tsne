# OT-tSNE

This repository contains the code for the paper **Optimal Transport Meets t-SNE: Integrated
Visualization without Batch Effects in OT-tSNE**.

OT-tSNE (Optimal Transport t-SNE) creates a shared two-dimensional embedding of two datasets
affected by batch effects, the systematic differences that can arise between separately
recorded datasets. It combines optimal transport with t-SNE to align the datasets while
preserving meaningful structure, without requiring paired observations or labels.

<p align="center">
  <img src="figures/overview.png" width="1000" alt="Joint t-SNE and OT-tSNE embeddings, each colored by batch and by cell type.">
</p>

## Installation

Install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/NDR-lab/ot-tsne.git
cd ot-tsne
uv sync --locked
```


## Usage example

The basic usage is to pass two batches to `compute_ot_tsne_embedding_from_arrays`.
Here is an example on a toy dataset consisting of three 3D Gaussian blobs, where one batch is translated along the third coordinate.

```python
from sklearn.datasets import make_blobs

from ne_aligned_utils.methods import compute_ot_tsne_embedding_from_arrays

X, labels = make_blobs(
    n_samples=300,
    centers=[[-4, 0, 0], [0, 4, 0], [4, 0, 0]],
    cluster_std=0.5,
    random_state=0,
)
# create batches
x0, x1 = X[:150].copy(), X[150:].copy()
x1[:, 2] += 6.0

# OT-tSNE
embedding = compute_ot_tsne_embedding_from_arrays(
    x0, x1,
    rel_reg=1.0,
    effective_k=15,
    init="eot",
    random_state=0,
    n_jobs=4,
)
```

To run the [example](examples/toy.py):

```bash
uv run python examples/toy.py
```

<p align="center">
  <img src="figures/toy_embedding.png" width="800" alt="Joint t-SNE before integration on the left and OT-tSNE on the right, colored by batch in the top row and population in the bottom row.">
</p>

Inputs are arrays with shapes `(n0, d)` and `(n1, d)`. The result has shape
`(n0 + n1, 2)`: `embedding[:n0]` belongs to the first batch and `embedding[n0:]` to the second.

### Main parameters

| Parameter | Default | Meaning |
|---|---|---|
| `rel_reg` | `1.0` | Entropic regularization relative to the median squared cross-batch distance: `epsilon = rel_reg * median_squared_distance`. Larger values smoothen the transport plan. |
| `effective_k` | `15` | Neighbor-selection budget per observation, divided between batches in proportion to their sizes. Symmetrization can increase the final degree. |
| `random_state` | `0` | Seed for the embedding. |
| `n_jobs` | `1` | Number of threads for the t-SNE stage. |
| `device` | `"cpu"` | OT solver device. `"cuda"` uses a CUDA-capable PyTorch installation for transport; the t-SNE layout still runs on the CPU. |
| `init` | `"random"` | Starting layout: `"random"`, `"pca"`, `"spectral"`, `"eot"` (two leading singular modes of the transport plan) or an array. The benchmark uses `"eot"`. |
| `svd_solver`, `svd_directions` | `"randomized"`, `500` | How the plan's singular vectors are computed (`"randomized"`, `"arpack"` or `"full"`), and how many leading directions are kept. Small plans use the full SVD. |

The default t-SNE schedule uses 250 early-exaggeration iterations followed by 500 main
iterations. The graph is supplied directly, so `effective_k` controls its neighborhood
budget in place of t-SNE's usual perplexity-based affinity construction.

### Single-cell data example

We benchmark OT-tSNE on single-cell integration tasks. Here we use the nCoV dataset from
[Zhu et al. (2020)](https://doi.org/10.1016/j.immuni.2020.07.009), containing 4,117 PBMCs
from patient 5 at days 1 and 13. The checkout includes preprocessed inputs: 50-dimensional
scores from a PCA fitted jointly to both batches.

```text
data/nCoV/preprocessed/
├── X1_pca50.npy                 # 2,877 cells × 50 joint PCs (day 13)
├── X2_pca50.npy                 # 1,240 cells × 50 joint PCs (day 1)
├── labels1.npy, labels2.npy     # cell-type codes per batch
├── cell_type_names.npy          # names of the cell-type codes
├── dataset_names.npy            # names of the two batches
└── preprocessing_manifest.json  # hashes of the source file and every array
```

Load the PCA scores and pass the two batches directly to OT-tSNE:

```python
from pathlib import Path

import numpy as np

from ne_aligned_utils.methods import compute_ot_tsne_embedding_from_arrays

data_dir = Path("data/nCoV/preprocessed")
x0 = np.load(data_dir / "X1_pca50.npy")
x1 = np.load(data_dir / "X2_pca50.npy")
embedding = compute_ot_tsne_embedding_from_arrays(
    x0, x1, rel_reg=1.0, effective_k=15, init="eot", random_state=0, n_jobs=4,
)
```

The [example script](examples/ncov.py) generates the figure below.

<p align="center">
  <img src="figures/ncov_embedding.png" width="1000" alt="OT-tSNE embedding of 4,117 nCoV cells, colored by sample on the left and cell type on the right.">
</p>

### Using your own data

OT-tSNE can be used with any data represented in a shared feature space.
It is not restricted to single-cell data. Both batches may contain different numbers of observations.
The current implementation supports two batches and uses Euclidean distances. Paired observations and labels are not required. 


## Method

<p align="center">
  <img src="figures/method.png" width="1200" alt="OT-tSNE pipeline: cross-batch transport plan, block affinity construction, sparsification, and t-SNE optimization.">
</p>

1. **Match across batches.** Compute a balanced entropic optimal-transport plan from
   squared Euclidean distances between the two batches.
2. **Build a joint affinity matrix.** Use the plan for cross-batch affinities and its
   singular vectors and singular values to construct within-batch affinities. Observations
   are similar within a batch when their transport relationships to the other batch are similar.
3. **Select neighbors.** Retain the strongest affinities under a fixed neighbor budget,
   split proportionally between batches. By default, an edge is kept if either endpoint
   selects it; retained edges receive equal weight, and the graph is normalized.
4. **Embed.** Optimize a two-dimensional t-SNE layout of this graph using
   [openTSNE](https://github.com/pavlin-policar/openTSNE).

Optimal transport is computed with [POT](https://pythonot.github.io/). The worked example
uses the main OT-tSNE method (`ot` in the benchmark). Alternative within-batch constructions are provided in [the method implementations](src/ne_aligned_utils/methods/ot_methods.py).

## Notebooks

The two MNIST walkthroughs explore how the direction and magnitude of a batch effect 
change cross-batch matching and the resulting embeddings:

| Notebook | What it explores |
|---|---|
| [Orthogonal batch effects](notebooks/mnist_orthogonal_batch_effect.ipynb) | Perturbations in PCA-tail directions, comparing the original and signal-projected representations through transport diagnostics, neighbor graphs, and embeddings. |
| [Parallel batch effects](notebooks/mnist_parallel_batch_effect.ipynb) | Opposite shifts along the leading principal component, using constant or digit-dependent shift magnitudes. |

## Repository structure

```text
src/ne_aligned_utils/
  methods/
    ot_methods.py          OT-tSNE variants
    ot_transport.py        Entropic optimal-transport plan
    ot_affinity.py         Affinity construction and neighbor selection
    ot_graph.py            Graph initialization and t-SNE optimization
examples/                  Runnable toy and nCoV examples
notebooks/                 Synthetic experiments and figures
data/nCoV/preprocessed/     Prepared nCoV example inputs
```

## Reproducing the paper

The benchmark and all figures can be rerun from this repository. See the
[reproduction guide](reproduction/README.md) for environments, data, runs and figures.

## Citation


## License

MIT, see [LICENSE](LICENSE).
