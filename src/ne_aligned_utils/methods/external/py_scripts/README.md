# Isolated Python-subprocess method scripts

These scripts back benchmark methods whose Python dependencies **cannot coexist with this package's environment**. They are run by a *different* interpreter, so they may not import `ne_aligned_utils` (which pins `numpy>=2.2.6`).

Currently one method: **`multimap`**.

```
Python writes input.npz  ->  <isolated-python> multimap_runner.py input.npz output.npz  ->  Python reads output.npz
```

- `input.npz` — `x0` (n0×G), `x1` (n1×G), `batch_ids` (n0+n1), `params_json` (JSON with `n_components`, `seed`, `scale`, `n_neighbors`).
- `output.npz` — `embedding` ((n0+n1)×`n_components`), rows aligned to `concatenate([x0, x1])`, plus `n_constant_features` / `n_features_used`.

`.npz` is the interchange (not the HDF5 the R methods use) because both ends are Python. The Python side is `../multimap.py`.

## Why the separate interpreter

MultiMAP ([Teichlab/MultiMAP](https://github.com/Teichlab/MultiMAP), Jain et al. 2021) pulls in a UMAP/numba stack that requires `numpy<2`; this repo pins `numpy>=2.2.6`. The package installs into the main env but fails to *import* there. (The PyPI project named `multimap` is unrelated — a generic multi-value dict — so it must be installed from the GitHub repo.)

## Input space and variants

`multimap` receives the shared **processed** features (`input_kind="processed"`, like `bc-tsne` / `scanorama`) — MultiMAP does its own per-batch scaling and PCA and does not want raw counts. It therefore runs on every configured real-data benchmark.

There is **no `multimap-50-tsne` variant**, and `multimap` is not a handicapped "direct-2D" method the way `harmony-2d` / `scvi-2d` are: a 2D layout is MultiMAP's intended output (it is a UMAP-style layout of a multi-graph, not a latent representation to reduce further). This mirrors `bc-tsne`.

The runner calls the official `MultiMAP.Batch()` entry point, so the per-batch PCA (scanpy default, 50 comps), the pairwise joint PCA, the multi-graph and the layout all happen inside the package.

### The one deviation: per-batch scaling

`Batch(scale=True)` calls `sc.pp.scale` per batch, and scanpy computes the variance as `mean(x²) - mean(x)²` in float64. For a feature that is (near-)constant within a batch, catastrophic cancellation can make that **negative**, so `sqrt` gives `NaN`, the whole column becomes `NaN`, and the per-batch PCA dies with `Input X contains NaN`. scanpy guards `std == 0` but not `var < 0`. MNIST's always-dark border pixels trip this on 47 of 784 columns.

The runner therefore does the per-batch N(0,1) scaling itself with a stable variance and passes `scale=False`. Semantics are unchanged — same centring, same unit scaling, same `std == 0 → 1` convention — and unlike dropping the offending columns it discards no features. `n_constant_features` reports how many were constant within a batch (179 of 784 on MNIST; 0 on the single-cell sets).

## Setup

Install it with the repository's single setup command, which builds the isolated Python 3.11 MultiMAP environment from its pinned specification and writes `NE_ALIGNED_MULTIMAP_PYTHON` (see the [reproduction guide](../../../../../reproduction/README.md#benchmark-environments)):

```bash
bash scripts/setup_external_envs.sh
```

Its exact pins are in `envs/multimap/uv.lock`, a separate uv project with `pyproject.toml` and `.python-version` (numpy 1.26.4, scanpy 1.11.5, anndata 0.12.19, MultiMAP commit `681e608`). `scanpy` and `anndata` are needed because `MultiMAP.Batch()` is an AnnData-level API.

## Running it

`multimap` is opt-in (it is excluded from `DEFAULT_METHODS`), so name it:

```bash
uv run python -m ne_aligned_utils.benchmark.sweep run-task \
  --dataset nCoV --task-kind multimap --seed 0 --task-output <out>.npz
```

## Tests

`tests/test_multimap_subprocess.py` exercises the plumbing through a mock interpreter (`tests/mock_multimap_python.py`), so CI needs no MultiMAP install. The opt-in real integration test needs the venv above:

```bash
NE_ALIGNED_TEST_REAL_MULTIMAP=1 uv run python -m pytest -k real_multimap
```
