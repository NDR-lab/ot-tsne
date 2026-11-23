# External methods

Some benchmark methods cannot run in this package's Python environment. They run in a separate
process, and the Python wrapper in this directory exchanges files with it.

| Wrapper | Methods | What has to exist | Exchange |
|---|---|---|---|
| `r.py` + `r_scripts/` | `seurat-*`, `liger-*`, `bc-tsne` (and the unbenchmarked `seurat3-*`, `liger1-*`) | R with Seurat, rliger, bcTSNE | HDF5 in, HDF5 out |
| `multimap.py` + `py_scripts/` | `multimap` | a second Python with numpy < 2 | `.npz` in, `.npz` out |
| `rctsne.py` + `vendor/rctsne/` | `rc-tsne` | FFTW, a C++ compiler and a build step | the binary's own `.dat` files |

All three are installed by one command, which also writes the environment variables the
wrappers read (see the [reproduction guide](../../../../reproduction/README.md#benchmark-environments)):

```bash
bash scripts/setup_external_envs.sh
```

Each wrapper reads its interpreter or binary from one environment variable
(`NE_ALIGNED_RSCRIPT`, `NE_ALIGNED_LIGER_RSCRIPT`, `NE_ALIGNED_MULTIMAP_PYTHON`,
`NE_ALIGNED_RCTSNE_BIN`), split with `shlex.split`. When it is unset, MultiMAP falls back to
`~/.venvs/ne-aligned-multimap` and rc-tSNE to the built `vendor/rctsne/bin/`. R never falls back to
the `Rscript` on `PATH`.
The tests run each wrapper against a mock (`tests/mock_*`), so the suite needs none of these
installs.

## R methods (`r_scripts/`)

```
Python writes input.h5  ->  Rscript <method>.R input.h5 output.h5  ->  Python reads output.h5
```

- `input.h5`: datasets `x0` (n0×G), `x1` (n1×G), `batch_ids`; group `params` with attributes
  `method`, `dim`, `seed`, `n0`, `n1`, `n_features` and method options.
- `output.h5`: dataset `embedding` ((n0+n1)×D), rows in the order of `x0` then `x1`.
- `io.R` holds the shared HDF5 helpers.

| Script | Methods |
|---|---|
| `seurat.R` | `seurat-2d`, `seurat-50` (Seurat v5); `seurat3-2d`, `seurat3-50` (v3-style anchors) |
| `liger.R` | `liger-2d`, `liger-50` (rliger 2.2.1, RcppPlanc 2.0.16) |
| `liger1.R` | `liger1-2d`, `liger1-50` (rliger 1.0.1) |
| `bc_tsne.R` | `bc-tsne` |

The benchmark (`configs/benchmark.toml`) runs `seurat-*`, `liger-*` and `bc-tsne`. Seurat and
LIGER receive raw counts and normalise them in R. BC-tSNE receives the shared processed features
and removes the batch design itself. BC-tSNE is installed from a pinned upstream commit by
`envs/r/install-cran-extras.sh`, which applies `envs/r/bcTSNE-82e8be9.patch`.

Opt-in check against a real R install: `NE_ALIGNED_TEST_REAL_R=1 uv run python -m pytest tests/test_r_subprocess.py`.

## MultiMAP (`py_scripts/`)

```
Python writes input.npz  ->  <multimap python> multimap_runner.py input.npz output.npz  ->  Python reads output.npz
```

MultiMAP ([Jain et al. 2021](https://github.com/Teichlab/MultiMAP)) needs a UMAP/numba stack with
`numpy < 2`, and this package pins `numpy >= 2.2.6`. It therefore runs in its own uv project,
`envs/multimap` (numpy 1.26.4, scanpy 1.11.5, anndata 0.12.19, MultiMAP commit `681e608`). The
runner must not import `ne_aligned_utils`. The PyPI package called `multimap` is unrelated, so
MultiMAP is installed from its GitHub repository.

- `input.npz`: `x0`, `x1`, `batch_ids`, and `params_json` (`n_components`, `seed`, `scale`,
  `n_neighbors`, `use_native_scaling`, `center_batches`).
- `output.npz`: `embedding`, rows in the order of `x0` then `x1`; `n_constant_features`
  (and per batch), `n_features_used`.

The runner calls the official `MultiMAP.Batch()`, so the per-batch scaling and PCA, the joint
PCA, the multi-graph and the layout all happen inside MultiMAP. MultiMAP gets the shared processed
features. Its 2D layout is its intended output, so there is no 50D variant.

One numerical guard: scanpy computes a variance as E[x²] − E[x]², which can come out negative for
a column that is constant within a batch. The scaled column is then NaN and PCA fails (this
happens on MNIST border pixels). The runner keeps the input in float64 and sets columns that are
constant within a batch to zero before calling MultiMAP. No feature is removed.
`n_constant_features` reports how many columns this touched.

`multimap` is not in `DEFAULT_METHODS`, so name it to run it alone:

```bash
uv run python -m ne_aligned_utils.benchmark.sweep run-task \
  --dataset nCoV --task-kind multimap --seed 0 --task-output <out>.npz
```

Opt-in check against the real environment: `NE_ALIGNED_TEST_REAL_MULTIMAP=1 uv run python -m pytest -k real_multimap`.

## rc-tSNE (`vendor/rctsne/`)

`rc-tsne` (revised conditional t-SNE) is run with its authors' compiled program, a FIt-SNE fork,
so the scores reported for it come from their implementation. `vendor/rctsne/VENDOR.md` records
the upstream commit, the per-file licences and every patch.

It is the only method whose optimiser is not openTSNE. It runs the same number of iterations as
the other t-SNE methods (250 early exaggeration, 750 in total), but with its own learning rate and
initialisation.

`scripts/setup/build_rctsne.sh` builds the binary into `vendor/rctsne/bin/` (git-ignored). It
takes FFTW from the pinned `envs/rctsne` environment, uses `$CXX`, `g++` or `clang++`, and links
with `-rpath`. Upstream reads an uninitialised value in the conditioning step; the vendored
source fixes it, and the build fails if an `-Wuninitialized` warning comes back.

Tests: `tests/test_rctsne_subprocess.py` runs against `tests/mock_rctsne_binary.py`, which parses
the real `.dat` input. `NE_ALIGNED_TEST_REAL_RCTSNE=1` adds two tests that need the binary:
determinism per seed, and identical output from the patched and unpatched drivers.

To vendor another program, follow the same shape: sources under `vendor/<name>/` with upstream
licence headers kept, a `VENDOR.md`, a build script under `scripts/setup/`, a wrapper that reads
the binary from one environment variable, a mock under `tests/`, and an `extend-exclude` entry in
`pyproject.toml` so ruff leaves the vendored code alone.
