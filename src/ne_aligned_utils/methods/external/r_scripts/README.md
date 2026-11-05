# R-subprocess method scripts

These scripts back the isolated **R-subprocess benchmark methods** (`seurat-2d`, `seurat-50-tsne`, `seurat3-2d`, `seurat3-50-tsne`, `liger1-2d`, `liger1-50-tsne`, `liger-2d`, `liger-50-tsne`, `bc-tsne`). R is deliberately **not** a Python dependency of this package — the Python side (`../r.py`) shells out to `Rscript` over a small HDF5 file contract:

```
Python writes input.h5  ->  Rscript <method>.R input.h5 output.h5  ->  Python reads output.h5
```

- `input.h5` — datasets `x0` (n0×G), `x1` (n1×G), `batch_ids` (n0+n1); group `params` with attrs `{method, dim, seed, n0, n1, n_features, ...}`.
- `output.h5` — dataset `embedding` ((n0+n1)×D), rows aligned to `c(x0, x1)`.

`io.R` holds the shared HDF5 read/write helpers; each method script sources it.

## Input space

`seurat-*`, `seurat3-*`, `liger1-*`, and `liger-*` receive **raw counts** (`input_kind="raw_counts"`, like scVI) and run their own normalization inside R — that is what Seurat's and LIGER's pipelines expect. Raw counts are available for every configured real-data benchmark. `bc-tsne` receives the shared **processed** features and centres, scales, and projects out the batch design itself.

## Setup

Install it with the repository's single setup command, which builds the `rverify` and `liger2` environments from its pinned specification and writes `NE_ALIGNED_RSCRIPT` and `NE_ALIGNED_LIGER_RSCRIPT` (see the [reproduction guide](../../../../../reproduction/README.md#benchmark-environments)):

```bash
bash scripts/setup_external_envs.sh
```

`NE_ALIGNED_RSCRIPT` is **required**; there is no silent fallback to whatever `Rscript` is on `PATH` (see `rscript_command` in `../r.py`). BC-tSNE is installed from a pinned upstream commit by `envs/r/install-cran-extras.sh`, which applies the saved patch `envs/r/bcTSNE-82e8be9.patch` and removes the prebuilt objects upstream ships.

## Verification status

✅ **All five methods verified end-to-end** (Python plumbing + real R) on **R 4.5.3 / Seurat 5.5.1 / rliger 1.0.1 / bcTSNE 0.11.1 / hdf5r 1.3.12**, both as direct `compute_*` calls and via the CLI `run-task` on real nCoV raw-count data. The Python plumbing and t-SNE tails additionally have offline unit tests through a mock Rscript (`tests/test_r_subprocess.py`), so CI stays green without R.

| script | method(s) | status |
|---|---|---|
| `io.R` | shared HDF5 I/O | ✅ verified (`hdf5r` 1.3.12) |
| `seurat.R` | `seurat-*` (v5 layers), `seurat3-*` (historical anchors) | ✅ verified (Seurat 5.5.1) |
| `liger1.R` | `liger1-2d`, `liger1-50-tsne` | ✅ verified (rliger **1.0.1**) |
| `liger.R` | `liger-2d`, `liger-50-tsne` | ✅ verified with rliger **2.2.1** / RcppPlanc 2.0.16 |
| `bc_tsne.R` | `bc-tsne` | ✅ verified (bcTSNE 0.11.1, installed per above) |
