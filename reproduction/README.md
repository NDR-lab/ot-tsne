# Reproducing the OT-tSNE benchmark

Install the [main Python environment](../README.md#installation) first. Run all commands
below from the repository root. The benchmark compares OT-tSNE and its variants with
integration and visualization baselines on seven two-batch single-cell datasets.

## OT-tSNE input and memory considerations

The default uses balanced transport with uniform mass per observation. Differences in
population proportions and populations present in only one batch should therefore be
considered when interpreting the alignment. The transport calculation stores a dense
`n0 × n1` plan, so memory use grows with the product of the batch sizes.

## Benchmark environments

OT-tSNE and most benchmark methods run in the main Python environment managed by `uv`. Four comparison methods require isolated environments because their Python, R or compiled dependencies conflict with the main environment.

Run all commands below from the project root.

### Requirements

The automated setup supports Linux and macOS on x86-64 or ARM64. Windows is not supported by the setup script. **Linux x86-64 is the reference platform**: there, the conda environments are created from exact package lists (`envs/linux-64/*.lock`) that reproduce the environments behind the published benchmark. On other platforms they are solved from `environment.yml`, which pins the method versions but lets conda choose platform builds.

Install or provide:

- [uv](https://docs.astral.sh/uv/)
- Bash, `curl`, Git and `tar`
- a C++ compiler available as `g++` or `clang++`, or through the `CXX` environment variable
- internet access to conda-forge, bioconda, CRAN and GitHub
- approximately 12 GB of free disk space for all environments

The installer downloads micromamba 2.8.1 when no usable micromamba executable is available.

### Installation

Install the main Python environment first:

```bash
uv sync --locked
```

`uv.lock` fixes every package version and hash, and `.python-version` fixes the interpreter (Python 3.11.15; uv downloads it if needed). `--locked` fails instead of re-resolving if the lock is out of date. Add `--group dev` for notebooks.

Install the isolated method environments:

```bash
bash scripts/setup_external_envs.sh
```

The first installation can take tens of minutes because several R packages and rc-tSNE are compiled from source. Environments that already exist are never updated in place: a rerun reuses them as they are, installs any pinned R package that is missing, and checks the versions. To rebuild an environment, delete it or choose new locations (see below).

When setup finishes, load the generated launcher variables into the current shell and verify every environment:

```bash
source .ne-aligned-env.sh
uv run python scripts/check_environment.py
```

Run `source .ne-aligned-env.sh` in each new terminal or batch job before starting the benchmark. The checker prints one `OK` line for each component and exits with a nonzero status when a required component is unavailable or has the wrong version.

To verify only the main Python environment, without requiring the external methods, run:

```bash
uv run python scripts/check_environment.py --allow-missing-external
```

### Installed environments

| methods | environment specification | launcher variable |
|---|---|---|
| Seurat, Seurat 3-style anchors, LIGER 1 and BC-tSNE | `envs/r/environment.yml`, `envs/r/install-cran-extras.sh` | `NE_ALIGNED_RSCRIPT` |
| LIGER 2 | `envs/liger/environment.yml`, `envs/liger/install-cran-extras.sh` | `NE_ALIGNED_LIGER_RSCRIPT` |
| MultiMAP | `envs/multimap/pyproject.toml`, `uv.lock`, `.python-version` (separate uv project; MultiMAP needs numpy < 2) | `NE_ALIGNED_MULTIMAP_PYTHON` |
| rc-tSNE | `envs/rctsne/environment.yml`, vendored source and `scripts/setup/build_rctsne.sh` | `NE_ALIGNED_RCTSNE_BIN` |

On linux-64, `envs/linux-64/{rverify,liger2,rctsne}.lock` list every conda package by URL and MD5 checksum; they are exports of the environments that produced the published results. R packages that are not on conda are installed from CRAN at exact versions by the `install-cran-extras.sh` scripts (rliger 1.0.1 and its CRAN-only dependencies for `rverify`; HighFive 3.3.0 and rliger 2.2.1 for `liger2`). BC-tSNE 0.11.1 is built from upstream commit `82e8be9` with the saved patch `envs/r/bcTSNE-82e8be9.patch`. MultiMAP is pinned to commit `681e608`. rc-tSNE is compiled with the C++ compiler on `PATH` (the published results used GCC 8.5.0 on Linux).

### Custom installation locations

Set any of these variables before running `scripts/setup_external_envs.sh`:

| variable | default | purpose |
|---|---|---|
| `NE_ALIGNED_R_ROOT` | `~/.local/opt/ne-aligned-r` | micromamba executable, root and R/FFTW environments |
| `NE_ALIGNED_MULTIMAP_VENV` | `~/.venvs/ne-aligned-multimap` | MultiMAP Python environment |
| `NE_ALIGNED_LAUNCHER_FILE` | `./.ne-aligned-env.sh` | generated shell file containing launcher variables |
| `NE_ALIGNED_MICROMAMBA` | automatically detected | existing micromamba executable to use |
| `NE_ALIGNED_R_TIMEOUT` | `14400` | maximum runtime in seconds for one R-method task |

For example:

```bash
export NE_ALIGNED_R_ROOT=/shared/software/ot-tsne/r
export NE_ALIGNED_MULTIMAP_VENV=/shared/software/ot-tsne/multimap
export NE_ALIGNED_LAUNCHER_FILE=/shared/software/ot-tsne/environment.sh
bash scripts/setup_external_envs.sh
source /shared/software/ot-tsne/environment.sh
```

The default four-hour R timeout accommodates the slowest recorded BC-tSNE runs. Set `NE_ALIGNED_R_TIMEOUT` before installation if your workload requires a different limit; values that are too small terminate otherwise valid tasks.

### Cluster installation

Install the environments once, on a node with network access and a compiler (if your site does not allow work on login nodes, use an interactive or batch job). Place them on storage visible to every compute node, and source the generated launcher file in each job:

```bash
source /shared/software/ot-tsne/environment.sh
uv run python scripts/check_environment.py
bash scripts/submit_benchmark.sh
```

The launcher file contains absolute paths. Rerun the setup script if the installation is moved to a different location.

### Troubleshooting

If setup fails, correct the reported network, compiler or filesystem problem and rerun:

```bash
bash scripts/setup_external_envs.sh
```

Common issues:

- **Launcher variables are missing:** source `.ne-aligned-env.sh` before running the checker or benchmark.
- **rc-tSNE cannot compile:** install `g++` or `clang++`, or set `CXX` to a working C++ compiler.
- **A cluster compute node cannot download packages:** run setup on a node that has network access and use a shared installation path.
- **An R task reaches the time limit:** increase `NE_ALIGNED_R_TIMEOUT`, rerun setup to regenerate the launcher file, and resume the benchmark.
- **An environment was partially created:** delete that environment's directory (under `$NE_ALIGNED_R_ROOT/mmroot/envs/` or the MultiMAP venv) and rerun the setup command.

Keep the supplied version pins when repairing an environment. Changing a method version changes the benchmark configuration and can make results incomparable with the release.

### Reference hardware

The reference results were computed on Intel Xeon Gold 6240 nodes with four threads per
benchmark task. Layouts can vary across CPU models and thread counts; compare metrics
within seed-to-seed variation rather than expecting identical coordinates.

---

## Benchmark implementations

The benchmark uses the implementations and versions below.

Exact dependency resolution for the main Python environment is recorded in `uv.lock`. Isolated external environments are pinned under `envs/`.

| Method | Implementation used |
|---|---|
| OT-tSNE / EOT variants | This repository; POT 0.9.6.post1 for optimal transport; openTSNE 1.0.4 for the common t-SNE stage |
| Joint PCA / joint t-SNE | scikit-learn 1.8.0; openTSNE 1.0.4 |
| Harmony | harmonypy 2.0.0 |
| Scanorama | Scanorama 1.7.4 |
| scVI | scvi-tools 1.4.2 (Python 3.11, as pinned by `.python-version`; uv.lock resolves 1.4.3 on Python 3.12+) |
| Seurat | Seurat 5.5.1; SeuratObject 5.4.0; R 4.5.3 |
| LIGER | rliger 1.0.1 (LIGER 1) and rliger 2.2.1 (LIGER 2) |
| MultiMAP | upstream MultiMAP pinned to commit `681e608c45cdb6b139dfb6700e40c7520bc6096d` |
| rc-tSNE | revised conditional t-SNE implementation vendored from commit `afda44b6d04066077d6fe46da1127de28de6a034`; FFTW 3.3.11 |
| BC-tSNE | bcTSNE 0.11.1, pinned to upstream commit `82e8be9ff42f03ace8f7112d36d4a3818485aace` |
| Ref-tSNE | in-repository implementation of the Poličar et al. reference t-SNE procedure; openTSNE 1.0.4 |

Preprocessing uses **Scanpy 1.11.5** on the pinned Python 3.11 (`uv.lock` resolves 1.12.1 on Python 3.12+, which is not the published configuration).

---

## Obtain the data

The checkout includes only the compact nCoV example inputs under
`data/nCoV/preprocessed/`. Full benchmark inputs, including count matrices and metadata,
must be obtained separately. The required raw file names for each dataset are declared
in [configs/benchmark.toml](../configs/benchmark.toml).

For benchmark preprocessing, place raw files in the layout below. The preparation script
creates `data/benchmark/<dataset>/` when run; this generated directory is not included
in the checkout.

```text
data/
├── <dataset>/
│   └── raw/
│       └── <source files>
└── benchmark/
    └── <dataset>/
        └── <prepared files>
```

To keep the data elsewhere, set:

```bash
export NE_ALIGNED_RAW_ROOT=/path/to/release-data
export NE_ALIGNED_DATA_ROOT=/path/to/release-data/benchmark
```

`NE_ALIGNED_RAW_ROOT` should contain the `<dataset>/raw/` directories.

---

## Prepare a dataset

Check that the expected raw files are present:

```bash
uv run python scripts/prepare_data.py PBMC --check-only
```

Prepare the dataset:

```bash
uv run python scripts/prepare_data.py PBMC
```

Prepared files are written to:

```text
data/benchmark/PBMC/
```

Every dataset follows the same preprocessing pipeline:

1. cell and gene filtering;

2. batch-aware selection of 1,000 genes from raw counts;

3. library-size normalization;

4. `log1p`;

5. one joint 50-component PCA over both batches.

The prepared directory contains:

```text
X1.npy
X2.npy
X1_counts.npy
X2_counts.npy
X1_pca50.npy
X2_pca50.npy
labels1.npy
labels2.npy
cell_ids1.npy
cell_ids2.npy
cells_metadata.csv
features.txt
combined_1000genes.h5ad
preprocessing_manifest.json
```

`preprocessing_manifest.json` records hashes of the source files and generated content.

Floating-point PCA bytes can vary across numerical platforms. Use the same prepared
inputs as the reference run when comparing preprocessing hashes.

---

## Smoke test

Run a small end-to-end benchmark on nCoV:

```bash
uv run python scripts/run_benchmark.py reproduce \
    --dataset nCoV \
    --profile smoke
```

The smoke profile runs each method once with seed 0 and canonical parameters, then:

```text
run
→ aggregate
→ score
→ validate
```

The current nCoV smoke profile contains **23 tasks**, with four CPU threads requested per task by default. Runtime depends on the hardware and method environments.

It exercises the complete pipeline, including the external method environments.

---

## Reproduce the benchmark

### Local run

Inspect the planned tasks without executing anything:

```bash
uv run python scripts/run_benchmark.py plan --dataset nCoV
```

Run the complete benchmark for one dataset:

```bash
uv run python scripts/run_benchmark.py reproduce --dataset nCoV
```

`reproduce`:

1. runs all missing task embeddings;

2. reuses valid completed task files;

3. aggregates compatible runs;

4. scores the resulting embeddings;

5. validates the outputs.

An interrupted run can therefore be resumed by running the same command again.

### Cluster run

On a Slurm cluster:

```bash
bash scripts/submit_benchmark.sh
```

The script submits independent jobs for each dataset and experiment, followed by one dependent finalization job for scoring and validation.

The repository contains no site-specific Slurm settings. Supply account, partition, wall time, memory, CPUs, and other scheduler options through your cluster's usual `SBATCH_*` environment variables or through a local wrapper.

Some comparison methods require several hours on the largest datasets.

### Results

Results are written under:

```text
results/benchmark/benchmark--<config-hash>/
```

where `<config-hash>` is the first 12 characters of the benchmark-configuration hash.

Editing `configs/benchmark.toml` changes the hash and therefore creates a separate result tree.

To write results elsewhere:

```bash
export NE_ALIGNED_RESULT_ROOT=/path/to/results
```

The result layout is:

```text
benchmark--<config-hash>/
├── metrics.csv
├── task_metadata.csv
├── metrics/
│   └── <dataset>--all-experiments--all-devices.csv
└── <dataset>/
    └── <experiment>/
        └── <method>--<arm-id>/
            ├── tasks/
            │   └── <configuration>/
            │       └── <task>.npz
            ├── aggregate.npz
            └── metrics.csv
```

The main files are:

- `task_metadata.csv` — one readable row per task, including method, seeds, parameters, hashes, and scientific identifiers;

- `<task>.npz` — the 2D embedding and metadata for one seed/configuration;

- `aggregate.npz` — all compatible seeds/configurations for one method arm;

- `metrics.csv` — scored rows for the corresponding scope.

---

## Build figures from completed results

The figure entry point reads completed results; it never runs a benchmark method.

```bash
uv run python scripts/make_figures.py \
    --results-root /path/to/completed/results \
    --supplementary-root /path/to/supplementary \
    --output-dir /path/to/new/output
```

By default it builds every benchmark family for the seven canonical datasets:

- `metrics`: per-dataset metric grids (`f_benchmark_kbet_purity`, `f_benchmark_ilisi_silhouette`)
  and the pooled grid `f_pooled_benchmark` (mean over datasets, SD across datasets);
- `runtime`: runtime of every method per dataset, and OT-tSNE's stage breakdown;
- `ranks`: aggregate ranks (all metrics, bio conservation, batch mixing; each column in its own
  order with the mean rank printed) and top-r finish counts for each `--top-r` (default `3 2`);
- `qualitative`: embedding galleries per dataset and stacked;
- `composition`: cell-type composition by batch.

The output directory must be new or empty (default `<results-root>/paper_figures/`). Use
`--families` to build a subset (for one dataset with only its metric table, `--datasets nCoV
--families metrics`), and `--dry-run` to validate inputs without writing. Each build ends with
`figure-manifest.json`. The methods schematic is a separate
`--families methods --method-arrays /path/to/arrays.npz --output-dir /path/to/new/output` run.

### Figure notebooks

[method_figure.ipynb](../notebooks/method_figure.ipynb) recomputes the small methods
schematic example. [reproduce_paper_figures.ipynb](../notebooks/reproduce_paper_figures.ipynb)
builds embedding and ranking figures from separately supplied handoff data; its
configuration cell sets the data and output paths.
