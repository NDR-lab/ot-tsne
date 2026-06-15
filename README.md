# ne-aligned

Research code for aligned nonlinear embeddings across paired datasets. The main pieces are now grouped by job: embedding methods in `src/ne_aligned_utils/methods.py`, metric formulas in `src/ne_aligned_utils/metrics.py`, and small CLIs for benchmark reports.

## Layout

- `src/ne_aligned_utils/methods.py`: PCA, t-SNE, Seurat, Harmony, scVI, EOT, EOT+t-SNE, and OT-tSNE implementations.
- `src/ne_aligned_utils/metrics.py`: neighborhood purity, silhouette, iLISI, kBET, shared-cell-type batch mixing, concordance, FOSCTTM, and Rand-index metrics.
- `src/ne_aligned_utils/benchmark_metrics.py`: aggregate-cache metric reporting with per-metric unavailable reasons.
- `src/ne_aligned_utils/benchmark_reporting.py`: compact overview CSV/figure generation used by the overview notebook.
- `scripts/compute_benchmark_metrics.py`: computes per-run and summary metric CSVs from aggregate `.npz` caches.
- `scripts/generate_benchmark_overview.py`: writes `results/benchmark_overview/*` and executes `notebooks/benchmarking_new_embedding_overview.ipynb`.
- `notebooks/`: analysis, benchmark, tutorial, and smoke-test notebooks.
- `tests/`: unit and regression tests.

## Setup

```bash
uv sync --frozen --group dev
```

Some methods are optional in practice: Harmony requires `harmonypy`, scVI requires raw count matrices and `scvi-tools`, and the legacy CNE comparison expects the external `contrastive-ne` package if you choose to run that path.

## Checks

```bash
uv run --with pytest python -m pytest -q
uv lock --check
uv build
git diff --check
```

## Reports

Compute all applicable metrics for one or more aggregate caches:

```bash
uv run python scripts/compute_benchmark_metrics.py \
  --aggregate-cache benchmarking_new/nCov/n_per_batch_all__q_2__epochs_200__tsne_pca_50/nCoV_benchmark_embeddings.npz \
  --output-csv results/benchmark_overview/ncov_metrics.csv
```

Regenerate the compact overview artifacts and executed overview notebook:

```bash
uv run python scripts/generate_benchmark_overview.py
```

Generated figures are ignored by git. The compact CSV summaries under `results/benchmark_overview/` are small enough to review.
