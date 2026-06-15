# ne-aligned

Research code for OT-tSNE, a method for building a joint two-dimensional embedding of two related datasets. The method first computes an optimal transport plan between the datasets, turns that plan into a joint affinity graph, sparsifies the graph, and then runs t-SNE on the resulting affinities.

The repository also contains baseline embedding methods, benchmark notebooks, and metric scripts used for the accompanying experiments.

## Installation

This project uses `uv` for environment management.

```bash
uv sync --frozen --group dev
```

Optional methods have extra practical requirements: Harmony uses `harmonypy`, scVI needs raw count matrices and `scvi-tools`, and some legacy comparison notebooks expect the external `contrastive-ne` package.

## Minimal Usage

```python
from ne_aligned_utils.methods import compute_ot_tsne_embedding_from_arrays

embedding = compute_ot_tsne_embedding_from_arrays(
    x0,
    x1,
    eps_times=10.0,
    effective_k=15,
    random_state=0,
    n_epochs=200,
)

y0 = embedding[: len(x0)]
y1 = embedding[len(x0) :]
```

For callers that already have an OT plan, use `compute_ot_tsne_embedding_from_plan`.

## Repository Layout

- `src/ne_aligned_utils/methods.py`: OT-tSNE and baseline embedding methods.
- `src/ne_aligned_utils/metrics.py`: benchmark metric formulas.
- `src/ne_aligned_utils/benchmark_metrics.py`: metric reporting from aggregate embedding caches.
- `src/ne_aligned_utils/benchmark_reporting.py`: compact benchmark overview tables and figures.
- `scripts/compute_benchmark_metrics.py`: command line metric runner.
- `scripts/generate_benchmark_overview.py`: command line overview generator.
- `notebooks/`: dataset preparation, benchmark, tutorial, and analysis notebooks.
- `tests/`: unit and regression tests.

## Data

Large datasets, raw counts, generated figures, Slurm logs, and embedding caches are not tracked. See `data/README.md` for the expected local directory layout.

Compact benchmark summaries are tracked under `results/benchmark_overview/`.

## Benchmarks

Compute metrics for one or more aggregate caches:

```bash
uv run python scripts/compute_benchmark_metrics.py \
  --aggregate-cache benchmarking_new/nCov/n_per_batch_all__q_2__epochs_200__tsne_pca_50/nCoV_benchmark_embeddings.npz \
  --output-csv results/benchmark_overview/ncov_metrics.csv
```

Regenerate the overview CSVs and executed overview notebook:

```bash
uv run python scripts/generate_benchmark_overview.py
```

## Checks

```bash
uv run python -m pytest -q
uv lock --check
uv build
git diff --check
```
