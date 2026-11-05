# Paper figures

Build every benchmark figure from completed results (no method is run):

```bash
uv run python scripts/make_figures.py \
  --results-root <benchmark root> --supplementary-root <supplementary root> \
  --output-dir <new empty directory>
```

Options: `--families` (default: all of `metrics runtime ranks qualitative composition`),
`--datasets` (default: the seven canonical datasets), `--top-r` (default: `3 2`),
`--dry-run` (validate inputs, write nothing). The methods schematic is built on its own with
`--families methods --method-arrays <npz> --output-dir <dir>`. Synthetic figures:
`scripts/figures/synthetic.py`.

## What each family writes

| Family | Files |
|---|---|
| metrics | `f_benchmark_kbet_purity`, `f_benchmark_ilisi_silhouette` (one row per dataset), `f_pooled_benchmark` (mean over datasets, SD across datasets), `metric-summary*.csv` |
| runtime | `runtime/runtime_all_methods`, `runtime/runtime_ot_tsne_breakdown`, four `runtime_all_methods_device*` overlays when GPU timings exist, audit tables |
| ranks | `ranks/rank-{raw,minmax}` (all metrics, bio, mixing; each column in its own order, mean rank printed), `ranks/top-{r}-{raw,minmax}` for each r |
| qualitative | `qualitative/embeddings-<dataset>`, `qualitative/embeddings-all` and `-grouped` (datasets stacked) |
| composition | `composition/batch-composition-{1,2}` (cell types per batch), `cell-type-counts.csv` |

Every build writes `figure-manifest.json` last: input hashes, the tables behind each figure, and
the hash and size of every output. Two manifests can be compared with
`scripts/figures/compare_manifests.py`.

Pooled summaries (pooled figure, ranks, top-r, composition) use only the seven canonical
datasets; `hematopoietic_shared` can be drawn as a grid row but never joins a pool.

## Layout

- `registry.py`: one entry per method: name, colour, marker, family, figure order. A key is the
  benchmark's method name; a non-default configuration gets a suffix (`multimap-k15`).
- `selection/benchmark.py`: the reported configuration (`rel_reg`, k, backend, device). The only
  place these values are named.
- `inputs/`: read completed results (metric tables, task timings, aggregate caches).
- `summaries/`: seed summaries, pooling, ranks, top-r, runtime medians and validation.
- `rendering/`: presentation constants (`styles.py`, `runtime.py`) and the factories that turn the
  registry into drawing inputs (`presentation.py`); legend, heading and point primitives.
- `figures/`: one module per figure family; each draws from preloaded tables only.
- `build.py`: preflight (read and validate everything), build, manifest.
- `layout.py`: page geometry, the paper style and the print-size checks on save.
