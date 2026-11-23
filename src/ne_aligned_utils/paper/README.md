# Paper figures

Build every benchmark figure from completed results (no method is run):

```bash
uv run python scripts/make_figures.py \
  --results-root <benchmark root> --supplementary-root <supplementary root> \
  --output-dir <new empty directory>
```

Options: `--families` (default: all of `metrics runtime ranks qualitative composition`),
`--datasets` (default: the seven canonical datasets), `--top-r` (default: `3 2`),
`--dry-run` (validate inputs, write nothing).

Every figure shows one configuration. By default that is the reported benchmark: `rel_reg` 1,
k 15, "or" symmetrisation. `--rel-reg`, `--effective-k` and `--graph-symmetrization` build the
same figure set for another configuration, from a results root that holds it; the manifest
records which one was used. Ranks always use native MultiMAP and refuse the matched-k arm.

The methods schematic is built on its own with
`--families methods --method-arrays <npz> --output-dir <dir>`.

Other figure scripts, all under `scripts/figures/`:

| Script | Draws |
|---|---|
| `synthetic.py --source-root <checkout> --output-dir <dir>` | the MNIST orthogonal-batch-effect figures, from saved runs |
| `parallel_synthetic.py --run-root <run> --output-dir <dir>` | the MNIST parallel-batch-effect figures |
| `sensitivity.py`, `hyperparameters.py` | sensitivity and initialisation figures (below) |
| `compare_manifests.py` | the difference between two `figure-manifest.json` files |
| `palette_validation.py` | contrast and colour-vision checks for a list of colours |

## What each family writes

| Family | Files |
|---|---|
| metrics | `f_benchmark_kbet_purity`, `f_benchmark_ilisi_silhouette` (one row per dataset), `f_pooled_benchmark` (mean over datasets, SD across datasets), `metric-summary*.csv` |
| runtime | `runtime/runtime_all_methods`, `runtime/runtime_ot_tsne_breakdown`, four `runtime_all_methods_device*` overlays when GPU timings exist, audit tables |
| ranks | `ranks/rank-{raw,minmax}` (all metrics, bio, mixing; each column in its own order, mean rank printed), `ranks/top-{r}-{raw,minmax}` for each r |
| qualitative | `qualitative/embeddings-<dataset>`, `qualitative/embeddings-all` and `-grouped` (datasets stacked), `qualitative/embeddings-seeds-<dataset>` (the same columns, two rows per seed), `qualitative/embeddings-ot-seeds-<dataset>` (OT-tSNE and its diagonal variants, two rows per seed) |
| composition | `composition/batch-composition-{1,2}` (cell types per batch), `cell-type-counts.csv` |

Every build writes `figure-manifest.json` last: input hashes, the tables behind each figure, and
the hash and size of every output. Two manifests can be compared with
`scripts/figures/compare_manifests.py`.

Pooled summaries (pooled figure, ranks, top-r, composition) use only the seven canonical
datasets; `hematopoietic_shared` can be drawn as a grid row but never joins a pool.

## Sensitivity trajectories

`scripts/figures/sensitivity.py` draws a bio metric against a mixing metric along one swept
setting: one panel per dataset, one trajectory per run configuration, arrows pointing the way the
setting grows, and the reported configuration in red. It reads metric tables only.

```bash
# rel_reg trajectories at the benchmark k, from the OT sensitivity grid's results root
uv run python scripts/figures/sensitivity.py --metrics <root> --sweep rel_reg \
    --fix effective_k=15 --output-dir <dir>
# k trajectories, one per rel_reg
uv run python scripts/figures/sensitivity.py --metrics <root> --sweep effective_k --output-dir <dir>
```

- `--sweep`: a metric-table column (`rel_reg`, `effective_k`, `rel_reg_m`) or a
  `method_settings` key (for example `beta`).
- `--fix column=value ...`: keep only those rows. Every other setting that differs between rows
  gives its own trajectory; settings are never averaged together. Seeds are averaged per run,
  and a seed seen twice in one run is an error.
- `--pairs`: `purity-kbet`, `silhouette-ilisi` (default both) and `bio-mixing` (each half's
  metrics averaged per seed).
- `--methods`, `--datasets`, `--ncols`. Writes `sensitivity-<sweep>-<pair>.png` and
  `sensitivity-<sweep>-points.csv` (the seed means and SDs drawn).

OT-tSNE variants keep their registry colours and differ by line style. Several trajectories of
one method take steps of its colour scale.

## Hyperparameter and initialisation figures

`scripts/figures/hyperparameters.py` shows changes from a reference configuration instead of
metric-vs-metric paths, with a blue (better) / red (worse) scale and no method colours. It reads
metric tables only; `--galleries` and `--agreement` also read the aggregate caches.

```bash
# OT-tSNE sensitivity grid; --main-root adds OT-tSNE's rank among all methods per grid point
uv run python scripts/figures/hyperparameters.py grid --metrics <sweep root> \
    [--main-root <main benchmark root>] --output-dir <dir>
# Initialisations: the random root is the reference; the main root holds the eot init
uv run python scripts/figures/hyperparameters.py init --metrics <random root> <main root> \
    <pca root> <spectral root> [--galleries] [--agreement] --output-dir <dir>
```

- `grid`: `sensitivity-pooled` (overall, bio and mixing) and `sensitivity-pooled-metrics` (the four
  metrics, then the overall score, the mean of bio and mixing): the mean change over datasets per
  rel_reg and k, with how many datasets rise or fall by at least 1 point and beyond 2 SE of the
  seed means. Also `sensitivity-lines` (the two one-at-a-time lines, one grey line per dataset), `sensitivity-datasets-<score>` (one grid per
  dataset), and the tables behind them. These are relative to the default.
- `grid --main-root` adds figures that are not: `sensitivity-placing-{minmax,raw}` (OT-tSNE's
  placing among all methods at each setting, every other method as reported, and its lead over
  the best other method), `sensitivity-placing-lines-*` (the same along the two lines),
  `sensitivity-strip-<score>` (OT-tSNE at all 25 settings against every other method, per
  dataset) and `sensitivity-absolute-<score>` (OT-tSNE's actual score along the lines, with the
  best and the median other method).
- `init`: `init-pooled`, `init-pooled-metrics` and `init-datasets-<score>` (change from the random init),
  `init-seed-spread` (seed SD of the scores, and with `--agreement` the share of 15 nearest
  neighbours two seeds' layouts agree on), `init-dots` (seed means and SDs), and with
  `--galleries` `init-seeds-<variant>-<dataset>` (one column per init, two rows per seed).
- A change counts when it is at least 1 point (`--min-change`, default 0.01) and larger than
  2 (`--noise-factor`) times the noise: by default the SE of the difference of seed means
  (`--noise se`), or the reference run's seed SD (`--noise default-sd`). The figures name the rule.
- Runs are never averaged across settings: rows that differ in anything but the swept setting
  (or the init) stop the script with the setting's name.

## Layout

- `registry.py`: one entry per method: name, colour, marker, family, figure order. A key is the
  benchmark's method name; a non-default configuration gets a suffix (`multimap-k15`).
- `selection/benchmark.py`: the reported configuration (`rel_reg`, k, backend, device). The only
  place these values are named.
- `inputs/`: read completed results; they never run a method or fill in a missing stage.
  - `current.load_canonical_frame()`: one row per (dataset, method, seed) in the reported
    configuration, from the metric CSVs, task metadata and optional supplementary arms.
  - `runtime.load_runtime_frame()`, `runtime.load_gpu_medians()`: CPU timings, and GPU medians
    kept apart from them (from the results root, else the supplementary root). Only
    `time_total` counts as a runtime; wall-clock `seconds` is never substituted.
  - `embeddings.load_reported_embedding()`, `load_seed_embeddings()`: coordinates from the
    aggregate cache a selected row names.
  - `methods.load_method_arrays()`, `synthetic.preflight()`: methods-schematic and synthetic inputs.
  - MultiMAP with its native defaults is `multimap`; the matched-neighbourhood run is
    `multimap-k15` (`current.multimap_key`). The native flag is read from the task files, never
    inferred from `n_neighbors`.
- `summaries/`: seed summaries, pooling, ranks, top-r, runtime medians and validation.
- `rendering/`: presentation constants (`styles.py`, `runtime.py`) and the factories that turn the
  registry into drawing inputs (`presentation.py`); legend, heading and point primitives.
- `figures/`: one module per figure family; each draws from preloaded tables only.
- `build.py`: preflight (read and validate everything), build, manifest.
- `layout.py`: page geometry, the paper style and the print-size checks on save.
