# Benchmark package

- `pipeline.py` owns manifest expansion, task identity, local orchestration, readable provenance, aggregation, scoring, and validation.
- `data.py` loads the exact representation requested by the method registry.
- `sweep/registry.py` owns method families and representation routing.
- `sweep/tasks.py` executes one task and validates resumable artifacts.
- `sweep/aggregation.py` constructs aggregate caches and dispatches scoring.
- `scoring.py` interprets caches and computes metrics.

The public entry point is `scripts/run_benchmark.py`; `scripts/submit_benchmark.sh` only expresses the parallel Slurm job layout around the same commands.

## OT-tSNE initializations

The `ot_init` setting picks the initialization of every OT-tSNE layout. The main benchmark uses `eot`. Only OT-tSNE reads it: the other methods start from their own initialization, and their run ids do not depend on it.

- `random`: Gaussian points from the task seed.
- `spectral`: diffusion coordinates of the final OT affinity graph, as openTSNE's spectral initialization. It is computed on the sparse graph.
- `eot`: the transport plan's first two non-trivial singular vectors. Every diagonal variant on the same plan starts from the same layout.
- `pca`: the first two principal components of the joint PCs, as openTSNE's `initialization="pca"`. The batch effect is in this initial layout.

The last three are scaled like openTSNE's (first axis standard deviation 1e-4) and get a 1% jitter from the task seed. That jitter is the only thing that differs between seeds in the init; `ot_init_jitter = 0.0` turns it off.

There is no fallback. If the spectral init cannot be computed, for example because a cell has no edges, the task fails with the reason. Each task file records the init it used as `init_used`. The aggregate step refuses tasks whose `init_used` differs from the manifest's `ot_init`, and the scored rows carry both columns, so a random layout can never be scored as spectral.

`configs/ot_init_random.toml`, `configs/ot_init_spectral.toml`, `configs/ot_init_eot.toml` and `configs/ot_init_pca.toml` rerun the five OT-tSNE variants of the main experiment at the default point, one initialization each, in their own results folders. On a Slurm cluster:

    bash scripts/submit_benchmark.sh --manifest configs/ot_init_spectral.toml
