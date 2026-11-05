# Benchmark package

- `pipeline.py` owns manifest expansion, task identity, local orchestration, readable provenance, aggregation, scoring, and validation.
- `data.py` loads the exact representation requested by the method registry.
- `sweep/registry.py` owns method families and representation routing.
- `sweep/tasks.py` executes one task and validates resumable artifacts.
- `sweep/aggregation.py` constructs aggregate caches and dispatches scoring.
- `scoring.py` interprets caches and computes metrics.

The public entry point is `scripts/run_benchmark.py`; `scripts/submit_benchmark.sh` only expresses the parallel Slurm job layout around the same commands.
