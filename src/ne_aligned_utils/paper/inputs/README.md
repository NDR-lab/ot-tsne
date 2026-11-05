# Paper inputs

These modules read completed results. They never run a method or fill in a missing stage.

| Function | Reads | Returns |
|---|---|---|
| `current.load_canonical_frame()` | Dataset metric CSVs, task metadata, optional supplementary arms | One row per (dataset, method, seed) in the reported configuration, with method keys and labels |
| `runtime.load_runtime_frame()` | Saved CPU task files | The same selection for timings; supplementary arms replace their frozen slots |
| `runtime.load_gpu_medians()` | CUDA task files in the results root, else the October CUDA run in the supplementary root | GPU medians, kept separate from CPU timings |
| `embeddings.load_reported_embedding()` | The aggregate cache a selected row names | Seed-0 coordinates and the cache (labels, batches) |
| `methods.load_method_arrays()` | One NPZ with the six methods-schematic stages | Shape-checked arrays |
| `synthetic.preflight()` | Synthetic run configs, tables and caches | Validated synthetic inputs |

MultiMAP: the run with MultiMAP's native defaults is `multimap`; the frozen matched-neighbourhood
run is `multimap-k15` (`current.multimap_key`). The native flag is read from the task files,
never inferred from `n_neighbors`.

Only `time_total` is used as a runtime; the wall-clock `seconds` field is never substituted
(`summaries/task_timing.py`).
