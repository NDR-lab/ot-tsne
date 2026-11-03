"""Run MultiMAP in its isolated NumPy 1.x environment.

Accept input/output NPZ paths. Inputs contain batches, batch IDs, and JSON
parameters; output embedding rows follow batch 0 then batch 1. Keep module-level
imports limited to the standard library and NumPy; do not import ``ne_aligned_utils``.
"""

import json
import sys
import time

import numpy as np


def _scale_per_batch(joint: np.ndarray, n0: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Match Scanpy's sample-standardization with a stable variance calculation."""
    scaled = np.empty_like(joint)
    constant_counts = []
    for start, stop in ((0, n0), (n0, joint.shape[0])):
        block = joint[start:stop].astype(np.float64)
        mean = block.mean(axis=0)
        # Scanpy uses sample variance (n - 1 denominator).
        std = block.std(axis=0, ddof=1)
        degenerate = ~(std > 0)
        constant_counts.append(int(degenerate.sum()))
        std[degenerate] = 1.0
        scaled[start:stop] = ((block - mean) / std).astype(joint.dtype)
    return scaled, (constant_counts[0], constant_counts[1])


def _prepare_batches(x0, x1, *, scale, use_native_scaling):
    """Preserve precision for Scanpy's native variance calculation.

    Tiny nonzero MNIST border values can survive a float32 cast while their squares
    underflow to zero. Scanpy computes E[X²] - E[X]², yielding negative variances
    and NaNs in PCA. Keep float64 for native scaling and zero exactly constant
    columns within each batch, as native centering should do. Even float64's
    E[X²] - E[X]² can be negative for a nonzero constant. No features are removed.
    Retain the old float32 contract for explicit compatibility runs.
    """
    dtype = np.float64 if use_native_scaling else np.float32
    joint = np.concatenate([x0, x1], axis=0).astype(dtype)
    n0 = len(x0)
    constant_counts = (0, 0)
    if scale:
        if use_native_scaling:
            counts = []
            for block in (joint[:n0], joint[n0:]):
                constant = np.ptp(block, axis=0) == 0
                counts.append(int(constant.sum()))
                block[:, constant] = 0.0
            constant_counts = tuple(counts)
        else:
            joint, constant_counts = _scale_per_batch(joint, n0)
    return joint, constant_counts


def _warmup_multimap(anndata, MultiMAP, n_components: int) -> None:
    """Call MultiMAP on a small input so required JIT kernels compile outside algorithm timing."""
    n = 24
    rng = np.random.RandomState(0)
    tiny = anndata.AnnData(X=rng.rand(n, 6).astype(np.float32))
    tiny.obs_names = [f"warm_{i}" for i in range(n)]
    tiny.var_names = [f"f_{i}" for i in range(6)]
    tiny.obs["batch"] = np.asarray([f"batch_{i % 2}" for i in range(n)], dtype=object)
    tiny.obs["batch"] = tiny.obs["batch"].astype("category")
    MultiMAP.Batch(
        tiny,
        batch_key="batch",
        seed=0,
        random_state=0,
        scale=False,
        n_components=int(n_components),
        n_neighbors=4,
        n_epochs=2,
    )


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: python multimap_runner.py <input.npz> <output.npz>")
    input_path, output_path = sys.argv[1], sys.argv[2]

    with np.load(input_path, allow_pickle=False) as handle:
        x0 = np.asarray(handle["x0"], dtype=np.float64)
        x1 = np.asarray(handle["x1"], dtype=np.float64)
        batch_ids = np.asarray(handle["batch_ids"], dtype=np.int64)
        params = json.loads(str(handle["params_json"]))

    n0, n1 = int(x0.shape[0]), int(x1.shape[0])
    if batch_ids.shape[0] != n0 + n1:
        raise SystemExit(f"batch_ids has {batch_ids.shape[0]} entries, expected {n0 + n1}.")

    import anndata
    import MultiMAP

    # Compile through the real MultiMAP entry point before algorithm timing.
    # Report time_warmup separately; a failed warmup records NaN without aborting the method.
    warmup_seconds = float("nan")
    try:
        warmup_start = time.perf_counter()
        _warmup_multimap(anndata, MultiMAP, int(params.get("n_components", 2)))
        warmup_seconds = time.perf_counter() - warmup_start
    except Exception:
        pass

    # Start algorithm timing after imports, input loading, and JIT warmup.
    algo_start = time.perf_counter()

    scale = bool(params.get("scale", True))
    use_native_scaling = bool(params.get("use_native_scaling", True))
    joint, constant_counts = _prepare_batches(
        x0, x1, scale=scale, use_native_scaling=use_native_scaling
    )

    adata = anndata.AnnData(X=joint)
    # Unique obs names matter: Batch() rebuilds the original row order by looking
    # each name up in the per-batch concatenation.
    adata.obs_names = [f"cell_{index}" for index in range(joint.shape[0])]
    adata.var_names = [f"feature_{index}" for index in range(joint.shape[1])]
    adata.obs["batch"] = np.asarray(
        [f"batch_{value}" for value in batch_ids.tolist()], dtype=object
    )
    adata.obs["batch"] = adata.obs["batch"].astype("category")

    seed = int(params.get("seed", 0))
    center_batches = bool(params.get("center_batches", True))
    batch_kwargs = {
        "batch_key": "batch",
        "seed": seed,
        # MultiMAP uses a separate seed for its final layout.
        "random_state": seed,
        "scale": bool(scale and use_native_scaling),
        "n_components": int(params.get("n_components", 2)),
    }
    if not center_batches:
        if scale:
            raise ValueError("center_batches=False requires scale=False.")

        # Sensitivity mode: preserve batch-wide translations before MultiMAP's joint PCA.
        from sklearn.decomposition import TruncatedSVD

        def uncentered_svd(batch_adata):
            n_components = min(50, batch_adata.n_obs - 1, batch_adata.n_vars - 1)
            reducer = TruncatedSVD(n_components=n_components, random_state=seed)
            batch_adata.obsm["X_pca"] = reducer.fit_transform(batch_adata.X)

        batch_kwargs["dimred_func"] = uncentered_svd
    # Omitted layout parameters reach MultiMAP unchanged, enabling its native profile.
    for key, cast in (
        ("n_neighbors", int),
        ("min_dist", float),
        ("spread", float),
        ("n_epochs", int),
        ("init", str),
    ):
        value = params.get(key)
        if value is not None:
            batch_kwargs[key] = cast(value)

    MultiMAP.Batch(adata, **batch_kwargs)
    embedding = np.asarray(adata.obsm["X_multimap"], dtype=np.float64)
    # MultiMAP is timed as one fused integration-and-layout method.
    algo_seconds = time.perf_counter() - algo_start

    np.savez_compressed(
        output_path,
        embedding=embedding,
        n_constant_features=np.asarray(max(constant_counts)),
        n_constant_features_batch0=np.asarray(constant_counts[0]),
        n_constant_features_batch1=np.asarray(constant_counts[1]),
        n_features_used=np.asarray(int(joint.shape[1])),
        center_batches=np.asarray(center_batches),
        time_total=np.asarray(float(algo_seconds)),
        time_warmup=np.asarray(float(warmup_seconds)),
    )


if __name__ == "__main__":
    main()
