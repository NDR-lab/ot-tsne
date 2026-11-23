"""Emulate the R launcher using deterministic HDF5 embeddings without requiring R.

Accept script, input, and output paths; copy leading joint feature columns and pad
to the requested width. NE_ALIGNED_MOCK_R_ALGO_SECONDS optionally supplies child timing.
"""

import os
import sys

import h5py
import numpy as np


def main() -> None:
    _script_path, input_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with h5py.File(input_path, "r") as handle:
        x0 = np.asarray(handle["x0"], dtype=float)
        x1 = np.asarray(handle["x1"], dtype=float)
        dim = int(handle["params"].attrs["dim"])

    joint = np.concatenate([x0, x1], axis=0)
    n_total, n_features = joint.shape
    embedding = np.zeros((n_total, dim), dtype=float)
    take = min(dim, n_features)
    embedding[:, :take] = joint[:, :take]

    reported = os.environ.get("NE_ALIGNED_MOCK_R_ALGO_SECONDS")
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("embedding", data=embedding)
        if reported is not None:
            group = handle.create_group("timings")
            # bc_tsne.R reports a total (it is fused); seurat.R and liger.R report the
            # integration, which the Python side adds its backend call to.
            key = "time_total" if "bc_tsne" in str(_script_path) else "time_pre_embedding"
            group.attrs[key] = float(reported)


if __name__ == "__main__":
    main()
