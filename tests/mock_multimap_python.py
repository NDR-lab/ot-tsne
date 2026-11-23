"""Emulate the isolated MultiMAP launcher with deterministic NPZ embeddings.

Accept runner, input, and output paths; copy joint features and pad to the requested
width. NE_ALIGNED_MOCK_MULTIMAP_ALGO_SECONDS optionally supplies child timing.
"""

import json
import os
import sys

import numpy as np


def main() -> None:
    _script_path, input_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with np.load(input_path, allow_pickle=False) as handle:
        x0 = np.asarray(handle["x0"], dtype=float)
        x1 = np.asarray(handle["x1"], dtype=float)
        params = json.loads(str(handle["params_json"]))

    n_components = int(params["n_components"])
    joint = np.concatenate([x0, x1], axis=0)
    n_total, n_features = joint.shape
    embedding = np.zeros((n_total, n_components), dtype=float)
    take = min(n_components, n_features)
    embedding[:, :take] = joint[:, :take]

    reported = os.environ.get("NE_ALIGNED_MOCK_MULTIMAP_ALGO_SECONDS")
    extra = {}
    if reported is not None:
        extra["time_total"] = np.asarray(float(reported))
        extra["time_warmup"] = np.asarray(0.0)
    np.savez_compressed(output_path, embedding=embedding, **extra)


if __name__ == "__main__":
    main()
