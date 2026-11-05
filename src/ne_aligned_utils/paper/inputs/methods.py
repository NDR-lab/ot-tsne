"""Load completed methods-schematic arrays; never compute missing stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ARRAY_NAMES = ("x0", "x1", "plan", "affinity", "sparse", "embedding")


def load_method_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read one explicit numeric NPZ and validate the six drawing-stage shapes."""
    with np.load(Path(path), allow_pickle=False) as cache:
        missing = set(ARRAY_NAMES) - set(cache.files)
        if missing:
            raise ValueError(f"methods arrays missing {sorted(missing)}")
        arrays = {name: np.asarray(cache[name]) for name in ARRAY_NAMES}
    for name in ("x0", "x1"):
        value = arrays[name]
        if value.ndim != 2 or value.shape[1] < 2 or value.shape[0] < 1:
            raise ValueError(f"{name} must contain nonempty points with at least two coordinates")
    n0, n1 = len(arrays["x0"]), len(arrays["x1"])
    expected = dict(
        plan=(n0, n1),
        affinity=(n0 + n1, n0 + n1),
        sparse=(n0 + n1, n0 + n1),
        embedding=(n0 + n1, 2),
    )
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} expected shape {shape}, found {arrays[name].shape}")
    for name, value in arrays.items():
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite numeric values")
    if not (arrays["plan"] > 0).any() or not (arrays["affinity"] > 0).any():
        raise ValueError("plan and affinity need positive values for the existing heatmap scale")
    return arrays
