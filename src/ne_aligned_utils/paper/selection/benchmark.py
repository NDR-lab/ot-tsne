"""Which benchmark rows the paper reports: the one place that names the configuration.

Backend is not filtered here, and MultiMAP's native/matched-k identity is resolved by the input
adapter rather than inferred from k.
"""

from dataclasses import dataclass

import pandas as pd

CANONICAL_REL_REG = 1.0
CANONICAL_EFFECTIVE_K = 15.0

# Retired arms present in the raw tables (liger1, seurat3) and the split zero-diagonal ablation.
EXCLUDED_METHODS = frozenset({"liger1-2d", "seurat3-2d", "ot-zerodiag-split"})
OT_METHODS = frozenset(
    {
        "ot",
        "ot-gramdiag",
        "ot-knndiag",
        "ot-selfotmedian",
        "ot-zerodiag",
        "ot-zerodiag-split",
    }
)


@dataclass(frozen=True)
class BenchmarkSelectionPolicy:
    """The reported configuration: device, transport, OT regularisation, k and symmetrisation."""

    name: str = "canonical-benchmark"
    device: str = "cpu"
    transport: str = "balanced"
    rel_reg: float = CANONICAL_REL_REG
    effective_k: float = CANONICAL_EFFECTIVE_K
    graph_symmetrization: str = "or"
    #: Embedding backend of the reported layouts (methods without a backend axis always pass).
    backend: str = "tsne"
    excluded_methods: frozenset[str] = EXCLUDED_METHODS
    ot_methods: frozenset[str] = OT_METHODS


CURRENT_BENCHMARK = BenchmarkSelectionPolicy()


def reported_metric_rows(
    frame: pd.DataFrame,
    *,
    backend: str,
    no_backend: str,
    rel_reg: float,
    effective_k: float,
) -> pd.DataFrame:
    """Rows the metric grids summarise: one backend, and ``rel_reg``/k applied to every method.

    The canonical loader checks k only for OT and does not filter backend; this second filter
    applies both to every method that records them.
    """
    accepted = {backend, no_backend}
    data = frame[frame["backend"].isin(accepted)].copy()
    swept = data["rel_reg"].notna()
    data = data[~swept | (data["rel_reg"] == rel_reg)]
    has_k = data["effective_k"].notna()
    data = data[~has_k | (data["effective_k"] == effective_k)]
    return data


def is_canonical_row(
    row: pd.Series,
    policy: BenchmarkSelectionPolicy = CURRENT_BENCHMARK,
) -> bool:
    """Whether one benchmark row belongs to the reported configuration."""
    method = str(row["method"])
    if method in policy.excluded_methods:
        return False
    if str(row["device"]) != policy.device or str(row["transport"]) != policy.transport:
        return False
    # EOT also carries regularization: any non-null value is a configuration choice.
    if pd.notna(row["rel_reg"]) and float(row["rel_reg"]) != policy.rel_reg:
        return False
    if method in policy.ot_methods:
        return (
            float(row["effective_k"]) == policy.effective_k
            and str(row["graph_symmetrization"]) == policy.graph_symmetrization
        )
    return True


def canonical_mask(
    frame: pd.DataFrame,
    policy: BenchmarkSelectionPolicy = CURRENT_BENCHMARK,
) -> pd.Series:
    """Row mask of :func:`is_canonical_row`; does not reorder or modify ``frame``."""
    return frame.apply(is_canonical_row, axis=1, policy=policy)
