"""Sparsify raw transport plans by neighbor count, retained mass, or relative maxima.

These operate on the rectangular plan; OT-tSNE selects joint affinities with the
four-block sparsifiers in ``ot_graph``.
"""

import numpy as np

_SUPPORTED_METHODS = {
    "percentage_neighbors",
    "percentage_mass",
    "percentage_below_top",
}


def _validate_plan(P):
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"P must be a 2D array, got shape {P.shape}.")
    if P.shape[0] == 0 or P.shape[1] == 0:
        raise ValueError("P must have non-zero dimensions.")
    if np.any(P < 0):
        raise ValueError("P must be non-negative.")
    return P


def _validate_percentage(percentage):
    p = float(percentage)
    if not (0.0 < p <= 1.0):
        raise ValueError(f"percentage must be in (0, 1], got {percentage}.")
    return p


def _ensure_nonempty(mask, P):
    mask = np.array(mask, dtype=bool, copy=True)
    n0, n1 = mask.shape

    empty_rows = np.where(~mask.any(axis=1))[0]
    if empty_rows.size > 0:
        row_argmax = np.argmax(P[empty_rows], axis=1)
        mask[empty_rows, row_argmax] = True

    empty_cols = np.where(~mask.any(axis=0))[0]
    if empty_cols.size > 0:
        col_argmax = np.argmax(P[:, empty_cols], axis=0)
        mask[col_argmax, empty_cols] = True

    assert mask.shape == (n0, n1)
    return mask


def _finalize_sparse_plan(
    P,
    mask,
    method,
    percentage,
    renormalize=True,
    return_stats=False,
):
    P_sparse = np.where(mask, P, 0.0)
    retained_mass_before_renorm = float(P_sparse.sum())

    if renormalize:
        if retained_mass_before_renorm <= 0:
            raise ValueError("Sparsification produced zero retained mass; cannot renormalize.")
        P_sparse = P_sparse / retained_mass_before_renorm

    stats = {
        "nnz": int(np.count_nonzero(P_sparse)),
        "density": float(np.count_nonzero(P_sparse) / P_sparse.size),
        "retained_mass_before_renorm": retained_mass_before_renorm,
        "method": method,
        "percentage": float(percentage),
    }

    if return_stats:
        return P_sparse, stats
    return P_sparse


def sparsify_plan_percentage_neighbors(
    P,
    percentage,
    ensure_nonempty=True,
    renormalize=True,
    return_stats=False,
):
    """Keep the union of top fractional neighbor counts from each row and column."""
    P = _validate_plan(P)
    p = _validate_percentage(percentage)
    n0, n1 = P.shape

    k_rows = max(1, int(np.ceil(p * n1)))
    k_cols = max(1, int(np.ceil(p * n0)))

    mask = np.zeros_like(P, dtype=bool)

    if k_rows >= n1:
        mask[:, :] = True
    else:
        row_idx = np.argpartition(P, n1 - k_rows, axis=1)[:, -k_rows:]
        rows = np.arange(n0)[:, None]
        mask[rows, row_idx] = True

    if k_cols >= n0:
        mask[:, :] = True
    else:
        col_idx = np.argpartition(P, n0 - k_cols, axis=0)[-k_cols:, :]
        cols = np.arange(n1)[None, :]
        mask[col_idx, cols] = True

    if ensure_nonempty:
        mask = _ensure_nonempty(mask, P)

    return _finalize_sparse_plan(
        P=P,
        mask=mask,
        method="percentage_neighbors",
        percentage=p,
        renormalize=renormalize,
        return_stats=return_stats,
    )


def sparsify_plan_percentage_mass(
    P,
    percentage,
    ensure_nonempty=True,
    renormalize=True,
    return_stats=False,
):
    """Keep each row's smallest descending prefix reaching the requested mass fraction."""
    P = _validate_plan(P)
    p = _validate_percentage(percentage)

    # Row-wise mass retention: in each row, keep the smallest top prefix
    # whose cumulative mass reaches at least p of that row's mass.
    mask = np.zeros_like(P, dtype=bool)
    for i in range(P.shape[0]):
        row = P[i]
        row_sum = float(row.sum())
        if row_sum <= 0:
            continue
        target_mass = p * row_sum
        order = np.argsort(row)[::-1]
        cumsum = np.cumsum(row[order])
        keep_count = int(np.searchsorted(cumsum, target_mass, side="left")) + 1
        mask[i, order[:keep_count]] = True

    if ensure_nonempty:
        mask = _ensure_nonempty(mask, P)

    return _finalize_sparse_plan(
        P=P,
        mask=mask,
        method="percentage_mass",
        percentage=p,
        renormalize=renormalize,
        return_stats=return_stats,
    )


def sparsify_plan_percentage_below_top(
    P,
    percentage,
    ensure_nonempty=True,
    renormalize=True,
    return_stats=False,
    scale_exponent=2.0,
):
    """Keep entries near either their row or column maximum.

    The threshold is ``(1 - percentage**scale_exponent) * maximum``.
    """
    P = _validate_plan(P)
    p = _validate_percentage(percentage)

    # Use a stricter scaling than linear p to avoid fast densification.
    p_eff = p ** float(scale_exponent)

    row_max = P.max(axis=1, keepdims=True)
    row_threshold = (1.0 - p_eff) * row_max
    row_mask = P >= row_threshold

    col_max = P.max(axis=0, keepdims=True)
    col_threshold = (1.0 - p_eff) * col_max
    col_mask = P >= col_threshold

    mask = row_mask | col_mask

    if ensure_nonempty:
        mask = _ensure_nonempty(mask, P)

    return _finalize_sparse_plan(
        P=P,
        mask=mask,
        method="percentage_below_top",
        percentage=p,
        renormalize=renormalize,
        return_stats=return_stats,
    )


def sparsify_transport_plan(
    P,
    method,
    percentage,
    ensure_nonempty=True,
    renormalize=True,
    return_stats=False,
    below_top_scale_exponent=2.0,
):
    """Apply a raw-plan sparsifier, optionally restoring coverage and unit total mass.

    ``return_stats`` also returns density and retained mass before renormalization.
    Coverage restoration selects maxima but cannot make zero-valued entries positive.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"Unknown method '{method}'. Supported: {sorted(_SUPPORTED_METHODS)}.")

    if method == "percentage_neighbors":
        return sparsify_plan_percentage_neighbors(
            P,
            percentage=percentage,
            ensure_nonempty=ensure_nonempty,
            renormalize=renormalize,
            return_stats=return_stats,
        )
    if method == "percentage_mass":
        return sparsify_plan_percentage_mass(
            P,
            percentage=percentage,
            ensure_nonempty=ensure_nonempty,
            renormalize=renormalize,
            return_stats=return_stats,
        )

    return sparsify_plan_percentage_below_top(
        P,
        percentage=percentage,
        ensure_nonempty=ensure_nonempty,
        renormalize=renormalize,
        return_stats=return_stats,
        scale_exponent=below_top_scale_exponent,
    )
