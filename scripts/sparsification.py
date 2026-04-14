

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class SparseResult:
    matrix: np.ndarray
    stats: dict[str, Any]


def _validate_nonnegative_matrix(M: np.ndarray, name: str = "matrix") -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {M.shape}.")
    if M.shape[0] == 0 or M.shape[1] == 0:
        raise ValueError(f"{name} must have non-zero dimensions.")
    if np.any(M < 0):
        raise ValueError(f"{name} must be non-negative.")
    return M


def _validate_percentage(percentage: float) -> float:
    p = float(percentage)
    if not (0.0 < p <= 1.0):
        raise ValueError(f"percentage must be in (0, 1], got {percentage}.")
    return p


def _sanitize_to_zero(M: np.ndarray) -> np.ndarray:
    """Force finite nonnegative output; replace NaN/inf with 0."""
    M = np.asarray(M, dtype=float)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    M[M < 0] = 0.0
    return M


def _ensure_nonempty(mask: np.ndarray, M: np.ndarray) -> np.ndarray:
    mask = np.array(mask, dtype=bool, copy=True)

    empty_rows = np.where(~mask.any(axis=1))[0]
    if empty_rows.size > 0:
        row_argmax = np.argmax(M[empty_rows], axis=1)
        mask[empty_rows, row_argmax] = True

    empty_cols = np.where(~mask.any(axis=0))[0]
    if empty_cols.size > 0:
        col_argmax = np.argmax(M[:, empty_cols], axis=0)
        mask[col_argmax, empty_cols] = True

    return mask


def _sparse_from_mask(M: np.ndarray, mask: np.ndarray, method: str) -> SparseResult:
    out = _sanitize_to_zero(np.where(mask, M, 0.0))
    nnz = int(np.count_nonzero(out))
    stats = {
        "method": method,
        "shape": tuple(M.shape),
        "nnz": nnz,
        "density": float(nnz / M.size),
        "sum": float(out.sum()),
    }
    return SparseResult(matrix=out, stats=stats)


def _validate_four_block_shape(
    M: np.ndarray,
    n0: int,
    n1: int | None,
    name: str = "M",
) -> tuple[int, int]:
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be square for 4-block sparsification, got {M.shape}.")
    n = M.shape[0]
    n0i = int(n0)
    n1i = int(n - n0i) if n1 is None else int(n1)
    if n0i <= 0 or n1i <= 0:
        raise ValueError(f"Invalid block sizes n0={n0i}, n1={n1i}.")
    if n0i + n1i != n:
        raise ValueError(f"Block sizes must satisfy n0+n1={n}, got n0={n0i}, n1={n1i}.")
    return n0i, n1i


def _row_or_col_topk_mask(B: np.ndarray, row_k: int, col_k: int) -> np.ndarray:
    r, c = B.shape
    row_k = min(max(1, int(row_k)), c)
    col_k = min(max(1, int(col_k)), r)
    m = np.zeros_like(B, dtype=bool)

    if row_k >= c:
        m[:, :] = True
    else:
        row_idx = np.argpartition(B, c - row_k, axis=1)[:, -row_k:]
        rows = np.arange(r, dtype=int)[:, None]
        m[rows, row_idx] = True

    if col_k >= r:
        m[:, :] = True
    else:
        col_idx = np.argpartition(B, r - col_k, axis=0)[-col_k:, :]
        cols = np.arange(c, dtype=int)[None, :]
        m[col_idx, cols] = True

    return m


def _top_percentage_neighbors_block_mask(B: np.ndarray, p: float) -> np.ndarray:
    r, c = B.shape
    row_k = max(1, int(np.ceil(p * c)))
    col_k = max(1, int(np.ceil(p * r)))
    return _row_or_col_topk_mask(B, row_k=row_k, col_k=col_k)


def _top_percentage_mass_block_mask(B: np.ndarray, p: float) -> np.ndarray:
    mask = np.zeros_like(B, dtype=bool)

    for i in range(B.shape[0]):
        row = B[i]
        row_sum = float(row.sum())
        if row_sum <= 0:
            continue
        target_mass = p * row_sum
        order = np.argsort(row)[::-1]
        cumsum = np.cumsum(row[order])
        keep_count = int(np.searchsorted(cumsum, target_mass, side="left")) + 1
        mask[i, order[:keep_count]] = True

    for j in range(B.shape[1]):
        col = B[:, j]
        col_sum = float(col.sum())
        if col_sum <= 0:
            continue
        target_mass = p * col_sum
        order = np.argsort(col)[::-1]
        cumsum = np.cumsum(col[order])
        keep_count = int(np.searchsorted(cumsum, target_mass, side="left")) + 1
        mask[order[:keep_count], j] = True

    return mask


def _values_above_percentage_top_block_mask(B: np.ndarray, p: float) -> np.ndarray:
    row_max = B.max(axis=1, keepdims=True)
    col_max = B.max(axis=0, keepdims=True)
    row_mask = B >= (p * row_max)
    col_mask = B >= (p * col_max)
    return row_mask | col_mask


def _build_blockwise_mask(
    M: np.ndarray,
    n0: int,
    n1: int,
    block_mask_fn: Callable[[np.ndarray], np.ndarray],
    ensure_nonempty: bool,
) -> np.ndarray:
    mask = np.zeros_like(M, dtype=bool)
    blocks = [
        (slice(0, n0), slice(0, n0)),
        (slice(0, n0), slice(n0, n0 + n1)),
        (slice(n0, n0 + n1), slice(0, n0)),
        (slice(n0, n0 + n1), slice(n0, n0 + n1)),
    ]
    for rs, cs in blocks:
        B = M[rs, cs]
        bmask = block_mask_fn(B)
        if ensure_nonempty:
            bmask = _ensure_nonempty(bmask, B)
        mask[rs, cs] = bmask

    np.fill_diagonal(mask, False)
    return mask


def sparsify_top_percentage_neighbors(
    M: np.ndarray,
    percentage: float,
    n0: int,
    n1: int | None = None,
    ensure_nonempty: bool = True,
) -> SparseResult:
    """Keep top-p neighbors per row/column within each of 4 blocks."""
    M = _validate_nonnegative_matrix(M, name="M")
    p = _validate_percentage(percentage)
    n0i, n1i = _validate_four_block_shape(M, n0=n0, n1=n1, name="M")
    mask = _build_blockwise_mask(
        M,
        n0=n0i,
        n1=n1i,
        block_mask_fn=lambda B: _top_percentage_neighbors_block_mask(B, p),
        ensure_nonempty=ensure_nonempty,
    )

    result = _sparse_from_mask(M, mask, method="top_percentage_neighbors")
    result.stats.update({"percentage": p, "n0": n0i, "n1": n1i, "blockwise": True})
    return result


def sparsify_top_percentage_mass(
    M: np.ndarray,
    percentage: float,
    n0: int,
    n1: int | None = None,
    ensure_nonempty: bool = True,
) -> SparseResult:
    """Keep minimal top prefixes reaching p mass within each of 4 blocks."""
    M = _validate_nonnegative_matrix(M, name="M")
    p = _validate_percentage(percentage)
    n0i, n1i = _validate_four_block_shape(M, n0=n0, n1=n1, name="M")
    mask = _build_blockwise_mask(
        M,
        n0=n0i,
        n1=n1i,
        block_mask_fn=lambda B: _top_percentage_mass_block_mask(B, p),
        ensure_nonempty=ensure_nonempty,
    )

    result = _sparse_from_mask(M, mask, method="top_percentage_mass")
    result.stats.update({"percentage": p, "n0": n0i, "n1": n1i, "blockwise": True})
    return result


def sparsify_values_above_percentage_top(
    M: np.ndarray,
    percentage: float,
    n0: int,
    n1: int | None = None,
    ensure_nonempty: bool = True,
) -> SparseResult:
    """Keep entries >= p * top value within each of 4 blocks."""
    M = _validate_nonnegative_matrix(M, name="M")
    p = _validate_percentage(percentage)
    n0i, n1i = _validate_four_block_shape(M, n0=n0, n1=n1, name="M")
    mask = _build_blockwise_mask(
        M,
        n0=n0i,
        n1=n1i,
        block_mask_fn=lambda B: _values_above_percentage_top_block_mask(B, p),
        ensure_nonempty=ensure_nonempty,
    )

    result = _sparse_from_mask(M, mask, method="values_above_percentage_top")
    result.stats.update({"percentage": p, "n0": n0i, "n1": n1i, "blockwise": True})
    return result


def sparsify_test_diag_eot_knn(
    A: np.ndarray,
    n0: int,
    n1: int,
    k: int = 15,
    scale_cross_by_ratio: bool = True,
    auto_scale_row_col: bool = True,
    k00_row: int | None = None,
    k00_col: int | None = None,
    k01_row: int | None = None,
    k01_col: int | None = None,
    k10_row: int | None = None,
    k10_col: int | None = None,
    k11_row: int | None = None,
    k11_col: int | None = None,
) -> SparseResult:
    """Pure test_diag_eot blockwise KNN sparsification on 4-block matrix A.

    Selects entries that are top-k by row OR by column in each block.
    When auto_scale_row_col=True, defaults are scaled by block shape:
    row_k = ceil((k / n0) * n_cols), col_k = ceil((k / n0) * n_rows).
    Row/column k can be overridden per block via kXY_row / kXY_col.
    If auto_scale_row_col=False, defaults preserve the previous behavior.
    Returns selected original values (not binarized/symmetrized/normalized).
    """
    A = _validate_nonnegative_matrix(A, name="A")
    n = n0 + n1
    if A.shape != (n, n):
        raise ValueError(f"A must have shape ({n}, {n}), got {A.shape}.")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    k_cross = int(k * n1 / n0) if scale_cross_by_ratio else k
    k_cross = max(1, k_cross)

    S = np.zeros_like(A)

    def _resolve_k(value: int | None, default: int, name: str) -> int:
        out = default if value is None else int(value)
        if out <= 0:
            raise ValueError(f"{name} must be positive, got {out}.")
        return out

    if auto_scale_row_col:
        # Base neighbor fraction induced by k on n0-sized rows.
        p_base = float(k) / float(n0)

        def _scaled_row_k(n_cols: int) -> int:
            return max(1, int(np.ceil(p_base * n_cols)))

        def _scaled_col_k(n_rows: int) -> int:
            return max(1, int(np.ceil(p_base * n_rows)))

        k00r_def, k00c_def = _scaled_row_k(n0), _scaled_col_k(n0)
        k01r_def, k01c_def = _scaled_row_k(n1), _scaled_col_k(n0)
        k10r_def, k10c_def = _scaled_row_k(n0), _scaled_col_k(n1)
        k11r_def, k11c_def = _scaled_row_k(n1), _scaled_col_k(n1)
    else:
        # Legacy defaults:
        # - diagonal-like row/col use k
        # - cross-like row/col use k_cross for A01 and k for A10
        k00r_def, k00c_def = k, k
        k01r_def, k01c_def = k_cross, k_cross
        k10r_def, k10c_def = k, k
        k11r_def, k11c_def = k_cross, k_cross

    k00r = _resolve_k(k00_row, k00r_def, "k00_row")
    k00c = _resolve_k(k00_col, k00c_def, "k00_col")
    k01r = _resolve_k(k01_row, k01r_def, "k01_row")
    k01c = _resolve_k(k01_col, k01c_def, "k01_col")
    k10r = _resolve_k(k10_row, k10r_def, "k10_row")
    k10c = _resolve_k(k10_col, k10c_def, "k10_col")
    k11r = _resolve_k(k11_row, k11r_def, "k11_row")
    k11c = _resolve_k(k11_col, k11c_def, "k11_col")

    m00 = _row_or_col_topk_mask(A[:n0, :n0], row_k=k00r, col_k=k00c)
    m01 = _row_or_col_topk_mask(A[:n0, n0:], row_k=k01r, col_k=k01c)
    m10 = _row_or_col_topk_mask(A[n0:, :n0], row_k=k10r, col_k=k10c)
    m11 = _row_or_col_topk_mask(A[n0:, n0:], row_k=k11r, col_k=k11c)

    S[:n0, :n0] = np.where(m00, A[:n0, :n0], 0.0)
    S[:n0, n0:] = np.where(m01, A[:n0, n0:], 0.0)
    S[n0:, :n0] = np.where(m10, A[n0:, :n0], 0.0)
    S[n0:, n0:] = np.where(m11, A[n0:, n0:], 0.0)
    np.fill_diagonal(S, 0.0)
    S = _sanitize_to_zero(S)

    result = _sparse_from_mask(A, S > 0, method="test_diag_eot_knn")
    result.matrix = S
    result.stats.update(
        {
            "n0": int(n0),
            "n1": int(n1),
            "k": int(k),
            "k_cross": int(k_cross),
            "scale_cross_by_ratio": bool(scale_cross_by_ratio),
            "auto_scale_row_col": bool(auto_scale_row_col),
            "k00_row": int(k00r),
            "k00_col": int(k00c),
            "k01_row": int(k01r),
            "k01_col": int(k01c),
            "k10_row": int(k10r),
            "k10_col": int(k10c),
            "k11_row": int(k11r),
            "k11_col": int(k11c),
        }
    )
    return result


def quantile_mapping_diag_blocks(
    A: np.ndarray,
    n0: int,
    q_low: float = 0.05,
    q_high: float = 0.95,
    include_offdiag: bool = True,
) -> SparseResult:
    """Simple row/column quantile mapping on 4-block matrix.

    For each selected block B and each positive entry x:
    - compute row-wise map with row quantiles q_low/q_high
    - compute col-wise map with col quantiles q_low/q_high
    - output is average(row_map, col_map)

    Mapping uses fixed acceleration f(u)=u^2:
      clip((f(x) - f(q_low_val)) / (f(q_high_val) - f(q_low_val)), 0, 1)
    """
    A = _validate_nonnegative_matrix(A, name="A")
    n0i, n1i = _validate_four_block_shape(A, n0=n0, n1=None, name="A")

    q_low = float(q_low)
    if not (0.0 <= q_low < 1.0):
        raise ValueError(f"q_low must be in [0, 1), got {q_low}.")
    q_high = float(q_high)
    if not (0.0 < q_high <= 1.0):
        raise ValueError(f"q_high must be in (0, 1], got {q_high}.")
    if q_high <= q_low:
        raise ValueError(f"q_high must be > q_low, got q_low={q_low}, q_high={q_high}.")

    A_out = np.array(A, copy=True)

    def _finalize_square(B: np.ndarray) -> np.ndarray:
        Bout = 0.5 * (B + B.T)
        np.fill_diagonal(Bout, 0.0)
        return Bout

    def _accelerate(u: np.ndarray) -> np.ndarray:
        # Fixed acceleration: no user tuning parameter.
        return np.square(np.maximum(u, 0.0))

    def _map_vals(vals: np.ndarray, lo: float, hi: float) -> np.ndarray:
        flo = float(_accelerate(np.array([lo]))[0])
        fhi = float(_accelerate(np.array([hi]))[0])
        den = max(fhi - flo, 1e-12)
        fv = _accelerate(vals)
        return np.clip((fv - flo) / den, 0.0, 1.0)

    def _remap_block(B: np.ndarray, is_square_block: bool) -> np.ndarray:
        B = np.array(B, copy=True)
        mask = B > 0
        if is_square_block:
            offdiag_mask = np.ones_like(B, dtype=bool)
            np.fill_diagonal(offdiag_mask, False)
            mask &= offdiag_mask

        vals = B[mask]
        if vals.size < 1:
            if is_square_block:
                return _finalize_square(B)
            return B

        n_rows, n_cols = B.shape
        row_map = np.zeros_like(B)
        col_map = np.zeros_like(B)
        row_has = np.zeros_like(B, dtype=bool)
        col_has = np.zeros_like(B, dtype=bool)

        # Row-wise mapping
        for i in range(n_rows):
            idx = mask[i]
            row_vals = B[i, idx]
            if row_vals.size == 0:
                continue
            if row_vals.size == 1:
                row_map[i, idx] = 1.0
                row_has[i, idx] = True
                continue
            lo = float(np.quantile(row_vals, q_low))
            hi = float(np.quantile(row_vals, q_high))
            if hi <= lo:
                row_map[i, idx] = 1.0
            else:
                row_map[i, idx] = _map_vals(row_vals, lo, hi)
            row_has[i, idx] = True

        # Column-wise mapping
        for j in range(n_cols):
            idx = mask[:, j]
            col_vals = B[idx, j]
            if col_vals.size == 0:
                continue
            if col_vals.size == 1:
                col_map[idx, j] = 1.0
                col_has[idx, j] = True
                continue
            lo = float(np.quantile(col_vals, q_low))
            hi = float(np.quantile(col_vals, q_high))
            if hi <= lo:
                col_map[idx, j] = 1.0
            else:
                col_map[idx, j] = _map_vals(col_vals, lo, hi)
            col_has[idx, j] = True

        # Average row and column mapped values where available.
        Bq = np.zeros_like(B)
        w = row_has.astype(float) + col_has.astype(float)
        use = w > 0
        Bq[use] = (row_map[use] + col_map[use]) / w[use]
        Bq[~mask] = 0.0
        if is_square_block:
            Bq = _finalize_square(Bq)
        return Bq

    A_out[:n0i, :n0i] = _remap_block(A_out[:n0i, :n0i], is_square_block=True)
    A_out[n0i:, n0i:] = _remap_block(A_out[n0i:, n0i:], is_square_block=True)
    if include_offdiag:
        A_out[:n0i, n0i:] = _remap_block(A_out[:n0i, n0i:], is_square_block=False)
        A_out[n0i:, :n0i] = _remap_block(A_out[n0i:, :n0i], is_square_block=False)

    np.fill_diagonal(A_out, 0.0)
    A_out = _sanitize_to_zero(A_out)

    nnz = int(np.count_nonzero(A_out))
    stats = {
        "method": "quantile_mapping_diag_blocks",
        "shape": tuple(A_out.shape),
        "nnz": nnz,
        "density": float(nnz / A_out.size),
        "sum": float(A_out.sum()),
        "n0": int(n0i),
        "n1": int(n1i),
        "q_low": float(q_low),
        "q_high": float(q_high),
        "mapping_mode": "row_col_quantile_avg_square",
        "include_offdiag": bool(include_offdiag),
    }
    return SparseResult(matrix=A_out, stats=stats)


# Post-processing helpers (apply after pure sparsification)
def binarize_selected(M: np.ndarray) -> np.ndarray:
    M = _sanitize_to_zero(M)
    return (M > 0).astype(float)


def symmetrize_max(M: np.ndarray) -> np.ndarray:
    M = _sanitize_to_zero(M)
    return _sanitize_to_zero(np.maximum(M, M.T))


def normalize_sum(M: np.ndarray) -> np.ndarray:
    M = _sanitize_to_zero(M)
    s = float(M.sum())
    if s > 0:
        return _sanitize_to_zero(M / s)
    return _sanitize_to_zero(M)


__all__ = [
    "SparseResult",
    "sparsify_top_percentage_neighbors",
    "sparsify_top_percentage_mass",
    "sparsify_values_above_percentage_top",
    "sparsify_test_diag_eot_knn",
    "quantile_mapping_diag_blocks",
    "binarize_selected",
    "symmetrize_max",
    "normalize_sum",
]
