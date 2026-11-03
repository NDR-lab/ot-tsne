"""OT graph construction, sparsification, initialization, and embedding."""

import time
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import umap
from openTSNE import TSNE, affinity, initialization
from openTSNE.affinity import Affinities
from openTSNE.initialization import jitter

from ne_aligned_utils.config import (
    DEFAULT_BACKEND,
    DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
    DEFAULT_TSNE_N_ITER,
)
from ne_aligned_utils.methods.common import (
    stage as _stage,
    validate_batch_sizes as _validate_batch_sizes,
    validate_matrix as _validate_matrix,
)
from ne_aligned_utils.methods.ot_transport import compute_eot_embedding_from_plan


#: OT-tSNE initialization schemes.
OT_TSNE_INIT_CHOICES = ("random", "spectral", "eot")


@dataclass(frozen=True)
class EffectiveKSplit:
    """Record the total neighbor budget and its per-batch quotas."""
    effective_k: int
    batch0_k: int
    batch1_k: int


def resolve_effective_k_split(effective_k: int, n0: int, n1: int) -> EffectiveKSplit:
    """Split an exact neighbor budget proportionally across two batches."""
    n0, n1 = _validate_batch_sizes(n0, n1)
    if not float(effective_k).is_integer():
        raise ValueError(f"effective_k must be an integer, got {effective_k}.")
    effective_k = int(effective_k)
    if effective_k <= 0:
        raise ValueError(f"effective_k must be positive, got {effective_k}.")

    capacities = np.array([max(n0 - 1, 0), max(n1 - 1, 0)], dtype=int)
    max_effective_k = int(capacities.sum())
    if effective_k > max_effective_k:
        raise ValueError(
            "effective_k exceeds the available non-self neighbors under a shared "
            f"batch split: got {effective_k}, max is {max_effective_k} for "
            f"n0={n0}, n1={n1}."
        )

    sizes = np.array([n0, n1], dtype=float)
    quotas = effective_k * sizes / float(n0 + n1)
    counts = np.floor(quotas).astype(int)
    counts = np.minimum(counts, capacities)
    remainders = quotas - np.floor(quotas)

    remaining = effective_k - int(counts.sum())
    while remaining > 0:
        eligible = np.flatnonzero(counts < capacities)
        if eligible.size == 0:
            raise ValueError(
                "Unable to allocate effective_k exactly within the available "
                f"batch capacities for n0={n0}, n1={n1}."
            )

        best_remainder = np.max(remainders[eligible])
        tied = eligible[np.isclose(remainders[eligible], best_remainder)]
        if tied.size > 1:
            tied_sizes = np.array([n0, n1], dtype=int)[tied]
            smallest_size = np.min(tied_sizes)
            tied = tied[tied_sizes == smallest_size]
        chosen = int(np.min(tied))
        counts[chosen] += 1
        remaining -= 1

    return EffectiveKSplit(
        effective_k=effective_k,
        batch0_k=int(counts[0]),
        batch1_k=int(counts[1]),
    )


def build_ot_tsne_four_block_matrix(
    plan: np.ndarray,
    return_singular_values: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Build the joint affinity matrix from the transport plan and its singular spectrum."""
    matrix, singular_values, _ = _build_four_block_parts(plan)
    if return_singular_values:
        return matrix, singular_values
    return matrix


def build_ot_tsne_zerodiag_four_block_matrix(
    plan: np.ndarray,
    return_singular_values: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Build zero-diagonal affinities with SVD-reconstructed cross blocks.

    Cross blocks equal the plan in exact arithmetic, with possible rounding differences.
    Use a cross-batch sparsifier: generic per-block top-k selection turns tied zeros
    into within-batch edges when binarizing.
    """
    matrix, singular_values, _ = _build_four_block_parts(plan)
    n0, n1 = np.asarray(plan, dtype=float).shape
    matrix[:n0, :n0] = 0.0
    matrix[n0:, n0:] = 0.0
    if return_singular_values:
        return matrix, singular_values
    return matrix


def _wants_eot_basis(init: str | np.ndarray) -> bool:
    """Whether the caller's ``init`` needs the stacked singular basis retained."""
    return isinstance(init, str) and init == "eot"


def _build_four_block_parts(
    plan: np.ndarray,
    return_eot: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return the joint affinity matrix, singular values, and optional stacked basis.

    Retain the basis only for ``return_eot=True``; its columns ``1:3`` provide the
    EOT initialization without another SVD.
    """
    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2:
        raise ValueError(f"plan must be a 2D array, got shape {plan.shape!r}.")

    n0, n1 = plan.shape
    u, singular_values, vt = np.linalg.svd(plan, full_matrices=False)
    eot = np.concatenate([u * np.sqrt(n0), vt.T * np.sqrt(n1)], axis=0)
    # Scale basis columns directly rather than materializing a diagonal matrix.
    matrix = (eot * singular_values) @ eot.T / np.sqrt(n0 * n1)
    matrix = np.asarray(matrix, dtype=float)
    # The preceding operation owns this buffer, so cleanup can run in place.
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    matrix[matrix < 0.0] = 0.0
    np.fill_diagonal(matrix, 0.0)
    return matrix, singular_values, (eot if return_eot else None)


def _topk_scores(block: np.ndarray, exclude_diagonal: bool) -> np.ndarray:
    """Prepare shared ranking scores, excluding self-entries when requested."""
    scores = np.nan_to_num(block, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    if exclude_diagonal:
        scores = np.array(scores, copy=True)
        np.fill_diagonal(scores, -np.inf)
    return scores


def _row_topk_indices(scores: np.ndarray, row_k: int) -> np.ndarray:
    """Return an ``(n_rows, row_k)`` array of selected column indices.

    ``argpartition`` selects exactly the quota, but tied indices are implementation-defined.
    Keep this selection shared by graph masks and directed neighbor lists.
    """
    n_rows, n_cols = scores.shape
    row_k = int(row_k)
    if row_k == 0:
        return np.empty((n_rows, 0), dtype=int)
    if row_k == n_cols:
        # argpartition has no valid pivot here; every column is selected.
        return np.tile(np.arange(n_cols, dtype=int), (n_rows, 1))
    return np.argpartition(scores, n_cols - row_k, axis=1)[:, -row_k:]


def _row_or_col_topk_mask(
    block: np.ndarray,
    row_k: int,
    col_k: int,
    exclude_diagonal: bool = False,
    return_row_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Union of each row's top ``row_k`` and each column's top ``col_k``, as a block mask.

    ``return_row_indices`` also returns the row pass's ``(n_rows, row_k)`` column indices -- the
    directed selection UMAP needs -- so a caller does not have to repeat the row pass to get them.
    """
    block = np.asarray(block, dtype=float)
    n_rows, n_cols = block.shape
    row_k = int(row_k)
    col_k = int(col_k)
    if row_k < 0 or col_k < 0:
        raise ValueError(f"row_k and col_k must be nonnegative, got {row_k}, {col_k}.")

    row_capacity = n_cols - 1 if exclude_diagonal else n_cols
    col_capacity = n_rows - 1 if exclude_diagonal else n_rows
    if row_k > row_capacity:
        raise ValueError(f"row_k={row_k} exceeds block row capacity {row_capacity}.")
    if col_k > col_capacity:
        raise ValueError(f"col_k={col_k} exceeds block column capacity {col_capacity}.")

    mask = np.zeros_like(block, dtype=bool)
    scores = _topk_scores(block, exclude_diagonal)

    # Same helper the directed selection uses, so the two cannot break ties differently.
    row_idx = _row_topk_indices(scores, row_k)
    if row_k > 0:
        rows = np.arange(n_rows, dtype=int)[:, None]
        mask[rows, row_idx] = True

    if col_k > 0:
        if col_k == n_rows:
            mask[:, :] = True
        else:
            col_idx = np.argpartition(scores, n_rows - col_k, axis=0)[-col_k:, :]
            cols = np.arange(n_cols, dtype=int)[None, :]
            mask[col_idx, cols] = True

    if exclude_diagonal:
        np.fill_diagonal(mask, False)
    return (mask, row_idx) if return_row_indices else mask


#: Directed top-k symmetrization: union (or) or reciprocal intersection (and).
GRAPH_SYMMETRIZATIONS = ("or", "and")
DEFAULT_GRAPH_SYMMETRIZATION = "or"


def _validate_symmetrization(value: str, name: str = "graph_symmetrization") -> str:
    if value not in GRAPH_SYMMETRIZATIONS:
        raise ValueError(f"{name} must be one of {GRAPH_SYMMETRIZATIONS}, got {value!r}.")
    return value


def _validate_symmetrizations_for_backend(backend: str, **modes: str) -> None:
    """Validate modes and reject mutual selection for UMAP.

    UMAP reconstructs a fuzzy union from directed neighbors, so that interface cannot
    preserve a requested mutual intersection.
    """
    mutual = sorted(
        name
        for name, value in modes.items()
        if _validate_symmetrization(value, name) != DEFAULT_GRAPH_SYMMETRIZATION
    )
    if mutual and backend != DEFAULT_BACKEND:
        raise ValueError(
            f"{', '.join(mutual)}='and' is only supported by the t-SNE backend; got "
            f"backend={backend!r}, which embeds a directed neighbour list whose fuzzy union "
            "reconstructs the 'or' graph regardless."
        )


def _mutual_topk_mask(
    block: np.ndarray,
    row_k: int,
    col_k: int,
    exclude_diagonal: bool = False,
    return_row_indices: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Intersect row and column top-k selections to retain reciprocal choices.

    Both passes use ``_row_topk_indices``, with contiguous transposed scores for the
    column pass. Consistent tie-breaking preserves symmetry across opposite blocks;
    an asymmetric mask would be widened by the final graph union.
    """
    block = np.asarray(block, dtype=float)
    n_rows, n_cols = block.shape
    row_k = int(row_k)
    col_k = int(col_k)
    if row_k < 0 or col_k < 0:
        raise ValueError(f"row_k and col_k must be nonnegative, got {row_k}, {col_k}.")

    row_capacity = n_cols - 1 if exclude_diagonal else n_cols
    col_capacity = n_rows - 1 if exclude_diagonal else n_rows
    if row_k > row_capacity:
        raise ValueError(f"row_k={row_k} exceeds block row capacity {row_capacity}.")
    if col_k > col_capacity:
        raise ValueError(f"col_k={col_k} exceeds block column capacity {col_capacity}.")

    scores = _topk_scores(block, exclude_diagonal)

    row_mask = np.zeros_like(block, dtype=bool)
    row_idx = _row_topk_indices(scores, row_k)
    if row_k > 0:
        row_mask[np.arange(n_rows, dtype=int)[:, None], row_idx] = True

    col_mask = np.zeros_like(block, dtype=bool)
    if col_k > 0:
        # Use the shared row selector on contiguous transposed scores to align tie-breaking.
        col_idx = _row_topk_indices(np.ascontiguousarray(scores.T), col_k)
        cols = np.arange(n_cols, dtype=int)[:, None]
        col_mask[col_idx, cols] = True

    mask = row_mask & col_mask
    if exclude_diagonal:
        np.fill_diagonal(mask, False)
    return (mask, row_idx) if return_row_indices else mask


#: Floating-point matrix products can leave opposite entries slightly unequal.
#: Average before ranking: near-tied scores can otherwise produce inconsistent selections.
def _symmetrised(matrix: np.ndarray) -> np.ndarray:
    """Return ``(matrix + matrix.T) / 2`` using one temporary.

    Multiplication by 0.5 preserves the division's binary floating-point result.
    """
    matrix = np.asarray(matrix, dtype=float)
    result = np.empty_like(matrix)
    np.add(matrix, matrix.T, out=result)
    result *= 0.5
    return result


def four_block_directed_knn(
    matrix: np.ndarray,
    n0: int,
    n1: int,
    effective_k: int,
    within_batch_indices: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, EffectiveKSplit]:
    """Return global directed neighbor indices and the proportional budget split.

    The array has shape ``(n0 + n1, effective_k)``: batch-0 neighbor columns precede
    batch-1 columns, with self-indices excluded. UMAP requires this fixed-width list
    rather than the symmetrized graph, whose degrees may exceed the budget.
    Row and column partitioning can break exact ties differently; graph equivalence
    must be checked on the supplied scores.
    """
    n0, n1 = _validate_batch_sizes(n0, n1)
    matrix = _symmetrised(matrix)
    expected = (n0 + n1, n0 + n1)
    if matrix.shape != expected:
        raise ValueError(
            f"Expected a square 4-block matrix with shape {expected}, got {matrix.shape}."
        )

    split = resolve_effective_k_split(effective_k=effective_k, n0=n0, n1=n1)
    batch_ks = (split.batch0_k, split.batch1_k)
    slices = (slice(0, n0), slice(n0, n0 + n1))
    offsets = (0, n0)

    columns: list[np.ndarray] = []
    for col_batch, col_slice in enumerate(slices):
        k = batch_ks[col_batch]
        if k == 0:
            continue
        per_row = []
        for row_batch, row_slice in enumerate(slices):
            block = matrix[row_slice, col_slice]
            # The same score preparation and the same tie-breaking the mask uses.
            scores = _topk_scores(block, exclude_diagonal=row_batch == col_batch)
            per_row.append(_row_topk_indices(scores, k) + offsets[col_batch])
        columns.append(np.concatenate(per_row, axis=0))

    if not columns:
        raise ValueError("effective_k resolved to zero neighbours in both batches.")
    knn_indices = np.concatenate(columns, axis=1).astype(np.int32, copy=False)

    if within_batch_indices is not None:
        # Replace only within-batch column groups with supplied kNN indices; cross choices stay
        # intact.
        idx0, idx1 = (np.asarray(a, dtype=np.int32) for a in within_batch_indices)
        if idx0.shape != (n0, split.batch0_k):
            raise ValueError(
                f"within-batch indices for batch 0 have shape {idx0.shape}, "
                f"expected {(n0, split.batch0_k)}"
            )
        if idx1.shape != (n1, split.batch1_k):
            raise ValueError(
                f"within-batch indices for batch 1 have shape {idx1.shape}, "
                f"expected {(n1, split.batch1_k)}"
            )
        # idx0 is already global: batch 0 occupies [0, n0). idx1 is batch-local and needs the
        # offset.
        knn_indices[:n0, : split.batch0_k] = idx0
        knn_indices[n0:, split.batch0_k :] = idx1 + n0
    if knn_indices.shape != (n0 + n1, split.effective_k):
        raise AssertionError(
            f"directed kNN has shape {knn_indices.shape}, expected {(n0 + n1, split.effective_k)}"
        )
    return knn_indices, split


def cross_batch_directed_knn(
    matrix: np.ndarray,
    n0: int,
    n1: int,
    effective_k: int,
) -> np.ndarray:
    """Return each point's top opposite-batch neighbors as global indices.

    The result has shape ``(n0 + n1, effective_k)`` and no self-indices. A full
    cross-batch budget gives UMAP a fixed width even when batch sizes differ.
    """
    n0, n1 = _validate_batch_sizes(n0, n1)
    matrix = _symmetrised(matrix)
    expected = (n0 + n1, n0 + n1)
    if matrix.shape != expected:
        raise ValueError(
            f"Expected a square 4-block matrix with shape {expected}, got {matrix.shape}."
        )
    effective_k = resolve_cross_batch_k(effective_k=effective_k, n0=n0, n1=n1)

    # Cross-batch scores contain no self-comparisons.
    top0 = (
        _row_topk_indices(_topk_scores(matrix[:n0, n0:], exclude_diagonal=False), effective_k) + n0
    )
    top1 = _row_topk_indices(_topk_scores(matrix[n0:, :n0], exclude_diagonal=False), effective_k)

    knn_indices = np.concatenate([top0, top1], axis=0).astype(np.int32, copy=False)
    if knn_indices.shape != (n0 + n1, effective_k):
        raise AssertionError(
            f"cross-batch directed kNN has shape {knn_indices.shape}, "
            f"expected {(n0 + n1, effective_k)}"
        )
    return knn_indices


def sparsify_four_block_top_k(
    matrix: np.ndarray,
    n0: int,
    n1: int,
    effective_k: int,
    binarize: bool = False,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
) -> tuple[np.ndarray, EffectiveKSplit]:
    """Select per-block top-k entries and combine directed choices.

    ``"or"`` retains choices from either endpoint; ``"and"`` requires reciprocity.
    Mutual selection can leave isolated points, so the budget is not a lower bound
    on realized degree.
    """
    n0, n1 = _validate_batch_sizes(n0, n1)
    graph_symmetrization = _validate_symmetrization(graph_symmetrization)
    matrix = _symmetrised(matrix)
    expected_shape = (n0 + n1, n0 + n1)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Expected a square 4-block matrix with shape {expected_shape}, got {matrix.shape}."
        )

    split = resolve_effective_k_split(effective_k=effective_k, n0=n0, n1=n1)
    batch_ks = (split.batch0_k, split.batch1_k)
    slices = (slice(0, n0), slice(n0, n0 + n1))
    block_mask_fn = (
        _row_or_col_topk_mask if graph_symmetrization == "or" else _mutual_topk_mask
    )

    mask = np.zeros_like(matrix, dtype=bool)
    for row_batch, row_slice in enumerate(slices):
        for col_batch, col_slice in enumerate(slices):
            block = matrix[row_slice, col_slice]
            block_mask = block_mask_fn(
                block,
                row_k=batch_ks[col_batch],
                col_k=batch_ks[row_batch],
                exclude_diagonal=row_batch == col_batch,
            )
            mask[row_slice, col_slice] = block_mask

    np.fill_diagonal(mask, False)
    sparse_matrix = np.where(mask, matrix, 0.0)
    # The selected array owns its buffer, so cleanup can run in place.
    sparse_matrix = np.nan_to_num(sparse_matrix, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    sparse_matrix[sparse_matrix < 0.0] = 0.0
    if binarize:
        sparse_matrix = mask.astype(float)
        np.fill_diagonal(sparse_matrix, 0.0)
    return sparse_matrix, split


def resolve_cross_batch_k(effective_k: int, n0: int, n1: int) -> int:
    """Validate a full opposite-batch budget, bounded by ``min(n0, n1)``.

    Cross-batch blocks have no self-entries to exclude.
    """
    n0, n1 = _validate_batch_sizes(n0, n1)
    if not float(effective_k).is_integer():
        raise ValueError(f"effective_k must be an integer, got {effective_k}.")
    effective_k = int(effective_k)
    if effective_k <= 0:
        raise ValueError(f"effective_k must be positive, got {effective_k}.")
    capacity = min(n0, n1)
    if effective_k > capacity:
        raise ValueError(
            "effective_k exceeds the cross-batch neighbours available to every point: got "
            f"{effective_k}, max is {capacity} for n0={n0}, n1={n1}. A cross-batch-only "
            "selection draws a point's whole budget from the opposite batch, so the ceiling is "
            "the smaller batch rather than the two capacities summed."
        )
    return effective_k


def sparsify_four_block_cross_batch_only(
    matrix: np.ndarray,
    n0: int,
    n1: int,
    effective_k: int,
    binarize: bool = False,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
) -> np.ndarray:
    """Select only cross-batch edges, using the full budget for each point.

    Within-batch blocks are excluded from selection: binarized top-k on zero blocks
    would create edges from ties. ``"or"`` combines either endpoint's choices;
    ``"and"`` requires reciprocity and can reduce realized degree below the budget.
    """
    n0, n1 = _validate_batch_sizes(n0, n1)
    graph_symmetrization = _validate_symmetrization(graph_symmetrization)
    matrix = _symmetrised(matrix)
    expected_shape = (n0 + n1, n0 + n1)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Expected a square 4-block matrix with shape {expected_shape}, got {matrix.shape}."
        )
    effective_k = resolve_cross_batch_k(effective_k=effective_k, n0=n0, n1=n1)

    mask = np.zeros_like(matrix, dtype=bool)
    # Cross blocks contain no self-comparisons and each uses the full neighbor budget.
    block_mask_fn = (
        _row_or_col_topk_mask if graph_symmetrization == "or" else _mutual_topk_mask
    )
    mask[:n0, n0:] = block_mask_fn(
        matrix[:n0, n0:], row_k=effective_k, col_k=effective_k, exclude_diagonal=False
    )
    mask[n0:, :n0] = block_mask_fn(
        matrix[n0:, :n0], row_k=effective_k, col_k=effective_k, exclude_diagonal=False
    )

    np.fill_diagonal(mask, False)
    sparse_matrix = np.where(mask, matrix, 0.0)
    # The preceding operation owns this buffer, so cleanup can run in place.
    sparse_matrix = np.nan_to_num(sparse_matrix, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    sparse_matrix[sparse_matrix < 0.0] = 0.0
    if binarize:
        sparse_matrix = mask.astype(float)
        np.fill_diagonal(sparse_matrix, 0.0)
    return sparse_matrix


def sparsify_four_block_cross_batch_split(
    matrix: np.ndarray,
    n0: int,
    n1: int,
    effective_k: int,
    binarize: bool = False,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
) -> tuple[np.ndarray, EffectiveKSplit]:
    """Keep only cross-batch edges under the base method's proportional budget split.

    Unlike the full-budget zero-diagonal arm, this preserves the base method's cross-batch edge
    set exactly. The returned split records the unequal per-batch degrees.
    """
    n0, n1 = _validate_batch_sizes(n0, n1)
    graph_symmetrization = _validate_symmetrization(graph_symmetrization)
    matrix = _symmetrised(matrix)
    expected_shape = (n0 + n1, n0 + n1)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Expected a square 4-block matrix with shape {expected_shape}, got {matrix.shape}."
        )

    split = resolve_effective_k_split(effective_k=effective_k, n0=n0, n1=n1)
    batch_ks = (split.batch0_k, split.batch1_k)
    block_mask_fn = (
        _row_or_col_topk_mask if graph_symmetrization == "or" else _mutual_topk_mask
    )

    mask = np.zeros_like(matrix, dtype=bool)
    # Match the base sparsifier's cross-block quotas without selecting tied zeros in the diagonal.
    mask[:n0, n0:] = block_mask_fn(
        matrix[:n0, n0:], row_k=batch_ks[1], col_k=batch_ks[0], exclude_diagonal=False
    )
    mask[n0:, :n0] = block_mask_fn(
        matrix[n0:, :n0], row_k=batch_ks[0], col_k=batch_ks[1], exclude_diagonal=False
    )

    np.fill_diagonal(mask, False)
    sparse_matrix = np.where(mask, matrix, 0.0)
    sparse_matrix = np.nan_to_num(sparse_matrix, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    sparse_matrix[sparse_matrix < 0.0] = 0.0
    if binarize:
        sparse_matrix = mask.astype(float)
        np.fill_diagonal(sparse_matrix, 0.0)
    return sparse_matrix, split


def _variable_topk_mask_along(
    scores: np.ndarray, budgets: np.ndarray, k_max: int, axis: int
) -> np.ndarray:
    """Select a variable number of top entries along each row or column.

    Partition with the fixed-budget call before ranking and truncating candidates.
    When every budget equals ``k_max``, this preserves the fixed mask's tie-breaking.
    """
    n_rows, n_cols = scores.shape
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    k_max = int(k_max)
    if k_max <= 0:
        return mask

    if axis == 1:
        length = n_cols
        if k_max >= length:
            candidates = np.tile(np.arange(length, dtype=int), (n_rows, 1))
        else:
            candidates = np.argpartition(scores, length - k_max, axis=1)[:, -k_max:]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    else:
        length = n_rows
        if k_max >= length:
            candidates = np.tile(np.arange(length, dtype=int)[:, None], (1, n_cols))
        else:
            candidates = np.argpartition(scores, length - k_max, axis=0)[-k_max:, :]
        candidate_scores = np.take_along_axis(scores, candidates, axis=0)

    # Best first, so truncating to a budget keeps the strongest edges.
    order = np.argsort(-candidate_scores, axis=axis, kind="stable")
    ranked = np.take_along_axis(candidates, order, axis=axis)

    budgets = np.asarray(budgets, dtype=int)
    if axis == 1:
        keep = np.arange(k_max)[None, :] < budgets[:, None]
        rows = np.broadcast_to(np.arange(n_rows, dtype=int)[:, None], ranked.shape)
        mask[rows[keep], ranked[keep]] = True
    else:
        keep = np.arange(k_max)[:, None] < budgets[None, :]
        cols = np.broadcast_to(np.arange(n_cols, dtype=int)[None, :], ranked.shape)
        mask[ranked[keep], cols[keep]] = True
    return mask


def resolve_mass_budgets(mass: np.ndarray, k: int, k_min: int = 1) -> np.ndarray:
    """Scale neighbor budgets by each cell's relative share of transported mass.

    Normalize the marginal by its own total: global mass loss must not shrink all
    budgets. A uniform share retains ``k``; ``k_min`` limits loss of connectivity.
    """
    mass = np.asarray(mass, dtype=float)
    if mass.ndim != 1:
        raise ValueError(f"mass must be a 1D marginal, got shape {mass.shape}.")
    k = int(k)
    k_min = int(k_min)
    if k_min < 0 or k_min > k:
        raise ValueError(f"k_min must lie in [0, k] = [0, {k}], got {k_min}.")
    if k == 0:
        return np.zeros(mass.shape[0], dtype=int)

    total = float(mass.sum())
    if not (total > 0.0):
        raise ValueError("Cannot derive mass budgets from a marginal with zero total mass.")
    ratio = (mass / total) * mass.shape[0]
    budgets = np.rint(k * ratio).astype(int)
    return np.clip(budgets, k_min, k)


def sparsify_four_block_top_k_mass_aware(
    matrix: np.ndarray,
    n0: int,
    n1: int,
    effective_k: int,
    row_budgets: np.ndarray,
    col_budgets: np.ndarray,
    binarize: bool = False,
) -> tuple[np.ndarray, EffectiveKSplit]:
    """Use per-cell cross-batch budgets while keeping fixed within-batch quotas.

    ``row_budgets`` and ``col_budgets`` govern both incoming and outgoing choices
    for each cell, preserving symmetry. Budgets carry transported-mass information
    that rank selection and binarization would otherwise discard. Full quotas
    reproduce fixed-budget selection, including tie-breaking.
    """
    n0, n1 = _validate_batch_sizes(n0, n1)
    matrix = _symmetrised(matrix)
    expected_shape = (n0 + n1, n0 + n1)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Expected a square 4-block matrix with shape {expected_shape}, got {matrix.shape}."
        )

    split = resolve_effective_k_split(effective_k=effective_k, n0=n0, n1=n1)
    batch_ks = (split.batch0_k, split.batch1_k)
    slices = (slice(0, n0), slice(n0, n0 + n1))

    budgets = (
        np.asarray(row_budgets, dtype=int),
        np.asarray(col_budgets, dtype=int),
    )
    for batch, (expected_n, budget) in enumerate(zip((n0, n1), budgets)):
        if budget.shape != (expected_n,):
            raise ValueError(
                f"budgets for batch {batch} have shape {budget.shape}, expected {(expected_n,)}"
            )
        if budget.min() < 0 or budget.max() > batch_ks[1 - batch]:
            raise ValueError(
                f"budgets for batch {batch} must lie in [0, {batch_ks[1 - batch]}], "
                f"got [{budget.min()}, {budget.max()}]"
            )

    mask = np.zeros_like(matrix, dtype=bool)
    for row_batch, row_slice in enumerate(slices):
        for col_batch, col_slice in enumerate(slices):
            block = matrix[row_slice, col_slice]
            if row_batch == col_batch:
                block_mask = _row_or_col_topk_mask(
                    block,
                    row_k=batch_ks[col_batch],
                    col_k=batch_ks[row_batch],
                    exclude_diagonal=True,
                )
            else:
                # A cross block. Its rows are one batch and its columns the other, so the row pass
                # spends the row batch's budgets and the column pass the column batch's.
                scores = _topk_scores(block, exclude_diagonal=False)
                block_mask = _variable_topk_mask_along(
                    scores, budgets[row_batch], k_max=batch_ks[col_batch], axis=1
                )
                block_mask |= _variable_topk_mask_along(
                    scores, budgets[col_batch], k_max=batch_ks[row_batch], axis=0
                )
            mask[row_slice, col_slice] = block_mask

    np.fill_diagonal(mask, False)
    sparse_matrix = np.where(mask, matrix, 0.0)
    sparse_matrix = np.nan_to_num(sparse_matrix, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    sparse_matrix[sparse_matrix < 0.0] = 0.0
    if binarize:
        sparse_matrix = mask.astype(float)
        np.fill_diagonal(sparse_matrix, 0.0)
    return sparse_matrix, split


def sparsify_four_block_top_percentage(
    matrix: np.ndarray,
    n0: int,
    n1: int,
    percentage: float,
    ensure_nonempty: bool = True,
    binarize: bool = False,
) -> np.ndarray:
    """Keep per-block row and column top fractions, excluding self-entries."""
    n0, n1 = _validate_batch_sizes(n0, n1)
    matrix = np.asarray(matrix, dtype=float)
    expected_shape = (n0 + n1, n0 + n1)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Expected a square 4-block matrix with shape {expected_shape}, got {matrix.shape}."
        )

    percentage = float(percentage)
    if not (0.0 < percentage <= 1.0):
        raise ValueError(f"percentage must be in (0, 1], got {percentage}.")

    mask = np.zeros_like(matrix, dtype=bool)
    slices = (slice(0, n0), slice(n0, n0 + n1))
    for row_slice in slices:
        for col_slice in slices:
            block = matrix[row_slice, col_slice]
            row_k = max(1, int(np.ceil(percentage * block.shape[1])))
            col_k = max(1, int(np.ceil(percentage * block.shape[0])))
            block_mask = _row_or_col_topk_mask(block, row_k=row_k, col_k=col_k)
            if ensure_nonempty:
                empty_rows = np.where(~block_mask.any(axis=1))[0]
                if empty_rows.size:
                    block_mask[empty_rows, np.argmax(block[empty_rows], axis=1)] = True
                empty_cols = np.where(~block_mask.any(axis=0))[0]
                if empty_cols.size:
                    block_mask[np.argmax(block[:, empty_cols], axis=0), empty_cols] = True
            mask[row_slice, col_slice] = block_mask

    np.fill_diagonal(mask, False)
    sparse_matrix = np.where(mask, matrix, 0.0)
    # The selected array owns its buffer, so cleanup can run in place.
    sparse_matrix = np.nan_to_num(sparse_matrix, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    sparse_matrix[sparse_matrix < 0.0] = 0.0
    if binarize:
        sparse_matrix = mask.astype(float)
        np.fill_diagonal(sparse_matrix, 0.0)
    return sparse_matrix


def normalize_ot_tsne_graph(matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric, loop-free binary adjacency normalized to unit total mass."""
    graph = np.asarray(matrix, dtype=float)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError(f"Expected a square 2D graph matrix, got shape {graph.shape}.")

    # Copy before mutation because the caller may retain the input as an artifact.
    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0, copy=True)
    graph[graph < 0.0] = 0.0
    graph[:] = graph > 0.0
    # Form the symmetric boolean mask before assignment to avoid overlapping float input and output.
    graph[:] = np.logical_or(graph > 0.0, graph.T > 0.0)
    np.fill_diagonal(graph, 0.0)

    total_mass = float(graph.sum())
    if total_mass <= 0.0:
        raise ValueError("Graph has zero total mass after normalization.")
    graph /= total_mass
    return graph


#: Fix the ARPACK seed so repeated eigenspaces use a stable basis across experiment seeds.
#: Seed variation then enters through initialization jitter.
SPECTRAL_SOLVER_SEED = 0


def _flip_init_signs(init: np.ndarray) -> np.ndarray:
    """Make each column's largest-magnitude entry positive to fix singular-vector sign ambiguity."""
    pivots = np.argmax(np.abs(init), axis=0)
    signs = np.sign(init[pivots, np.arange(init.shape[1])])
    signs[signs == 0.0] = 1.0
    return init * signs


def _finalize_tsne_init(
    init: np.ndarray,
    init_scale: float = 10000.0,
    random_state: int = 0,
    jitter_scale: float = 0.01,
) -> np.ndarray:
    """Fix column signs, scale the first axis to ``1 / init_scale``, and add seeded jitter.

    Jitter allows seed variation for deterministic EOT and spectral coordinates.
    Reject non-finite inputs and a degenerate first axis rather than distort geometry.
    """
    init = np.asarray(init, dtype=float)
    if init.ndim != 2 or init.shape[1] != 2:
        raise ValueError(f"init must have shape (n, 2), got {init.shape}.")
    if not np.isfinite(init).all():
        raise ValueError(
            "Initialization contains non-finite values; the upstream decomposition failed."
        )

    init = _flip_init_signs(init)

    init_std = float(init[:, 0].std())
    if init_std < 1e-12:
        raise ValueError(
            "Initialization's first coordinate is degenerate (zero variance) and cannot be "
            "rescaled to the target standard deviation."
        )
    init = init / (init_std * float(init_scale))

    if jitter_scale > 0.0:
        init = jitter(
            init,
            scale=float(jitter_scale),
            random_state=np.random.RandomState(int(random_state)),
        )
    return np.ascontiguousarray(init, dtype=float)


def _is_symmetric(
    matrix: np.ndarray, rtol: float = 1e-7, atol: float = 1e-12, block: int = 1024
) -> bool:
    """Check symmetry in row blocks to bound temporary memory."""
    for start in range(0, matrix.shape[0], block):
        stop = min(start + block, matrix.shape[0])
        if not np.allclose(matrix[start:stop, :], matrix[:, start:stop].T, rtol=rtol, atol=atol):
            return False
    return True


def _spectral_init(graph: np.ndarray) -> tuple[np.ndarray | None, dict[str, object]]:
    """Initialize from diffusion coordinates of the joint OT affinity graph.

    Solve ``A v = lambda D v``, sort eigenpairs together, drop the leading mode,
    and weight the remaining vectors by their eigenvalues. Return ``(None, info)``
    for structural or numerical degeneracy so the caller can use random coordinates.
    Asymmetric input is an error: initialization must use the optimized affinities.
    """
    info: dict[str, object] = {}
    n_points = graph.shape[0]

    if not _is_symmetric(graph):
        raise ValueError(
            "Spectral initialization requires a symmetric graph; refusing to symmetrize it "
            "implicitly, which would initialize from different affinities than t-SNE "
            "optimizes."
        )

    # run_tsne_from_graph has already removed invalid values, negative weights, and self-loops.
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    if n_points < 4:  # eigsh needs k = n_components + 1 < n
        info["spectral_skipped"] = "n_points<4"
        return None, info
    if float(degrees.min()) <= 0.0:  # a zero-degree row makes D singular
        info["spectral_skipped"] = "zero_degree_row"
        return None, info

    adjacency = sp.csr_matrix(graph)
    v0 = np.random.RandomState(SPECTRAL_SOLVER_SEED).uniform(-1.0, 1.0, n_points)
    try:
        eigenvalues, eigenvectors = sp.linalg.eigsh(
            adjacency, M=sp.diags(degrees), k=3, tol=1e-4, which="LM", v0=v0
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        # ArpackNoConvergence < ArpackError < RuntimeError.
        info["spectral_skipped"] = type(error).__name__
        return None, info

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    embedding = eigenvectors[:, 1:] * eigenvalues[1:]

    if not np.isfinite(embedding).all():
        info["spectral_skipped"] = "non_finite"
        return None, info
    if float(embedding[:, 0].std()) < 1e-12:
        info["spectral_skipped"] = "degenerate_first_axis"
        return None, info

    info["spectral_eigenvalues"] = np.asarray(eigenvalues, dtype=float)
    # Count unit eigenvalues among the three retrieved modes, not all graph components.
    info["spectral_n_unit_eigenvalues_returned"] = int(
        np.isclose(eigenvalues, 1.0, atol=1e-6).sum()
    )
    # Disconnected components can dominate the initialization through component indicators.
    info["spectral_n_components"] = int(
        sp.csgraph.connected_components(adjacency, directed=False, return_labels=False)
    )
    return np.asarray(embedding, dtype=float), info


def _resolve_plan_init(
    init: str | np.ndarray,
    plan: np.ndarray,
    eot_coordinates: np.ndarray | None = None,
) -> str | np.ndarray:
    """Resolve ``"eot"`` to two nontrivial stacked singular modes of the plan.

    For ``E = [sqrt(n0) U; sqrt(n1) V]``, ``E.T @ E = (n0 + n1) I``, so its
    columns are eigenvectors of ``E diag(s) E.T / sqrt(n0 * n1)``. Use the shared
    plan basis across variants to keep initialization independent of diagonal changes.
    """
    if isinstance(init, np.ndarray) or init in ("random", "spectral"):
        return init
    if init != "eot":
        raise ValueError(
            f"init must be one of {OT_TSNE_INIT_CHOICES} or an (n, 2) array, got {init!r}."
        )
    if eot_coordinates is None:
        eot = np.asarray(
            compute_eot_embedding_from_plan(plan, t=0.0, embd_dim=2, start=1), dtype=float
        )
    else:
        # At t=0, singular-value weights are one; the stacked basis needs no rescaling.
        eot = np.asarray(eot_coordinates[:, 1:3], dtype=float)
    if eot.shape[1] < 2:
        raise ValueError(
            "init='eot' needs two non-trivial singular modes of the transport plan, but "
            f"only {eot.shape[1]} were available."
        )
    return eot


def run_umap_from_graph(
    knn_indices: np.ndarray,
    x_joint: np.ndarray,
    random_state: int = 0,
    n_epochs: int = 200,
    init: str | np.ndarray = "random",
    min_dist: float = 0.1,
    spread: float = 1.0,
    n_jobs: int = 1,
    return_artifacts: bool = False,
    timer: object | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Embed directed OT neighbors with unit distances and fuzzy-union symmetrization.

    Unit distances give each edge membership 1, so UMAP preserves the supplied
    unweighted graph. ``x_joint`` supplies row counts and input-space centroids for
    spectral initialization of disconnected components. PCA initialization is rejected
    because it would introduce input geometry into a graph-based comparison.
    """
    knn_indices = np.asarray(knn_indices)
    if knn_indices.ndim != 2:
        raise ValueError(f"knn_indices must be 2D, got shape {knn_indices.shape}.")
    x_joint = _validate_matrix(x_joint, "x_joint")
    n_points, k = knn_indices.shape
    if x_joint.shape[0] != n_points:
        raise ValueError(f"x_joint has {x_joint.shape[0]} rows but knn_indices has {n_points}.")
    if isinstance(init, str) and init == "pca":
        raise ValueError("init='pca' reads x_joint; use 'random' or 'spectral'.")
    if isinstance(init, np.ndarray):
        init = np.asarray(init, dtype=float)
        if init.shape != (n_points, 2):
            raise ValueError(
                f"A precomputed UMAP init must have shape {(n_points, 2)}, got {init.shape}."
            )
        if not np.isfinite(init).all():
            raise ValueError("A precomputed UMAP init must contain only finite values.")
    n_epochs = int(n_epochs)
    if n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive, got {n_epochs}.")

    knn_indices = np.ascontiguousarray(knn_indices, dtype=np.int32)
    knn_dists = np.ones_like(knn_indices, dtype=np.float32)

    reducer = umap.UMAP(
        n_components=2,
        # Match the supplied width: a smaller value truncates edges; a larger one may
        # make UMAP recompute neighbors from input features.
        n_neighbors=k,
        precomputed_knn=(knn_indices, knn_dists),
        # Use the union to preserve choices made by either endpoint.
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
        # Jaccard, Hellinger, and Dice disconnect edges at distance 1; Euclidean does not.
        metric="euclidean",
        min_dist=float(min_dist),
        spread=float(spread),
        init=init,
        n_epochs=n_epochs,
        random_state=int(random_state),
        n_jobs=int(n_jobs),
        densmap=False,  # needs real distances
        unique=False,  # unsupported with precomputed_knn, and would reindex the rows
        verbose=False,
    )
    with warnings.catch_warnings():
        # No search index is supplied, so transform is unavailable for this fixed graph.
        warnings.filterwarnings("ignore", message=".*knn_search_index.*")
        embedding = np.asarray(reducer.fit_transform(x_joint), dtype=float)
    if timer is not None:
        # Unit-distance membership preserves the OT graph; its construction was timed as OT
        # postprocessing.
        timer.seconds.pop("time_embedding_graph", None)

    # Check UMAP behavior required to preserve the supplied graph.
    if reducer._disconnection_distance != np.inf:
        raise AssertionError(
            f"disconnection distance is {reducer._disconnection_distance}, not infinite; "
            "the constant distance of 1 would have cut edges."
        )
    if reducer._knn_indices.shape != knn_indices.shape:
        raise AssertionError(
            f"UMAP used a neighbour list of shape {reducer._knn_indices.shape}, "
            f"not the {knn_indices.shape} it was given."
        )
    if not np.array_equal(reducer._knn_indices, knn_indices):
        raise AssertionError("UMAP replaced the supplied neighbour list.")
    graph = reducer.graph_.tocoo()
    weights = np.asarray(graph.data, dtype=float)
    if weights.size and not np.allclose(weights, 1.0):
        raise AssertionError(
            f"graph weights are not all 1 (min {weights.min()}, max {weights.max()}); "
            "the affinity stage was not the identity it is assumed to be."
        )

    if not return_artifacts:
        return embedding
    artifacts: dict[str, object] = {
        "umap_n_neighbors": int(k),
        "umap_min_dist": float(min_dist),
        "umap_spread": float(spread),
        "umap_init": init if isinstance(init, str) else "array",
        "umap_n_epochs": n_epochs,
        "umap_a": float(getattr(reducer, "_a", float("nan"))),
        "umap_b": float(getattr(reducer, "_b", float("nan"))),
        "umap_graph_edges": int(graph.nnz),
        "random_state": int(random_state),
        "n_jobs": int(n_jobs),
    }
    return embedding, artifacts


def run_tsne_from_graph(
    graph: np.ndarray,
    random_state: int = 0,
    n_epochs: int = DEFAULT_TSNE_N_ITER,
    init_scale: float = 10000.0,
    init: str | np.ndarray = "random",
    init_info: dict[str, object] | None = None,
    timer: object | None = None,
    n_jobs: int = 1,
    early_exaggeration_iter: int | None = DEFAULT_TSNE_EARLY_EXAGGERATION_ITER,
) -> np.ndarray:
    """Optimize a 2D t-SNE layout from dense or sparse joint affinities.

    ``init`` accepts random coordinates, graph diffusion coordinates, or an ``(n, 2)``
    array. Spectral degeneracy falls back to random initialization; ``init_info``
    records the actual scheme and diagnostics in place.

    By default this is openTSNE's schedule: ``early_exaggeration_iter`` (250) exaggerated
    iterations, then ``n_epochs`` (500) main-phase iterations. ``early_exaggeration_iter=None``
    restores the schedule of earlier runs: ``n_epochs`` is the total and its first third is
    exaggerated (they used ``n_epochs=200``).
    """
    sparse_input = sp.issparse(graph)
    if sparse_input:
        # Our own copy: openTSNE rescales P.data in place during the fit.
        graph = sp.csr_matrix(graph, dtype=float, copy=True)
    else:
        graph = np.asarray(graph, dtype=float)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError(f"Expected a square 2D graph matrix, got shape {graph.shape}.")

    n_epochs = int(n_epochs)
    if n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive, got {n_epochs}.")

    if sparse_input:
        # Match dense cleaning and discard entries that become zero.
        graph.sum_duplicates()
        graph.data = np.nan_to_num(graph.data, nan=0.0, posinf=0.0, neginf=0.0)
        graph.data[graph.data < 0.0] = 0.0
        graph.setdiag(0.0)
        graph.eliminate_zeros()
        graph.sort_indices()
        total_mass = float(graph.data.sum())
        if total_mass <= 0.0:
            raise ValueError("Graph has zero total mass; cannot run t-SNE.")
        graph.data /= total_mass
    else:
        graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
        graph[graph < 0.0] = 0.0
        np.fill_diagonal(graph, 0.0)

        total_mass = float(graph.sum())
        if total_mass <= 0.0:
            raise ValueError("Graph has zero total mass; cannot run t-SNE.")
        graph = graph / total_mass

    n_points = graph.shape[0]
    info: dict[str, object] = {}
    resolved: np.ndarray | None = None
    # Time initialization here because openTSNE receives prepared coordinates.
    _init_start = time.perf_counter()
    if isinstance(init, np.ndarray):
        candidate = np.asarray(init, dtype=float)
        if candidate.shape != (n_points, 2):
            raise ValueError(
                f"A precomputed init must have shape {(n_points, 2)}, got {candidate.shape}."
            )
        resolved = _finalize_tsne_init(candidate, init_scale, int(random_state))
        info["init_used"] = "array"
    elif init == "spectral":
        # The spectral initializer requires dense input.
        candidate, spectral_info = _spectral_init(graph.toarray() if sparse_input else graph)
        info.update(spectral_info)
        if candidate is None:
            warnings.warn(
                "Spectral initialization is unavailable for this graph "
                f"({spectral_info.get('spectral_skipped', 'unknown')}); falling back to the "
                "random initialization.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            resolved = _finalize_tsne_init(candidate, init_scale, int(random_state))
            info["init_used"] = "spectral"
    elif init != "random":
        raise ValueError(
            f"init must be one of {OT_TSNE_INIT_CHOICES} or an (n, 2) array, got {init!r}."
        )

    if resolved is None:
        # Random initialization also serves as the spectral fallback.
        # Do not apply the sign fixing or jitter used for non-random coordinates.
        resolved = np.random.RandomState(int(random_state)).randn(n_points, 2)
        init_std = float(resolved[:, 0].std())
        resolved = resolved / (max(init_std, 1e-12) * float(init_scale))
        info.setdefault("init_used", "random")

    if timer is not None:
        timer.record("time_embedding_initialization", time.perf_counter() - _init_start)

    if init_info is not None:
        init_info.update(info)

    init = resolved
    if early_exaggeration_iter is None:
        n_early = n_epochs // 3
        n_late = n_epochs - n_early
    else:
        n_early = int(early_exaggeration_iter)
        if n_early < 0:
            raise ValueError(f"early_exaggeration_iter must be non-negative, got {n_early}.")
        n_late = n_epochs

    # Supply OT affinities directly; graph construction is timed as OT postprocessing.
    affinities = Affinities()
    affinities.P = graph if sparse_input else sp.csr_matrix(graph)
    embedder = TSNE(
        n_components=2,
        random_state=int(random_state),
        verbose=False,
        early_exaggeration_iter=n_early,
        n_iter=n_late,
        n_jobs=int(n_jobs),
    )
    return np.asarray(embedder.fit(affinities=affinities, initialization=init))


def _embed_ot_graph(
    *,
    backend: str,
    graph: np.ndarray,
    four_block_matrix: np.ndarray | None,
    n0: int,
    n1: int,
    effective_k: int | None,
    within_batch_indices: tuple[np.ndarray, np.ndarray] | None,
    x_joint: np.ndarray | None,
    random_state: int,
    n_epochs: int,
    init_scale: float,
    init,
    init_info: dict[str, object],
    cross_batch_only: bool = False,
    knn_indices: np.ndarray | None = None,
    timer: object | None = None,
    n_jobs: int = 1,
    early_exaggeration_iter: int | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Dispatch OT affinities to t-SNE or directed OT neighbors to UMAP.

    UMAP uses supplied ``knn_indices`` or derives them from ``four_block_matrix``;
    ``cross_batch_only`` restricts that derivation to opposite-batch neighbors.
    Its fuzzy union preserves edge support, while t-SNE receives normalized affinities.
    """
    if backend == "tsne":
        embedding = run_tsne_from_graph(
            graph,
            random_state=random_state,
            n_epochs=n_epochs,
            init_scale=init_scale,
            init=init,
            init_info=init_info,
            timer=timer,
            n_jobs=n_jobs,
            early_exaggeration_iter=early_exaggeration_iter,
        )
        return embedding, {"embedding_backend": "tsne"}
    # early_exaggeration_iter is a t-SNE parameter; UMAP has no such phase and ignores it.
    if backend != "umap":
        raise ValueError(f"backend must be 'tsne' or 'umap', got {backend!r}.")

    if four_block_matrix is None and knn_indices is None:
        raise ValueError(
            "the UMAP backend needs the four-block matrix to build its neighbour list."
        )
    if effective_k is None:
        raise ValueError(
            "the UMAP backend needs effective_k. A percentage-based sparsification gives rows "
            "different degrees, which a dense neighbour list cannot represent."
        )
    if x_joint is None:
        raise ValueError(
            "the UMAP backend needs x_joint: a spectral initialisation of a disconnected graph "
            "places each component by its centroid in the input space."
        )
    # Directed-neighbor extraction belongs to OT postprocessing, not backend affinity construction.
    with _stage(timer, "time_ot_postprocess"):
        if knn_indices is not None:
            pass
        elif cross_batch_only:
            if within_batch_indices is not None:
                raise ValueError(
                    "cross_batch_only and within_batch_indices are mutually exclusive: the first "
                    "says there are no within-batch neighbours, the second supplies some."
                )
            knn_indices = cross_batch_directed_knn(
                four_block_matrix, n0=n0, n1=n1, effective_k=effective_k
            )
        else:
            knn_indices, _ = four_block_directed_knn(
                four_block_matrix,
                n0=n0,
                n1=n1,
                effective_k=effective_k,
                within_batch_indices=within_batch_indices,
            )
    embedding, artifacts = run_umap_from_graph(
        knn_indices,
        x_joint,
        random_state=random_state,
        n_epochs=n_epochs,
        init=init,
        n_jobs=n_jobs,
        return_artifacts=True,
        timer=timer,
    )
    return embedding, {"embedding_backend": "umap", **artifacts}
