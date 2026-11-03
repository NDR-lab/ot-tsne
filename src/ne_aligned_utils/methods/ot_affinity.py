"""Build sparse OT-tSNE affinities from score blocks and selected index pairs.

The cross block uses the plan directly. SVD reconstruction is equivalent in exact
arithmetic, but rounding can change near-tied selections. Within-batch blocks retain
the dense construction's arithmetic; selection retains its tie-breaking rules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ne_aligned_utils.methods.common import validate_batch_sizes as _validate_batch_sizes
from ne_aligned_utils.methods.ot_graph import (
    DEFAULT_GRAPH_SYMMETRIZATION,
    EffectiveKSplit,
    _mutual_topk_mask,
    _row_or_col_topk_mask,
    _row_topk_indices,
    _symmetrised,
    _topk_scores,
    _validate_symmetrization,
    _variable_topk_mask_along,
    resolve_cross_batch_k,
    resolve_effective_k_split,
)

#: Budget rules: proportional blocks, mass-aware cross quotas, full cross quotas,
#: or proportional cross quotas without within-batch edges.
SELECTION_RULES = ("split", "mass", "cross_only", "cross_split")


# Blocks


@dataclass
class PlanSpectrum:
    """Store the plan's thin SVD as stacked coordinates ``E = [sqrt(n0) U; sqrt(n1) V]``.

    Blocks use ``(E * s) @ E.T / sqrt(n0 * n1)``. Scaling rows on demand avoids
    storing a second weighted basis without changing the elementwise arithmetic.
    """

    n0: int
    n1: int
    singular_values: np.ndarray
    eot: np.ndarray

    @classmethod
    def from_plan(cls, plan: np.ndarray) -> PlanSpectrum:
        """Use the plan for cross scores and ``source`` for within-batch scores.

        ``PlanDiagonals`` requires a matching spectrum. ``WithinBatchKnn`` supplies fixed
        neighbors separately; it and ``None`` provide no within-batch score blocks.
        """
        plan = np.asarray(plan, dtype=float)
        if plan.ndim != 2:
            raise ValueError(f"plan must be a 2D array, got shape {plan.shape!r}.")
        n0, n1 = plan.shape
        u, singular_values, vt = np.linalg.svd(plan, full_matrices=False)
        # Scale directly into the stacked basis to avoid scaled copies of both singular factors.
        eot = np.empty((n0 + n1, singular_values.size), dtype=np.result_type(u, vt))
        np.multiply(u, np.sqrt(n0), out=eot[:n0])
        del u
        np.multiply(vt.T, np.sqrt(n1), out=eot[n0:])
        del vt
        return cls(n0, n1, singular_values, eot)

    def _rows(self, batch: int) -> slice:
        return slice(0, self.n0) if batch == 0 else slice(self.n0, self.n0 + self.n1)

    def raw_block(self, row_batch: int, col_batch: int) -> np.ndarray:
        """Return a cleaned block before symmetrization, with self-scores removed."""
        weighted_rows = self.eot[self._rows(row_batch)] * self.singular_values
        block = (weighted_rows @ self.eot[self._rows(col_batch)].T) / np.sqrt(self.n0 * self.n1)
        del weighted_rows
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        block[block < 0.0] = 0.0
        if row_batch == col_batch:
            np.fill_diagonal(block, 0.0)
        return block

    def diagonal_block(self, batch: int) -> np.ndarray:
        """Return a symmetrized within-batch block from the plan spectrum."""
        return _symmetrised(self.raw_block(batch, batch))

    def eot_coordinates(self) -> np.ndarray:
        return self.eot


def plan_cross_block(plan: np.ndarray) -> np.ndarray:
    """Return the plan with non-finite and negative entries replaced by zero.

    A finite, nonnegative plan is returned without copying; selectors must not mutate it.
    The opposite cross block is its transpose, so no averaging is needed.
    """
    plan = np.asarray(plan, dtype=float)
    if plan.ndim != 2:
        raise ValueError(f"plan must be a 2D array, got shape {plan.shape!r}.")
    if np.isfinite(plan).all() and not (plan < 0.0).any():
        return plan
    cleaned = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0, copy=True)
    cleaned[cleaned < 0.0] = 0.0
    return cleaned


def replacement_diagonal_block(diagonal: np.ndarray) -> np.ndarray:
    """Copy a supplied within-batch block, remove self-scores, and symmetrize it."""
    block = np.array(diagonal, dtype=float, copy=True)
    np.fill_diagonal(block, 0.0)
    return _symmetrised(block)


@dataclass(frozen=True)
class PlanDiagonals:
    """Within-batch blocks taken from the plan's SVD: the default construction (``ot``, UOT)."""


@dataclass(frozen=True)
class SuppliedDiagonals:
    """Within-batch score blocks computed by the caller (Gram, self-OT), selected like the rest."""

    block0: np.ndarray
    block1: np.ndarray


@dataclass(frozen=True)
class WithinBatchKnn:
    """Euclidean kNN neighbors that replace the plan's within-batch blocks (``ot-knndiag``).

    ``indices0`` and ``indices1`` are block-local, of shape ``(n0, batch0_k)`` and
    ``(n1, batch1_k)``. ``symmetrization`` turns each directed list into an undirected block:
    ``"or"`` keeps an edge either endpoint chose, ``"and"`` one both chose.
    """

    indices0: np.ndarray
    indices1: np.ndarray
    symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION

    def adjacency(self, batch: int) -> sp.csr_matrix:
        """The undirected 0/1 block, self-loops removed (``_adjacency_from_indices``)."""
        indices = np.asarray(self.indices0 if batch == 0 else self.indices1)
        n, k = indices.shape
        directed = sp.csr_matrix(
            (np.ones(n * k), (np.repeat(np.arange(n), k), indices.ravel())), shape=(n, n)
        )
        directed.data[:] = 1.0  # a repeated neighbor is one edge, not a weight of two
        combined = (
            directed.maximum(directed.T)
            if self.symmetrization == "or"
            else directed.minimum(directed.T)
        )
        combined = sp.csr_matrix(combined)
        combined.setdiag(0.0)
        combined.eliminate_zeros()
        return combined


#: Within-batch source; None excludes within-batch selection.
DiagonalSource = PlanDiagonals | SuppliedDiagonals | WithinBatchKnn | None


class BlockScores:
    """Provide validated cross-batch and optional within-batch score blocks on demand.

    The cross block has shape ``(n0, n1)``; its transpose is the opposite block.
    Blocks are not cached, so callers control their lifetimes.
    """

    def __init__(
        self,
        n0: int,
        n1: int,
        cross: Callable[[], np.ndarray],
        diagonal: Callable[[int], np.ndarray] | None = None,
    ):
        self.n0, self.n1 = _validate_batch_sizes(n0, n1)
        self._cross = cross
        self._diagonal = diagonal

    @property
    def has_diagonals(self) -> bool:
        return self._diagonal is not None

    def cross(self) -> np.ndarray:
        block = self._cross()
        if block.shape != (self.n0, self.n1):
            raise ValueError(
                f"cross block must have shape {(self.n0, self.n1)}, got {block.shape}."
            )
        return block

    def diagonal(self, batch: int) -> np.ndarray:
        if self._diagonal is None:
            raise ValueError("these block scores have no within-batch blocks.")
        n = (self.n0, self.n1)[batch]
        block = self._diagonal(batch)
        if block.shape != (n, n):
            raise ValueError(f"diagonal block {batch} must have shape {(n, n)}, got {block.shape}.")
        return block

    @classmethod
    def from_plan(
        cls,
        plan: np.ndarray,
        source: DiagonalSource,
        spectrum: PlanSpectrum | None = None,
    ) -> BlockScores:
        """Blocks for one variant: the plan as the cross block, diagonals as ``source`` says.

        ``spectrum`` is needed only for :class:`PlanDiagonals`. ``WithinBatchKnn`` and ``None``
        give no score blocks within a batch: the first supplies fixed neighbors instead, the
        second selects nothing within a batch.
        """
        plan = np.asarray(plan, dtype=float)
        if plan.ndim != 2:
            raise ValueError(f"plan must be a 2D array, got shape {plan.shape!r}.")
        sizes = plan.shape
        if isinstance(source, PlanDiagonals):
            if spectrum is None:
                raise ValueError("PlanDiagonals needs the plan's SVD (spectrum=...).")
            if (spectrum.n0, spectrum.n1) != sizes:
                raise ValueError(
                    f"spectrum is for a {(spectrum.n0, spectrum.n1)} plan, not {sizes}."
                )
            diagonal = spectrum.diagonal_block
        elif isinstance(source, SuppliedDiagonals):
            supplied = (source.block0, source.block1)
            for batch, block in enumerate(supplied):
                if np.shape(block) != (sizes[batch], sizes[batch]):
                    raise ValueError(
                        f"supplied diagonal block {batch} must have shape "
                        f"{(sizes[batch],) * 2}, got {np.shape(block)}."
                    )

            def diagonal(batch: int) -> np.ndarray:
                return replacement_diagonal_block(supplied[batch])

        elif source is None or isinstance(source, WithinBatchKnn):
            diagonal = None
        else:
            raise TypeError(
                "within-batch source must be PlanDiagonals, SuppliedDiagonals, WithinBatchKnn "
                f"or None, got {type(source).__name__}."
            )
        return cls(sizes[0], sizes[1], lambda: plan_cross_block(plan), diagonal)

    @classmethod
    def from_arrays(
        cls, cross: np.ndarray, diagonals: tuple[np.ndarray, np.ndarray] | None = None
    ) -> BlockScores:
        """Wrap precomputed, symmetrized score blocks."""
        n0, n1 = np.shape(cross)
        return cls(
            n0,
            n1,
            lambda: cross,
            None if diagonals is None else (lambda batch: diagonals[batch]),
        )


# Selection


@dataclass
class BlockSelection:
    """Store selected global index pairs, budget splits, and diagonal edge counts.

    Optional ``knn_indices`` contains directed neighbors from actual selections, without fillers.
    """

    rows: np.ndarray
    cols: np.ndarray
    split: EffectiveKSplit | None
    diag_edges: tuple[int, int]
    knn_indices: np.ndarray | None = None


#: Target working-memory budget beyond the score block; larger selections use slabs.
SELECTION_TEMP_BUDGET_BYTES = 64 * 2**20

#: Estimated bytes per entry for score copies, partition indices, and masks.
#: Diagonal exclusion adds another score copy; mutual selection copies transposed scores.
_FULL_SELECTION_BYTES_PER_ENTRY = {
    "or": 8 + 8 + 8 + 1,
    "and": 8 + 8 + 8 + 8 + 3,
    "mass": 8 + 8 + 8 + 3,
}


def _full_selection_bytes(n_rows: int, n_cols: int, mode: str, exclude_diagonal: bool) -> int:
    per_entry = _FULL_SELECTION_BYTES_PER_ENTRY[mode] + (8 if exclude_diagonal else 0)
    return per_entry * n_rows * n_cols


def _check_capacity(n_rows, n_cols, row_k, col_k, exclude_diagonal) -> None:
    """The capacity rule of ``_row_or_col_topk_mask``, for the slab path."""
    if row_k < 0 or col_k < 0:
        raise ValueError(f"row_k and col_k must be nonnegative, got {row_k}, {col_k}.")
    row_capacity = n_cols - 1 if exclude_diagonal else n_cols
    col_capacity = n_rows - 1 if exclude_diagonal else n_rows
    if row_k > row_capacity:
        raise ValueError(f"row_k={row_k} exceeds block row capacity {row_capacity}.")
    if col_k > col_capacity:
        raise ValueError(f"col_k={col_k} exceeds block column capacity {col_capacity}.")


def _slab_scores(block, rows: slice, cols: slice, exclude_diagonal: bool) -> np.ndarray:
    """``_topk_scores`` restricted to a slab: the same cleaning, the self entries set to -inf."""
    scores = np.nan_to_num(block[rows, cols], nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    if exclude_diagonal:
        # A within-batch block is square; global row i and column i are the same point.
        start = max(rows.start or 0, cols.start or 0)
        stop = min(rows.stop or block.shape[0], cols.stop or block.shape[1])
        if start < stop:
            idx = np.arange(start, stop)
            scores[idx - (rows.start or 0), idx - (cols.start or 0)] = -np.inf
    return scores


def _slabs(length: int, step: int):
    for start in range(0, length, step):
        yield slice(start, min(start + step, length))


def _block_keys_full(block, mode, row_k, col_k, exclude_diagonal, row_budgets, col_budgets):
    """Selected pairs of one block, selected whole with the dense path's own helpers."""
    n_cols = block.shape[1]
    row_idx = None
    if mode == "mass":
        scores = _topk_scores(block, exclude_diagonal=False)
        mask = _variable_topk_mask_along(scores, row_budgets, k_max=row_k, axis=1)
        mask |= _variable_topk_mask_along(scores, col_budgets, k_max=col_k, axis=0)
    else:
        mask_fn = _row_or_col_topk_mask if mode == "or" else _mutual_topk_mask
        mask, row_idx = mask_fn(
            block,
            row_k=row_k,
            col_k=col_k,
            exclude_diagonal=exclude_diagonal,
            return_row_indices=True,
        )
    r, c = np.nonzero(mask)
    return r.astype(np.int64) * n_cols + c, row_idx


def _block_keys_slabbed(
    block, mode, row_k, col_k, exclude_diagonal, row_budgets, col_budgets, budget_bytes
):
    """Select pairs in slabs of complete rows and columns.

    Each partition sees the same values as a full-block pass, preserving tie-breaking.
    Combine passes by union, or intersection for mutual selection.
    """
    n_rows, n_cols = block.shape
    _check_capacity(n_rows, n_cols, row_k, col_k, exclude_diagonal)
    # Per slab: the score copy and argpartition's full-size index array (a transposed copy too
    # for the mutual column pass), so 16-24 bytes per entry of the slab.
    rows_per_slab = max(1, budget_bytes // (16 * n_cols))
    cols_per_slab = max(1, budget_bytes // (24 * n_rows))

    row_keys = []
    row_idx = np.empty((n_rows, row_k), dtype=int) if mode != "mass" else None
    if row_k > 0:
        for rows in _slabs(n_rows, rows_per_slab):
            scores = _slab_scores(block, rows, slice(None), exclude_diagonal)
            if mode == "mass":
                mask = _variable_topk_mask_along(
                    scores, row_budgets[rows], k_max=row_k, axis=1
                )
                r, c = np.nonzero(mask)
                row_keys.append((r + rows.start).astype(np.int64) * n_cols + c)
                continue
            idx = _row_topk_indices(scores, row_k)
            row_idx[rows] = idx
            r = np.repeat(np.arange(rows.start, rows.stop, dtype=np.int64), row_k)
            row_keys.append(r * n_cols + idx.ravel())

    col_keys = []
    if col_k > 0:
        for cols in _slabs(n_cols, cols_per_slab):
            scores = _slab_scores(block, slice(None), cols, exclude_diagonal)
            width = cols.stop - cols.start
            if mode == "mass":
                mask = _variable_topk_mask_along(
                    scores, col_budgets[cols], k_max=col_k, axis=0
                )
                r, c = np.nonzero(mask)
                col_keys.append(r.astype(np.int64) * n_cols + (c + cols.start))
                continue
            if mode == "and":
                # As _mutual_topk_mask: the row pass of the transposed scores.
                idx = _row_topk_indices(np.ascontiguousarray(scores.T), col_k)  # (width, col_k)
                r = idx.ravel().astype(np.int64)
            elif col_k == n_rows:
                r = np.tile(np.arange(n_rows, dtype=np.int64), width)
            else:
                # As _row_or_col_topk_mask: argpartition down each column.
                idx = np.argpartition(scores, n_rows - col_k, axis=0)[-col_k:, :]  # (col_k, w)
                r = idx.T.ravel().astype(np.int64)
            c = np.repeat(np.arange(cols.start, cols.stop, dtype=np.int64), col_k)
            col_keys.append(r * n_cols + c)

    row_keys = np.concatenate(row_keys) if row_keys else np.empty(0, dtype=np.int64)
    col_keys = np.concatenate(col_keys) if col_keys else np.empty(0, dtype=np.int64)
    if mode == "and":
        keys = np.intersect1d(row_keys, col_keys)
    else:
        keys = np.union1d(row_keys, col_keys)
    if exclude_diagonal:
        keys = keys[keys // n_cols != keys % n_cols]
    return keys, row_idx


def select_block(
    block: np.ndarray,
    *,
    mode: str,
    row_k: int,
    col_k: int,
    exclude_diagonal: bool,
    row_budgets: np.ndarray | None = None,
    col_budgets: np.ndarray | None = None,
    binarize: bool = True,
    budget_bytes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """One block's selected pairs ``(rows, cols)``, plus the row pass's indices.

    ``mode`` is ``"or"`` (union of the row and column passes, the default construction), ``"and"``
    (their intersection, the mutual ablation) or ``"mass"`` (per-cell budgets on a cross block).
    The row indices are what UMAP's directed list is made of; they are ``None`` under ``"mass"``.

    The whole block is selected at once when its working arrays fit ``budget_bytes`` (default
    :data:`SELECTION_TEMP_BUDGET_BYTES`), with the dense path's own helpers; otherwise in slabs.
    Both give the same pairs. With ``binarize=False`` a selected pair is kept only if its value is
    positive and finite, as :func:`normalize_ot_tsne_graph` drops the rest.
    """
    if mode not in _FULL_SELECTION_BYTES_PER_ENTRY:
        modes = tuple(_FULL_SELECTION_BYTES_PER_ENTRY)
        raise ValueError(f"mode must be one of {modes}, got {mode!r}.")
    if mode == "mass" and (row_budgets is None or col_budgets is None or exclude_diagonal):
        raise ValueError("mode='mass' needs row and column budgets and applies to cross blocks.")
    budget_bytes = SELECTION_TEMP_BUDGET_BYTES if budget_bytes is None else int(budget_bytes)
    n_rows, n_cols = block.shape
    args = (block, mode, int(row_k), int(col_k), exclude_diagonal, row_budgets, col_budgets)
    if _full_selection_bytes(n_rows, n_cols, mode, exclude_diagonal) <= budget_bytes:
        keys, row_idx = _block_keys_full(*args)
    else:
        keys, row_idx = _block_keys_slabbed(*args, budget_bytes)
    r, c = keys // n_cols, keys % n_cols
    if not binarize:
        values = block[r, c]
        keep = np.isfinite(values) & (values > 0.0)
        r, c = r[keep], c[keep]
    return r, c, row_idx


def _validate_request(
    *,
    rule: str,
    has_diagonals: bool,
    within_batch_knn: WithinBatchKnn | None,
    mass_budgets,
    graph_symmetrization: str,
    directed_knn: bool,
) -> None:
    """Refuse every combination that would build a different graph from the one asked for."""
    if rule not in SELECTION_RULES:
        raise ValueError(f"rule must be one of {SELECTION_RULES}, got {rule!r}.")
    if rule in ("split", "mass"):
        # Split and mass rules require exactly one within-batch source.
        if has_diagonals == (within_batch_knn is not None):
            raise ValueError(
                f"rule={rule!r} needs exactly one source for the within-batch blocks: "
                "diagonal score blocks to select from, or `within_batch_knn` (fixed neighbours)."
            )
    elif has_diagonals or within_batch_knn is not None:
        raise ValueError(f"rule={rule!r} selects no within-batch edges; pass no diagonal source.")
    if rule == "mass":
        if mass_budgets is None:
            raise ValueError("rule='mass' needs mass_budgets=(row_budgets, col_budgets).")
        if within_batch_knn is not None:
            raise ValueError("rule='mass' is defined only with plan-derived diagonal blocks.")
        if graph_symmetrization != DEFAULT_GRAPH_SYMMETRIZATION:
            raise ValueError("rule='mass' has no mutual ('and') mode.")
    elif mass_budgets is not None:
        raise ValueError(f"mass_budgets are only used by rule='mass', not rule={rule!r}.")
    if directed_knn and rule in ("mass", "cross_split"):
        # Fixed-width directed lists cannot express variable per-point budgets.
        raise ValueError(f"rule={rule!r} gives points different degrees; no directed list exists.")


def select_four_block(
    scores: BlockScores,
    *,
    effective_k: int,
    rule: str = "split",
    within_batch_knn: WithinBatchKnn | None = None,
    graph_symmetrization: str = DEFAULT_GRAPH_SYMMETRIZATION,
    mass_budgets: tuple[np.ndarray, np.ndarray] | None = None,
    binarize: bool = True,
    directed_knn: bool = False,
    selection_budget_bytes: int | None = None,
) -> BlockSelection:
    """Select global neighbor pairs one score block at a time.

    Split and mass rules require either within-batch scores or fixed kNN neighbors;
    cross-only rules accept neither. ``directed_knn`` also returns the row selections
    for UMAP. ``selection_budget_bytes`` controls the working arrays in ``select_block``.
    """
    n0, n1 = scores.n0, scores.n1
    graph_symmetrization = _validate_symmetrization(graph_symmetrization)
    _validate_request(
        rule=rule,
        has_diagonals=scores.has_diagonals,
        within_batch_knn=within_batch_knn,
        mass_budgets=mass_budgets,
        graph_symmetrization=graph_symmetrization,
        directed_knn=directed_knn,
    )
    sizes = (n0, n1)
    offsets = (0, n0)

    if rule == "cross_only":
        split = None
        k = resolve_cross_batch_k(effective_k=effective_k, n0=n0, n1=n1)
        quotas = {(0, 1): (k, k), (1, 0): (k, k)}
        knn_widths = {(0, 1): k, (1, 0): k}
    else:
        split = resolve_effective_k_split(effective_k=effective_k, n0=n0, n1=n1)
        ks = (split.batch0_k, split.batch1_k)
        # row_k is the column batch's quota and col_k the row batch's, as in the dense sparsifier.
        quotas = {(rb, cb): (ks[cb], ks[rb]) for rb in (0, 1) for cb in (0, 1)}
        knn_widths = {(rb, cb): ks[cb] for rb in (0, 1) for cb in (0, 1)}

    if rule == "mass":
        budgets = tuple(np.asarray(b, dtype=int) for b in mass_budgets)
        for batch, budget in enumerate(budgets):
            if budget.shape != (sizes[batch],):
                raise ValueError(
                    f"budgets for batch {batch} have shape {budget.shape}, "
                    f"expected {(sizes[batch],)}"
                )
            if budget.min() < 0 or budget.max() > ks[1 - batch]:
                raise ValueError(
                    f"budgets for batch {batch} must lie in [0, {ks[1 - batch]}], "
                    f"got [{budget.min()}, {budget.max()}]"
                )
    if within_batch_knn is not None:
        for batch, indices in enumerate((within_batch_knn.indices0, within_batch_knn.indices1)):
            expected = (sizes[batch], ks[batch])
            if np.shape(indices) != expected:
                raise ValueError(
                    f"within-batch indices for batch {batch} have shape {np.shape(indices)}, "
                    f"expected {expected}"
                )

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    diag_edges = [0, 0]
    knn_parts: dict[tuple[int, int], np.ndarray] = {}

    def select(rb: int, cb: int, block: np.ndarray) -> None:
        row_k, col_k = quotas[(rb, cb)]
        within = rb == cb
        mass_cross = rule == "mass" and not within
        r, c, row_idx = select_block(
            block,
            mode="mass" if mass_cross else ("or" if rule == "mass" else graph_symmetrization),
            row_k=row_k,
            col_k=col_k,
            exclude_diagonal=within,
            row_budgets=budgets[rb] if mass_cross else None,
            col_budgets=budgets[cb] if mass_cross else None,
            binarize=binarize,
            budget_bytes=selection_budget_bytes,
        )
        if within:
            diag_edges[rb] = int(r.size)
        rows.append(r + offsets[rb])
        cols.append(c + offsets[cb])
        if directed_knn and knn_widths[(rb, cb)] > 0:
            # Reuse row selections so UMAP neighbors have identical tie-breaking.
            if row_idx.shape[1] != knn_widths[(rb, cb)]:
                raise AssertionError("the row pass and the directed list have different widths.")
            knn_parts[(rb, cb)] = row_idx + offsets[cb]

    cross = scores.cross()
    select(0, 1, cross)
    cross_t = np.ascontiguousarray(cross.T)
    del cross
    select(1, 0, cross_t)
    del cross_t
    if scores.has_diagonals and rule in ("split", "mass"):
        for batch in (0, 1):
            select(batch, batch, scores.diagonal(batch))

    if within_batch_knn is not None:
        fixed = (np.asarray(within_batch_knn.indices0), np.asarray(within_batch_knn.indices1))
        for batch in (0, 1):
            r, c = within_batch_knn.adjacency(batch).nonzero()
            diag_edges[batch] = int(r.size)
            rows.append(r + offsets[batch])
            cols.append(c + offsets[batch])
            if directed_knn and knn_widths[(batch, batch)] > 0:
                knn_parts[(batch, batch)] = fixed[batch] + offsets[batch]

    knn_indices = None
    if directed_knn:
        knn_indices = _assemble_directed_knn(knn_parts, cross_only=rule == "cross_only")

    return BlockSelection(
        rows=np.concatenate(rows).astype(np.int64, copy=False),
        cols=np.concatenate(cols).astype(np.int64, copy=False),
        split=split,
        diag_edges=(diag_edges[0], diag_edges[1]),
        knn_indices=knn_indices,
    )


def _assemble_directed_knn(
    parts: dict[tuple[int, int], np.ndarray], cross_only: bool
) -> np.ndarray:
    """Assemble global directed neighbors with batch-0 rows before batch-1 rows.

    For split budgets, batch-0 neighbor columns precede batch-1 columns; zero quotas
    are omitted. Cross-only lists contain only opposite-batch neighbors.
    """
    if cross_only:
        return np.concatenate([parts[(0, 1)], parts[(1, 0)]], axis=0).astype(np.int32, copy=False)
    columns = [
        np.concatenate([parts[(0, cb)], parts[(1, cb)]], axis=0)
        for cb in (0, 1)
        if (0, cb) in parts
    ]
    if not columns:
        raise ValueError("effective_k resolved to zero neighbours in both batches.")
    return np.concatenate(columns, axis=1).astype(np.int32, copy=False)


# Graph


def graph_from_pairs(rows: np.ndarray, cols: np.ndarray, n_points: int) -> sp.csr_matrix:
    """Build a symmetric, loop-free CSR graph with unit total mass.

    Duplicate pairs are merged. Each stored directed entry has weight ``1 / |E|``,
    where ``|E|`` counts entries after symmetrization.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    flat = np.unique(np.concatenate([rows * n_points + cols, cols * n_points + rows]))
    if flat.size == 0:
        raise ValueError("Graph has zero total mass after normalization.")
    weight = 1.0 / float(flat.size)
    return sp.coo_matrix(
        (np.full(flat.size, weight), (flat // n_points, flat % n_points)),
        shape=(n_points, n_points),
    ).tocsr()
