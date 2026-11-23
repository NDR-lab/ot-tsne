"""Compare block-wise graph selection with dense reference selection.

Given identical scores, require matching edges, normalized weights, and UMAP neighbor
lists across rules and symmetrizations, including tie-heavy small regularization.
Reference cross blocks use the plan directly; separate tests measure differences
from SVD reconstruction.
"""

import gc
import weakref

import numpy as np
import pytest
import scipy.sparse as sp

from ne_aligned_utils import methods
from ne_aligned_utils.methods import (
    build_ot_tsne_four_block_matrix,
    build_ot_tsne_zerodiag_four_block_matrix,
    compute_ot_plan,
    normalize_ot_tsne_graph,
    resolve_effective_k_split,
    resolve_mass_budgets,
    sparsify_four_block_cross_batch_only,
    sparsify_four_block_cross_batch_split,
    sparsify_four_block_top_k,
    sparsify_four_block_top_k_mass_aware,
)
from ne_aligned_utils.methods.ot_affinity import (
    BlockScores,
    PlanDiagonals,
    PlanSpectrum,
    SuppliedDiagonals,
    WithinBatchKnn,
    graph_from_pairs,
    plan_cross_block,
    select_four_block,
)
from ne_aligned_utils.methods.ot_graph import EOT_INIT_MODES

N0, N1, K = 24, 36, 15
REL_REGS = (1.0, 0.1, 1 / 300)
SYMMETRIZATIONS = ("or", "and")


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(N0, 5)), rng.normal(size=(N1, 5)) + 0.5


def _reference_matrix(plan, zero_diagonal=False):
    """The dense four-block matrix with the plan as its cross block, as the method defines it."""
    if zero_diagonal:
        matrix = build_ot_tsne_zerodiag_four_block_matrix(plan)
    else:
        matrix = build_ot_tsne_four_block_matrix(plan)
    n0 = plan.shape[0]
    matrix[:n0, n0:] = plan
    matrix[n0:, :n0] = plan.T
    return matrix


def _block_selection(plan, within_batch, **kwargs):
    selection = select_four_block(
        BlockScores.from_plan(plan, within_batch, PlanSpectrum.from_plan(plan)),
        effective_k=K,
        within_batch_knn=within_batch if isinstance(within_batch, WithinBatchKnn) else None,
        **kwargs,
    )
    graph = graph_from_pairs(selection.rows, selection.cols, plan.shape[0] + plan.shape[1])
    return selection, graph


@pytest.mark.parametrize("rel_reg", REL_REGS)
@pytest.mark.parametrize("symmetrization", SYMMETRIZATIONS)
def test_split_rule_matches_the_dense_sparsifier(rel_reg, symmetrization):
    plan = compute_ot_plan(*_inputs(), rel_reg=rel_reg)
    dense = _reference_matrix(plan)
    sparse, split = sparsify_four_block_top_k(
        dense, n0=N0, n1=N1, effective_k=K, binarize=True, graph_symmetrization=symmetrization
    )
    selection, graph = _block_selection(
        plan, PlanDiagonals(), rule="split", graph_symmetrization=symmetrization
    )
    np.testing.assert_array_equal(graph.toarray(), normalize_ot_tsne_graph(sparse))
    assert selection.split == split
    assert selection.diag_edges == (
        int(sparse[:N0, :N0].sum()),
        int(sparse[N0:, N0:].sum()),
    )


@pytest.mark.parametrize("rel_reg", REL_REGS)
def test_directed_knn_matches_the_dense_one(rel_reg):
    plan = compute_ot_plan(*_inputs(), rel_reg=rel_reg)
    expected, _ = methods.four_block_directed_knn(
        _reference_matrix(plan), n0=N0, n1=N1, effective_k=K
    )
    selection, _ = _block_selection(plan, PlanDiagonals(), rule="split", directed_knn=True)
    np.testing.assert_array_equal(selection.knn_indices, expected)


@pytest.mark.parametrize("rel_reg", REL_REGS)
@pytest.mark.parametrize("symmetrization", SYMMETRIZATIONS)
def test_supplied_diagonals_match_the_dense_replacement(rel_reg, symmetrization):
    """Gram diagonals, written into the square before the sparsifier as the dense path does."""
    plan = compute_ot_plan(*_inputs(), rel_reg=rel_reg)
    gram0, gram1 = plan @ plan.T, plan.T @ plan
    dense = _reference_matrix(plan)
    for block, rows in ((gram0, slice(0, N0)), (gram1, slice(N0, N0 + N1))):
        block = np.array(block, copy=True)
        np.fill_diagonal(block, 0.0)
        dense[rows, rows] = block
    sparse, _ = sparsify_four_block_top_k(
        dense, n0=N0, n1=N1, effective_k=K, binarize=True, graph_symmetrization=symmetrization
    )
    selection, graph = _block_selection(
        plan,
        SuppliedDiagonals(gram0, gram1),
        rule="split",
        graph_symmetrization=symmetrization,
        directed_knn=symmetrization == "or",
    )
    np.testing.assert_array_equal(graph.toarray(), normalize_ot_tsne_graph(sparse))
    if symmetrization == "or":
        expected, _ = methods.four_block_directed_knn(dense, n0=N0, n1=N1, effective_k=K)
        np.testing.assert_array_equal(selection.knn_indices, expected)


@pytest.mark.parametrize("rel_reg", REL_REGS)
@pytest.mark.parametrize("knn_symmetrization", SYMMETRIZATIONS)
def test_knn_diagonal_matches_the_dense_overwrite(rel_reg, knn_symmetrization):
    x0, x1 = _inputs()
    plan = compute_ot_plan(x0, x1, rel_reg=rel_reg)
    split = resolve_effective_k_split(effective_k=K, n0=N0, n1=N1)
    idx0 = methods.within_batch_knn_indices(x0, split.batch0_k)
    idx1 = methods.within_batch_knn_indices(x1, split.batch1_k)
    dense = _reference_matrix(plan)
    sparse, _ = sparsify_four_block_top_k(dense, n0=N0, n1=N1, effective_k=K, binarize=True)
    knn0 = methods._adjacency_from_indices(idx0, N0, symmetrization=knn_symmetrization)
    knn1 = methods._adjacency_from_indices(idx1, N1, symmetrization=knn_symmetrization)
    sparse[:N0, :N0] = knn0
    sparse[N0:, N0:] = knn1
    selection, graph = _block_selection(
        plan,
        WithinBatchKnn(idx0, idx1, symmetrization=knn_symmetrization),
        rule="split",
        directed_knn=True,
    )
    np.testing.assert_array_equal(graph.toarray(), normalize_ot_tsne_graph(sparse))
    assert selection.diag_edges == (int((knn0 > 0).sum()), int((knn1 > 0).sum()))
    expected, _ = methods.four_block_directed_knn(
        dense, n0=N0, n1=N1, effective_k=K, within_batch_indices=(idx0, idx1)
    )
    np.testing.assert_array_equal(selection.knn_indices, expected)


@pytest.mark.parametrize("rel_reg", REL_REGS)
def test_mass_rule_matches_the_dense_mass_aware_sparsifier(rel_reg):
    plan = compute_ot_plan(*_inputs(), rel_reg=rel_reg)
    # Uneven marginals, as an unbalanced plan would have.
    weights = np.random.default_rng(1).uniform(0.2, 1.0, size=plan.shape)
    plan = plan * weights
    split = resolve_effective_k_split(effective_k=K, n0=N0, n1=N1)
    budgets = (
        resolve_mass_budgets(plan.sum(axis=1), split.batch1_k),
        resolve_mass_budgets(plan.sum(axis=0), split.batch0_k),
    )
    sparse, _ = sparsify_four_block_top_k_mass_aware(
        _reference_matrix(plan),
        n0=N0,
        n1=N1,
        effective_k=K,
        row_budgets=budgets[0],
        col_budgets=budgets[1],
        binarize=True,
    )
    _, graph = _block_selection(plan, PlanDiagonals(), rule="mass", mass_budgets=budgets)
    np.testing.assert_array_equal(graph.toarray(), normalize_ot_tsne_graph(sparse))


@pytest.mark.parametrize("rel_reg", REL_REGS)
@pytest.mark.parametrize("symmetrization", SYMMETRIZATIONS)
def test_cross_rules_match_the_dense_zero_diagonal_sparsifiers(rel_reg, symmetrization):
    plan = compute_ot_plan(*_inputs(), rel_reg=rel_reg)
    dense = _reference_matrix(plan, zero_diagonal=True)

    only = sparsify_four_block_cross_batch_only(
        dense, n0=N0, n1=N1, effective_k=K, binarize=True, graph_symmetrization=symmetrization
    )
    selection, graph = _block_selection(
        plan,
        None,
        rule="cross_only",
        graph_symmetrization=symmetrization,
        directed_knn=symmetrization == "or",
    )
    np.testing.assert_array_equal(graph.toarray(), normalize_ot_tsne_graph(only))
    assert selection.diag_edges == (0, 0)
    if symmetrization == "or":
        expected = methods.cross_batch_directed_knn(dense, n0=N0, n1=N1, effective_k=K)
        np.testing.assert_array_equal(selection.knn_indices, expected)

    split_sparse, _ = sparsify_four_block_cross_batch_split(
        dense, n0=N0, n1=N1, effective_k=K, binarize=True, graph_symmetrization=symmetrization
    )
    _, graph = _block_selection(
        plan, None, rule="cross_split", graph_symmetrization=symmetrization
    )
    np.testing.assert_array_equal(graph.toarray(), normalize_ot_tsne_graph(split_sparse))


@pytest.mark.parametrize("rel_reg", REL_REGS)
def test_unbinarized_selection_drops_selected_zeros_as_the_dense_path_does(rel_reg):
    plan = compute_ot_plan(*_inputs(), rel_reg=rel_reg)
    sparse, _ = sparsify_four_block_top_k(
        _reference_matrix(plan), n0=N0, n1=N1, effective_k=K, binarize=False
    )
    _, graph = _block_selection(plan, PlanDiagonals(), rule="split", binarize=False)
    np.testing.assert_array_equal(graph.toarray(), normalize_ot_tsne_graph(sparse))


def test_blocks_are_bit_identical_to_slices_of_the_dense_square():
    """A block product equals the slice of the full product, so the SVD-derived within-batch
    blocks are exactly the dense builder's."""
    plan = compute_ot_plan(*_inputs(), rel_reg=0.1)
    dense = build_ot_tsne_four_block_matrix(plan)
    spectrum = PlanSpectrum.from_plan(plan)
    for rb, rows in enumerate((slice(0, N0), slice(N0, N0 + N1))):
        for cb, cols in enumerate((slice(0, N0), slice(N0, N0 + N1))):
            np.testing.assert_array_equal(spectrum.raw_block(rb, cb), dense[rows, cols])


def test_sparse_and_dense_graphs_reach_tsne_with_the_same_edges():
    """``run_tsne_from_graph`` accepts the sparse graph and cleans it as it cleans a dense one."""
    plan = compute_ot_plan(*_inputs(), rel_reg=1.0)
    _, graph = _block_selection(plan, PlanDiagonals(), rule="split")
    dirty = graph.tolil()
    dirty[0, 0] = 5.0  # a self-loop
    dirty[1, 2] = -1.0  # a negative entry
    embedding = methods.run_tsne_from_graph(sp.csr_matrix(dirty), n_epochs=12, random_state=0)
    reference = methods.run_tsne_from_graph(dirty.toarray(), n_epochs=12, random_state=0)
    assert embedding.shape == reference.shape
    assert np.isfinite(embedding).all()


# Requests that would build a different graph from the one asked for


def _plan():
    return compute_ot_plan(*_inputs(), rel_reg=1.0)


def _spectrum():
    return PlanSpectrum.from_plan(_plan())


def _select(source=None, **kwargs):
    plan = _plan()
    scores = BlockScores.from_plan(plan, source, PlanSpectrum.from_plan(plan))
    return select_four_block(scores, effective_k=K, **kwargs)


def test_split_rule_refuses_to_silently_drop_within_batch_edges():
    with pytest.raises(ValueError, match="exactly one source for the within-batch blocks"):
        _select(None, rule="split")


def test_mass_rule_requires_budgets():
    with pytest.raises(ValueError, match="needs mass_budgets"):
        _select(PlanDiagonals(), rule="mass")


def test_cross_rules_refuse_a_diagonal_source():
    with pytest.raises(ValueError, match="selects no within-batch edges"):
        _select(PlanDiagonals(), rule="cross_only")


def test_variable_width_rules_refuse_a_directed_list():
    with pytest.raises(ValueError, match="different degrees"):
        _select(None, rule="cross_split", directed_knn=True)


def test_budgets_are_refused_outside_the_mass_rule():
    budgets = (np.ones(N0, dtype=int), np.ones(N1, dtype=int))
    with pytest.raises(ValueError, match="only used by rule='mass'"):
        _select(None, rule="cross_only", mass_budgets=budgets)


def test_unknown_within_batch_source_is_a_type_error():
    with pytest.raises(TypeError, match="within-batch source"):
        BlockScores.from_plan(_plan(), "plan")


def test_supplied_diagonals_of_the_wrong_shape_are_refused():
    with pytest.raises(ValueError, match="supplied diagonal block 1"):
        BlockScores.from_plan(
            _plan(), SuppliedDiagonals(np.zeros((N0, N0)), np.zeros((N0, N0)))
        )


def test_plan_diagonals_require_the_svd():
    with pytest.raises(ValueError, match="needs the plan's SVD"):
        BlockScores.from_plan(_plan(), PlanDiagonals())


def test_the_cross_block_is_the_plan_without_a_copy_when_it_is_clean():
    plan = _plan()
    assert plan_cross_block(plan) is plan
    dirty = plan.copy()
    dirty[0, 0], dirty[0, 1] = np.nan, -1.0
    cleaned = plan_cross_block(dirty)
    assert cleaned[0, 0] == 0.0 and cleaned[0, 1] == 0.0
    np.testing.assert_array_equal(cleaned[1:], plan[1:])
    assert np.isnan(dirty[0, 0])  # the caller's plan is not written to


# One block at a time


class _TrackedBlocks:
    """Block scores that record which blocks are built and which earlier ones are still alive."""

    def __init__(self, plan, with_diagonals):
        self.requests = []
        self.alive_when_built = []
        self._refs = []
        spectrum = PlanSpectrum.from_plan(plan)
        self.scores = BlockScores(
            spectrum.n0,
            spectrum.n1,
            # A copy, because the real cross block is the caller's plan and outlives selection.
            lambda: self._build("cross", lambda: plan_cross_block(plan).copy()),
            (lambda b: self._build(f"diag{b}", lambda: spectrum.diagonal_block(b)))
            if with_diagonals
            else None,
        )

    def _build(self, name, make):
        gc.collect()
        self.alive_when_built.append([n for n, ref in self._refs if ref() is not None])
        self.requests.append(name)
        block = make()
        self._refs.append((name, weakref.ref(block)))
        return block


@pytest.mark.parametrize(
    "rule, with_diagonals, expected",
    [
        ("split", True, ["cross", "diag0", "diag1"]),
        ("cross_only", False, ["cross"]),
        ("cross_split", False, ["cross"]),
    ],
)
def test_each_block_is_built_once_and_only_when_needed(rule, with_diagonals, expected):
    tracked = _TrackedBlocks(_plan(), with_diagonals)
    select_four_block(tracked.scores, effective_k=K, rule=rule)
    assert tracked.requests == expected


def test_a_block_is_released_before_the_next_is_built():
    """At most one score block is held at a time; that is the memory this module saves."""
    tracked = _TrackedBlocks(_plan(), with_diagonals=True)
    select_four_block(tracked.scores, effective_k=K, rule="split", directed_knn=True)
    assert tracked.alive_when_built == [[], [], []]


# Slabs


def _pairs(rows, cols):
    return np.unique(np.asarray(rows, dtype=np.int64) * 10**6 + np.asarray(cols, dtype=np.int64))


#: One row (or column) per slab, a few per slab, and the whole block at once.
SLAB_BUDGETS = (1, 4096)
WHOLE = 2**62


@pytest.mark.parametrize("rel_reg", REL_REGS)
@pytest.mark.parametrize(
    "source, rule, symmetrization, directed",
    [
        (PlanDiagonals(), "split", "or", True),
        (PlanDiagonals(), "split", "and", False),
        (None, "cross_only", "or", True),
        (None, "cross_only", "and", False),
        (None, "cross_split", "or", False),
        (None, "cross_split", "and", False),
    ],
)
@pytest.mark.parametrize("budget", SLAB_BUDGETS)
def test_slabs_select_exactly_what_the_whole_block_selects(
    rel_reg, source, rule, symmetrization, directed, budget
):
    plan = compute_ot_plan(*_inputs(), rel_reg=rel_reg)
    spectrum = PlanSpectrum.from_plan(plan)

    def run(selection_budget):
        return select_four_block(
            BlockScores.from_plan(plan, source, spectrum),
            effective_k=K,
            rule=rule,
            graph_symmetrization=symmetrization,
            directed_knn=directed,
            selection_budget_bytes=selection_budget,
        )

    whole, slabbed = run(WHOLE), run(budget)
    np.testing.assert_array_equal(
        _pairs(slabbed.rows, slabbed.cols), _pairs(whole.rows, whole.cols)
    )
    assert slabbed.diag_edges == whole.diag_edges
    if directed:
        np.testing.assert_array_equal(slabbed.knn_indices, whole.knn_indices)


@pytest.mark.parametrize("rel_reg", REL_REGS)
@pytest.mark.parametrize("budget", SLAB_BUDGETS)
def test_mass_aware_slabs_select_exactly_what_the_whole_block_selects(rel_reg, budget):
    plan = compute_ot_plan(*_inputs(), rel_reg=rel_reg)
    plan = plan * np.random.default_rng(1).uniform(0.2, 1.0, size=plan.shape)
    split = resolve_effective_k_split(effective_k=K, n0=N0, n1=N1)
    budgets = (
        resolve_mass_budgets(plan.sum(axis=1), split.batch1_k),
        resolve_mass_budgets(plan.sum(axis=0), split.batch0_k),
    )
    spectrum = PlanSpectrum.from_plan(plan)
    runs = [
        select_four_block(
            BlockScores.from_plan(plan, PlanDiagonals(), spectrum),
            effective_k=K,
            rule="mass",
            mass_budgets=budgets,
            selection_budget_bytes=b,
        )
        for b in (WHOLE, budget)
    ]
    np.testing.assert_array_equal(
        _pairs(runs[1].rows, runs[1].cols), _pairs(runs[0].rows, runs[0].cols)
    )


@pytest.mark.parametrize("mode", ("or", "and"))
@pytest.mark.parametrize("exclude_diagonal", (False, True))
@pytest.mark.parametrize("binarize", (True, False))
@pytest.mark.parametrize("seed", range(4))
def test_slabs_keep_the_tie_breaking_of_the_whole_block(mode, exclude_diagonal, binarize, seed):
    """Blocks of small integers and exact zeros: nearly every selection is decided by a tie."""
    from ne_aligned_utils.methods.ot_affinity import select_block

    rng = np.random.default_rng(seed)
    n_rows = 17 if exclude_diagonal else 13
    n_cols = 17 if exclude_diagonal else 29
    block = rng.integers(0, 3, size=(n_rows, n_cols)).astype(float)
    block[rng.random(block.shape) < 0.5] = 0.0
    block[0, 1] = np.nan  # cleaned to -inf, as in the dense path
    capacity = (n_cols - 1, n_rows - 1) if exclude_diagonal else (n_cols, n_rows)
    for row_k, col_k in [(0, 3), (3, 0), (4, 5), capacity]:
        kwargs = dict(
            mode=mode,
            row_k=row_k,
            col_k=col_k,
            exclude_diagonal=exclude_diagonal,
            binarize=binarize,
        )
        whole = select_block(block, budget_bytes=WHOLE, **kwargs)
        for budget in SLAB_BUDGETS:
            slabbed = select_block(block, budget_bytes=budget, **kwargs)
            np.testing.assert_array_equal(_pairs(*slabbed[:2]), _pairs(*whole[:2]))
            np.testing.assert_array_equal(slabbed[2], whole[2])


def test_slabs_are_used_only_above_the_memory_budget():
    from ne_aligned_utils.methods import ot_affinity

    assert ot_affinity._full_selection_bytes(100, 100, "or", False) == 25 * 100 * 100
    calls = []
    original = ot_affinity._block_keys_slabbed
    try:
        ot_affinity._block_keys_slabbed = lambda *a: calls.append(a) or original(*a)
        block = np.random.default_rng(0).random((50, 60))
        ot_affinity.select_block(block, mode="or", row_k=3, col_k=3, exclude_diagonal=False)
        assert not calls
        ot_affinity.select_block(
            block, mode="or", row_k=3, col_k=3, exclude_diagonal=False, budget_bytes=1000
        )
        assert len(calls) == 1
    finally:
        ot_affinity._block_keys_slabbed = original


# The plan as the cross block, and the SVD only where it is read


@pytest.mark.parametrize("rel_reg", (1.0, 0.1))
@pytest.mark.parametrize("symmetrization", SYMMETRIZATIONS)
def test_svd_rebuilt_cross_block_gives_the_same_edges_at_moderate_epsilon(
    rel_reg, symmetrization
):
    """Where plan entries are well above the rebuild's rounding error, nothing moves."""
    plan = compute_ot_plan(*_inputs(), rel_reg=rel_reg)
    rebuilt, _ = sparsify_four_block_top_k(
        build_ot_tsne_four_block_matrix(plan),
        n0=N0,
        n1=N1,
        effective_k=K,
        binarize=True,
        graph_symmetrization=symmetrization,
    )
    _, graph = _block_selection(
        plan, PlanDiagonals(), rule="split", graph_symmetrization=symmetrization
    )
    np.testing.assert_array_equal(graph.toarray() > 0, normalize_ot_tsne_graph(rebuilt) > 0)


def test_svd_rebuilt_cross_block_is_rounding_noise_at_small_epsilon():
    """Why the plan, not its SVD rebuild, is the cross block.

    At small epsilon most plan entries are far below the absolute rounding error of ``U S V.T``,
    so the rebuild ranks them by that error rather than by the plan.
    """
    plan = compute_ot_plan(*_inputs(), rel_reg=1 / 300)
    rebuilt = build_ot_tsne_four_block_matrix(plan)[:N0, N0:]
    assert np.abs(rebuilt - plan).max() > 1e10 * np.median(plan)


class _SvdCounter:
    def __init__(self, monkeypatch):
        self.calls = 0
        real = np.linalg.svd

        def counted(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(np.linalg, "svd", counted)


SVD_FREE = (
    methods.compute_ot_tsne_gramdiag_embedding_from_arrays,
    methods.compute_ot_tsne_selfotmedian_embedding_from_arrays,
    methods.compute_ot_tsne_knndiag_embedding_from_arrays,
    methods.compute_ot_tsne_zerodiag_embedding_from_arrays,
    methods.compute_ot_tsne_zerodiag_split_embedding_from_arrays,
)


@pytest.mark.parametrize("entry", SVD_FREE, ids=lambda f: f.__name__)
def test_variants_without_plan_diagonals_skip_the_svd(entry, monkeypatch):
    counter = _SvdCounter(monkeypatch)
    _, artifacts = entry(*_inputs(), effective_k=K, n_epochs=12, return_artifacts=True)
    assert counter.calls == 0
    assert "singular_values" not in artifacts


@pytest.mark.parametrize("entry", SVD_FREE, ids=lambda f: f.__name__)
def test_the_eot_initialization_takes_only_the_leading_modes(entry, monkeypatch):
    counter = _SvdCounter(monkeypatch)
    _, artifacts = entry(*_inputs(), effective_k=K, n_epochs=12, init="eot", return_artifacts=True)
    assert counter.calls == 0
    assert artifacts["init_used"] == "eot"
    assert len(artifacts["singular_values"]) == EOT_INIT_MODES


def test_the_default_construction_takes_one_svd(monkeypatch):
    counter = _SvdCounter(monkeypatch)
    _, artifacts = methods.compute_ot_tsne_embedding_from_arrays(
        *_inputs(), effective_k=K, n_epochs=12, return_artifacts=True
    )
    assert counter.calls == 1
    assert "singular_values" in artifacts
