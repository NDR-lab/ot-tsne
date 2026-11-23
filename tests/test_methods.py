"""Check embedding methods, affinity construction, and backend equivalence."""

import inspect
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
from openTSNE import TSNE
from openTSNE.affinity import Affinities
from sklearn.decomposition import PCA

from ne_aligned_utils import methods
from ne_aligned_utils.config import DEFAULT_SCVI_BATCH_SIZE, DEFAULT_SCVI_MAX_EPOCHS
from ne_aligned_utils.methods import (
    OT_TSNE_INIT_CHOICES,
    _build_four_block_parts,
    _finalize_tsne_init,
    _is_symmetric,
    _resolve_plan_init,
    _spectral_init,
    _within_batch_knn_block,
    build_ot_tsne_four_block_matrix,
    build_ot_tsne_zerodiag_four_block_matrix,
    compute_cca_embedding_from_arrays,
    compute_cca_then_tsne_embedding_from_arrays,
    compute_eot_embedding_from_arrays,
    compute_eot_embedding_from_plan,
    compute_eot_then_tsne_embedding_from_arrays,
    compute_ot_plan,
    compute_ot_tsne_diagonal_embedding_from_plan,
    compute_ot_tsne_embedding_from_arrays,
    compute_ot_tsne_gramdiag_embedding_from_arrays,
    compute_ot_tsne_knndiag_embedding_from_arrays,
    compute_ot_tsne_knndiag_embedding_from_plan,
    compute_ot_tsne_uot_mass_embedding_from_arrays,
    compute_ot_tsne_uot_topk_embedding_from_arrays,
    compute_ot_tsne_zerodiag_embedding_from_arrays,
    compute_ot_tsne_zerodiag_split_embedding_from_arrays,
    compute_pca_embedding_from_arrays,
    compute_policar_ref_tsne_embedding_from_arrays,
    compute_ref_tsne_embedding_from_arrays,
    compute_scanorama_embedding_from_arrays,
    compute_scanorama_then_tsne_embedding_from_arrays,
    compute_tsne_embedding_from_arrays,
    normalize_ot_tsne_graph,
    resolve_cross_batch_k,
    resolve_effective_k_split,
    resolve_mass_budgets,
    run_tsne_from_graph,
    sparsify_four_block_cross_batch_only,
    sparsify_four_block_cross_batch_split,
    sparsify_four_block_top_k,
    sparsify_four_block_top_k_mass_aware,
)


def _align_svd_column_signs(actual, expected):
    aligned = np.array(actual, copy=True)
    for col in range(expected.shape[1]):
        pivot = int(np.argmax(np.abs(expected[:, col])))
        if aligned[pivot, col] * expected[pivot, col] < 0:
            aligned[:, col] *= -1.0
    return aligned


def test_compute_pca_embedding_from_arrays_matches_sklearn_joint_pca():
    x0 = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.2, 0.5],
            [0.1, 1.0, 0.2],
        ]
    )
    x1 = np.array(
        [
            [0.2, 0.1, 0.9],
            [1.1, 0.0, 0.4],
            [0.0, 1.2, 0.1],
            [0.8, 0.9, 0.3],
        ]
    )

    embedding, artifacts = compute_pca_embedding_from_arrays(
        x0,
        x1,
        embd_dim=2,
        random_state=0,
        return_artifacts=True,
    )

    expected = PCA(n_components=2, random_state=0).fit_transform(np.concatenate([x0, x1], axis=0))
    np.testing.assert_allclose(embedding, expected)
    np.testing.assert_allclose(artifacts["left_embedding"], expected[: x0.shape[0]])
    np.testing.assert_allclose(artifacts["right_embedding"], expected[x0.shape[0] :])
    assert artifacts["components"].shape == (2, x0.shape[1])
    assert artifacts["explained_variance"].shape == (2,)


def test_compute_tsne_embedding_from_arrays_smoke():
    x0 = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.2, 0.5],
            [0.1, 1.0, 0.2],
            [0.8, 0.8, 0.3],
        ]
    )
    x1 = np.array(
        [
            [0.2, 0.1, 0.9],
            [1.1, 0.0, 0.4],
            [0.0, 1.2, 0.1],
            [0.8, 0.9, 0.3],
        ]
    )

    embedding, artifacts = compute_tsne_embedding_from_arrays(
        x0,
        x1,
        random_state=0,
        perplexity=2.0,
        pca_components=2,
        n_iter=12,
        early_exaggeration_iter=4,
        return_artifacts=True,
    )

    assert embedding.shape == (8, 2)
    assert artifacts["left_embedding"].shape == (4, 2)
    assert artifacts["right_embedding"].shape == (4, 2)
    assert artifacts["pca_components"] == 2
    assert artifacts["perplexity"] == 2.0
    assert np.isfinite(embedding).all()


def test_compute_cca_embedding_from_arrays_matches_cross_product_svd():
    x0 = np.array(
        [
            [2.0, 0.0, 1.0],
            [1.0, -1.0, 0.5],
            [0.0, 1.5, 2.0],
        ]
    )
    x1 = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.5, 2.0, 1.0],
            [2.0, 0.0, 1.0],
            [1.0, -1.0, 1.5],
        ]
    )

    embedding, artifacts = compute_cca_embedding_from_arrays(
        x0,
        x1,
        embd_dim=2,
        return_artifacts=True,
    )

    # Centring is the default, so the reference is the cross-*covariance*: each batch on its own
    # feature means. Comparing against the raw product would pin the pre-fix behavior.
    c0 = x0 - x0.mean(axis=0, keepdims=True)
    c1 = x1 - x1.mean(axis=0, keepdims=True)
    u, singular_values, vt = np.linalg.svd(c0 @ c1.T, full_matrices=False)
    expected = np.concatenate([u[:, :2], vt.T[:, :2]], axis=0)

    np.testing.assert_allclose(_align_svd_column_signs(embedding, expected), expected)
    np.testing.assert_allclose(
        _align_svd_column_signs(artifacts["left_embedding"], u[:, :2]),
        u[:, :2],
    )
    np.testing.assert_allclose(
        _align_svd_column_signs(artifacts["right_embedding"], vt.T[:, :2]),
        vt.T[:, :2],
    )
    np.testing.assert_allclose(artifacts["singular_values"], singular_values)
    assert artifacts["embd_dim"] == 2
    assert artifacts["embd_dim_requested"] == 2
    assert artifacts["centered"] is True


def test_compute_cca_embedding_from_arrays_uncentered_keeps_the_raw_cross_product():
    """``center=False`` must still reproduce results published before centering was added.

    This is the whole point of keeping the option: the ``cca-2d-uncentered`` and
    ``cca-50-uncentered`` task-kinds exist so an old number can be recomputed, which is only
    true while this path stays byte-for-byte the old computation.
    """
    x0 = np.array([[2.0, 0.0, 1.0], [1.0, -1.0, 0.5], [0.0, 1.5, 2.0]])
    x1 = np.array([[1.0, 1.0, 0.0], [0.5, 2.0, 1.0], [2.0, 0.0, 1.0], [1.0, -1.0, 1.5]])

    embedding, artifacts = compute_cca_embedding_from_arrays(
        x0, x1, embd_dim=2, return_artifacts=True, center=False
    )

    u, singular_values, vt = np.linalg.svd(x0 @ x1.T, full_matrices=False)
    expected = np.concatenate([u[:, :2], vt.T[:, :2]], axis=0)

    np.testing.assert_allclose(_align_svd_column_signs(embedding, expected), expected)
    np.testing.assert_allclose(artifacts["singular_values"], singular_values)
    assert artifacts["centered"] is False


def test_compute_cca_embedding_centering_changes_the_result():
    """Guards against ``center`` being silently ignored, which both cases above would survive."""
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(12, 5)) + 10.0  # a large shared offset is what centering removes
    x1 = rng.normal(size=(9, 5)) + 10.0

    centered = compute_cca_embedding_from_arrays(x0, x1, embd_dim=2)
    uncentered = compute_cca_embedding_from_arrays(x0, x1, embd_dim=2, center=False)

    assert centered.shape == uncentered.shape
    assert not np.allclose(np.abs(centered), np.abs(uncentered))


def test_compute_cca_embedding_from_arrays_clips_dimension_to_available_rank():
    x0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    x1 = np.array([[1.0, 1.0], [1.0, -1.0], [0.5, 0.25]])

    embedding, artifacts = compute_cca_embedding_from_arrays(
        x0,
        x1,
        embd_dim=10,
        return_artifacts=True,
    )

    assert embedding.shape == (x0.shape[0] + x1.shape[0], min(x0.shape[0], x1.shape[0]))
    assert artifacts["embd_dim"] == min(x0.shape[0], x1.shape[0])
    assert artifacts["embd_dim_requested"] == 10


def test_compute_cca_embedding_from_arrays_validates_inputs():
    with pytest.raises(ValueError, match="2D arrays"):
        compute_cca_embedding_from_arrays(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError, match="same number of features"):
        compute_cca_embedding_from_arrays(np.zeros((2, 2)), np.zeros((3, 3)))

    with pytest.raises(ValueError, match="positive"):
        compute_cca_embedding_from_arrays(np.zeros((2, 2)), np.zeros((3, 2)), embd_dim=0)


def test_compute_cca_then_tsne_embedding_from_arrays_smoke():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(20, 6))
    x1 = rng.normal(size=(30, 6)) + 0.5

    embedding, artifacts = compute_cca_then_tsne_embedding_from_arrays(
        x0,
        x1,
        cca_dim=8,
        random_state=0,
        perplexity=5.0,
        n_iter=12,
        return_artifacts=True,
    )

    assert embedding.shape == (50, 2)
    assert np.isfinite(embedding).all()
    assert artifacts["left_embedding"].shape == (20, 2)
    assert artifacts["right_embedding"].shape == (30, 2)
    assert artifacts["cca_dim"] == 8


def test_compute_ref_tsne_projects_query_into_reference_and_is_deterministic():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(40, 8))
    x1 = rng.normal(size=(25, 8)) + 0.5

    first, artifacts = compute_ref_tsne_embedding_from_arrays(
        x0, x1, random_state=3, perplexity=6.0, n_iter=15, return_artifacts=True
    )
    second = compute_ref_tsne_embedding_from_arrays(
        x0, x1, random_state=3, perplexity=6.0, n_iter=15
    )

    assert first.shape == (65, 2)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)  # deterministic given seed
    # reference rows are the frozen batch-0 embedding; query is projected into it
    assert artifacts["left_embedding"].shape == (40, 2)
    assert artifacts["right_embedding"].shape == (25, 2)
    np.testing.assert_array_equal(first[:40], artifacts["left_embedding"])


def test_compute_policar_ref_tsne_uses_supplied_feature_panel():
    rng = np.random.default_rng(4)
    x0 = rng.poisson(3, size=(35, 8)).astype(float)
    x1 = rng.poisson(3, size=(20, 8)).astype(float)

    embedding, artifacts = compute_policar_ref_tsne_embedding_from_arrays(
        x0,
        x1,
        random_state=3,
        n_jobs=1,
        preselected_features=True,
        return_artifacts=True,
    )

    assert embedding.shape == (55, 2)
    assert np.isfinite(embedding).all()
    assert artifacts["preselected_features"] is True
    assert artifacts["reference_gene_count"] == 8
    assert artifacts["mapping_gene_count"] == 8
    assert artifacts["mapping_initialization"] == "median"
    assert artifacts["mapping_k"] == 10
    assert artifacts["initialization_random_state"] == 3


def test_compute_policar_ref_tsne_drops_reference_constant_genes_from_both_panels():
    rng = np.random.default_rng(41)
    x0 = rng.poisson(3, size=(35, 8)).astype(float)
    x1 = rng.poisson(3, size=(20, 8)).astype(float)
    x0[:, 0] = 0

    embedding, artifacts = compute_policar_ref_tsne_embedding_from_arrays(
        x0,
        x1,
        random_state=3,
        n_jobs=1,
        preselected_features=True,
        return_artifacts=True,
    )

    assert embedding.shape == (55, 2)
    assert artifacts["reference_gene_count"] == 7
    assert artifacts["mapping_gene_count"] == 7
    assert artifacts["dropped_zero_variance_gene_count"] == 1


def test_compute_scanorama_embedding_from_arrays_smoke():
    pytest.importorskip("scanorama")
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=(30, 12))
    x1 = rng.normal(size=(35, 12)) + 0.4

    embedding, artifacts = compute_scanorama_embedding_from_arrays(
        x0, x1, embd_dim=2, random_state=0, return_artifacts=True
    )

    assert embedding.shape == (65, 2)
    assert np.isfinite(embedding).all()
    assert artifacts["left_embedding"].shape == (30, 2)
    assert artifacts["right_embedding"].shape == (35, 2)


def test_harmony_receives_the_benchmark_seed(monkeypatch):
    received = {}

    def fake_run_harmony(embedding, metadata, batch_key, **kwargs):
        received.update(kwargs)
        return SimpleNamespace(Z_corr=np.asarray(embedding).T)

    monkeypatch.setitem(sys.modules, "harmonypy", SimpleNamespace(run_harmony=fake_run_harmony))
    embedding = np.arange(24, dtype=float).reshape(8, 3)
    corrected = methods._run_harmony(
        embedding,
        np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        "batch",
        random_state=7,
        harmony_kwargs={"verbose": False},
    )

    np.testing.assert_array_equal(corrected, embedding)
    assert received["random_state"] == 7


def test_harmony_rejects_a_second_seed_source():
    with pytest.raises(ValueError, match="through random_state"):
        methods._run_harmony(
            np.ones((8, 3)),
            np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            "batch",
            random_state=7,
            harmony_kwargs={"random_state": 3},
        )


def test_harmony_uses_precomputed_joint_pca_without_refitting(monkeypatch):
    from ne_aligned_utils.methods import general_integration_methods as harmony_module

    x0 = np.arange(150, dtype=float).reshape(3, 50)
    x1 = np.arange(200, dtype=float).reshape(4, 50)

    def reject_pca(*args, **kwargs):
        raise AssertionError("Harmony refitted PCA instead of using the preprocessing PCA")

    monkeypatch.setattr(harmony_module, "compute_pca_embedding", reject_pca)
    monkeypatch.setattr(
        harmony_module,
        "_run_harmony",
        lambda embedding, *args, **kwargs: np.asarray(embedding),
    )

    corrected, artifacts = methods.compute_harmony_embedding_from_arrays(
        x0,
        x1,
        embd_dim=50,
        input_is_pca=True,
        return_artifacts=True,
    )

    np.testing.assert_array_equal(corrected, np.concatenate([x0, x1]))
    assert artifacts["input_is_pca"] is True
    assert artifacts["harmony_dim_used"] == 50


def test_scanorama_actually_integrates_rather_than_returning_a_joint_svd():
    """Require Scanorama to align batches rather than return uncorrected joint SVD coordinates."""
    pytest.importorskip("scanorama")
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(0)
    shared = rng.normal(size=(60, 12))
    # one shared structure, shifted per batch -- integration should undo the shift
    x0 = shared + rng.normal(scale=0.1, size=(60, 12))
    x1 = shared + rng.normal(scale=0.1, size=(60, 12)) + 5.0

    embedding = compute_scanorama_embedding_from_arrays(x0, x1, embd_dim=8, random_state=0)

    batches = np.concatenate([np.zeros(len(x0), int), np.ones(len(x1), int)])
    neighbours = (
        NearestNeighbors(n_neighbors=11)
        .fit(embedding)
        .kneighbors(embedding, return_distance=False)[:, 1:]
    )
    cross_batch = float((batches[neighbours] != batches[:, None]).mean())
    # 0.5 is perfect mixing; the unintegrated joint SVD of this pair sits near 0.0
    assert cross_batch > 0.25, (
        f"only {cross_batch:.3f} of neighbours cross the batch boundary, so Scanorama did not "
        "integrate -- check that approx=False still reaches scanorama.integrate"
    )


def test_compute_scanorama_then_tsne_embedding_from_arrays_smoke():
    pytest.importorskip("scanorama")
    rng = np.random.default_rng(2)
    x0 = rng.normal(size=(30, 12))
    x1 = rng.normal(size=(35, 12)) + 0.4

    embedding, artifacts = compute_scanorama_then_tsne_embedding_from_arrays(
        x0, x1, scanorama_dim=8, random_state=0, perplexity=5.0, n_iter=12, return_artifacts=True
    )

    assert embedding.shape == (65, 2)
    assert np.isfinite(embedding).all()
    assert artifacts["left_embedding"].shape == (30, 2)
    assert artifacts["right_embedding"].shape == (35, 2)
    assert artifacts["scanorama_dim"] == 8


SWEEP_GRID_RECIPROCALS = (1, 10, 100, 300)


@pytest.mark.parametrize("divisor", SWEEP_GRID_RECIPROCALS)
def test_epsilon_is_rel_reg_times_the_median(divisor):
    """The canonical definition, pinned: ``epsilon = rel_reg * median_squared_distance``."""
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(30, 5))
    x1 = rng.normal(size=(24, 5)) + 0.3

    _, artifacts = compute_ot_plan(x0, x1, rel_reg=1 / divisor, return_artifacts=True)

    median = float(artifacts["median_squared_distance"])
    assert float(artifacts["epsilon"]) == median * (1 / divisor)


@pytest.mark.parametrize("divisor", SWEEP_GRID_RECIPROCALS)
def test_the_legacy_epsilon_is_reachable_through_the_explicit_argument(divisor):
    """The historical rounding is reproducible, but only deliberately and never by default.

    ``median * rel_reg`` and the old ``median / eps_times`` are the same real number and not always
    the same double. Canonical code uses the multiplication; anything that needs to reproduce the
    eps_times tree bit-for-bit recomputes the old value and passes it in as ``epsilon=``. This pins
    that the door exists and that the two really do differ, so the distinction cannot rot.
    """
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(30, 5))
    x1 = rng.normal(size=(24, 5)) + 0.3

    _, canonical = compute_ot_plan(x0, x1, rel_reg=1 / divisor, return_artifacts=True)
    median = float(canonical["median_squared_distance"])

    _, legacy = compute_ot_plan(x0, x1, epsilon=median / divisor, return_artifacts=True)
    assert float(legacy["epsilon"]) == median / divisor

    # And the reciprocation itself is exact on the sweep grid, which is why the legacy path can
    # recover the old divisor from a stored rel_reg at all.
    assert 1.0 / (1.0 / divisor) == float(divisor)


def test_compute_ot_plan_returns_artifacts_and_respects_explicit_epsilon():
    x0 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x1 = np.array([[0.1, 0.0], [1.0, 0.2], [0.2, 1.0], [0.8, 0.8]])

    plan, artifacts = compute_ot_plan(
        x0,
        x1,
        epsilon=0.5,
        return_artifacts=True,
    )

    assert plan.shape == (3, 4)
    assert artifacts["cost_matrix"].shape == (3, 4)
    assert artifacts["epsilon"] == 0.5
    assert artifacts["median_squared_distance"] > 0.0
    np.testing.assert_allclose(plan.sum(axis=1), np.full(3, 1.0 / 3.0), atol=1e-6)
    np.testing.assert_allclose(plan.sum(axis=0), np.full(4, 1.0 / 4.0), atol=1e-6)


def test_compute_eot_embedding_from_plan_matches_t0_svd():
    plan = np.array(
        [
            [0.12, 0.03, 0.05, 0.10],
            [0.04, 0.15, 0.08, 0.02],
            [0.09, 0.01, 0.06, 0.25],
        ],
        dtype=float,
    )
    plan = plan / plan.sum()

    embedding, artifacts = compute_eot_embedding_from_plan(
        plan,
        t=0,
        embd_dim=2,
        return_artifacts=True,
    )

    u, singular_values, vt = np.linalg.svd(plan, full_matrices=False)
    expected_left = u[:, 1:3] * np.sqrt(plan.shape[0])
    expected_right = vt.T[:, 1:3] * np.sqrt(plan.shape[1])
    expected = np.concatenate([expected_left, expected_right], axis=0)

    np.testing.assert_allclose(_align_svd_column_signs(embedding, expected), expected)
    np.testing.assert_allclose(
        _align_svd_column_signs(artifacts["left_embedding"], expected_left),
        expected_left,
    )
    np.testing.assert_allclose(
        _align_svd_column_signs(artifacts["right_embedding"], expected_right),
        expected_right,
    )
    np.testing.assert_allclose(artifacts["singular_values"], singular_values)
    assert artifacts["embd_dim"] == 2


def test_compute_eot_embedding_from_plan_applies_singular_value_power():
    plan = np.array(
        [
            [0.20, 0.05, 0.10],
            [0.03, 0.25, 0.12],
            [0.06, 0.04, 0.15],
        ],
        dtype=float,
    )
    plan = plan / plan.sum()

    embedding = compute_eot_embedding_from_plan(plan, t=1, embd_dim=2)

    u, singular_values, vt = np.linalg.svd(plan, full_matrices=False)
    weights = singular_values[1:3]
    expected_left = u[:, 1:3] * weights * np.sqrt(plan.shape[0])
    expected_right = vt.T[:, 1:3] * weights * np.sqrt(plan.shape[1])
    expected = np.concatenate([expected_left, expected_right], axis=0)

    np.testing.assert_allclose(_align_svd_column_signs(embedding, expected), expected)


@pytest.mark.parametrize("embd_dim, t", [(2, 0.0), (10, 0.0), (10, 1.0)])
def test_compute_eot_embedding_truncated_solve_matches_full_svd(embd_dim, t):
    """A plan large enough for the ARPACK path gives the full-SVD EOT coordinates, up to
    rounding and column signs, while computing only the leading modes."""
    rng = np.random.default_rng(0)
    plan = compute_ot_plan(rng.normal(size=(80, 5)), rng.normal(size=(60, 5)) + 0.3)
    n0, n1 = plan.shape

    embedding, artifacts = compute_eot_embedding_from_plan(
        plan, t=t, embd_dim=embd_dim, return_artifacts=True
    )

    u, singular_values, vt = np.linalg.svd(plan, full_matrices=False)
    stop = 1 + embd_dim
    weights = singular_values[1:stop] ** t
    expected = np.concatenate(
        [u[:, 1:stop] * weights * np.sqrt(n0), vt.T[:, 1:stop] * weights * np.sqrt(n1)], axis=0
    )
    assert embedding.shape == (n0 + n1, embd_dim)
    np.testing.assert_allclose(
        _align_svd_column_signs(embedding, expected), expected, rtol=1e-7, atol=1e-9
    )
    np.testing.assert_allclose(artifacts["singular_values"], singular_values[:stop], rtol=1e-10)
    assert artifacts["u"].shape == (n0, stop)
    assert artifacts["vt"].shape == (stop, n1)


def test_compute_eot_embedding_from_arrays_smoke():
    x0 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.2],
            [0.1, 1.0],
        ]
    )
    x1 = np.array(
        [
            [0.2, 0.1],
            [1.1, 0.0],
            [0.0, 1.2],
            [0.8, 0.9],
        ]
    )

    embedding, artifacts = compute_eot_embedding_from_arrays(
        x0,
        x1,
        t=0,
        rel_reg=1.0,
        embd_dim=2,
        return_artifacts=True,
    )

    assert embedding.shape == (7, 2)
    assert artifacts["left_embedding"].shape == (3, 2)
    assert artifacts["right_embedding"].shape == (4, 2)
    assert artifacts["ot_plan"].shape == (3, 4)
    assert artifacts["epsilon"] > 0.0
    assert artifacts["median_squared_distance"] > 0.0
    assert np.isfinite(embedding).all()


def test_compute_eot_embedding_rejects_invalid_embedding_dim():
    with pytest.raises(ValueError, match="embd_dim must be positive"):
        compute_eot_embedding_from_plan(np.eye(3), embd_dim=0)

    with pytest.raises(ValueError, match="embd_dim must be an integer"):
        compute_eot_embedding_from_plan(np.eye(3), embd_dim=1.5)


def test_compute_eot_then_tsne_embedding_from_arrays_smoke():
    x0 = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.2, 0.5],
            [0.1, 1.0, 0.2],
            [0.8, 0.8, 0.3],
            [1.2, 0.4, 0.7],
        ]
    )
    x1 = np.array(
        [
            [0.2, 0.1, 0.9],
            [1.1, 0.0, 0.4],
            [0.0, 1.2, 0.1],
            [0.8, 0.9, 0.3],
            [1.1, 1.0, 0.6],
            [0.3, 0.7, 0.8],
        ]
    )

    embedding, artifacts = compute_eot_then_tsne_embedding_from_arrays(
        x0,
        x1,
        t=0,
        rel_reg=1.0,
        eot_dim=50,
        random_state=0,
        perplexity=2.0,
        n_iter=12,
        early_exaggeration_iter=4,
        return_artifacts=True,
    )

    assert embedding.shape == (11, 2)
    assert artifacts["left_embedding"].shape == (5, 2)
    assert artifacts["right_embedding"].shape == (6, 2)
    assert artifacts["eot_embedding"].shape == (11, 4)
    assert artifacts["eot_left_embedding"].shape == (5, 4)
    assert artifacts["eot_right_embedding"].shape == (6, 4)
    assert artifacts["ot_plan"].shape == (5, 6)
    assert artifacts["eot_dim"] == 50
    assert artifacts["eot_dim_used"] == 4
    assert artifacts["tsne_perplexity"] == 2.0
    assert artifacts["epsilon"] > 0.0
    assert artifacts["median_squared_distance"] > 0.0
    # The task runner records the plan's device from these keys.
    assert artifacts["ot_device"] == "cpu"
    assert artifacts["torch_device"] == "cpu"
    assert np.isfinite(embedding).all()


def test_resolve_effective_k_split_uses_largest_remainder_examples():
    split = resolve_effective_k_split(effective_k=10, n0=5000, n1=10000)
    assert split.batch0_k == 3
    assert split.batch1_k == 7
    assert split.batch0_k + split.batch1_k == split.effective_k

    split = resolve_effective_k_split(effective_k=5, n0=2839, n1=1221)
    assert split.batch0_k == 3
    assert split.batch1_k == 2
    assert split.batch0_k + split.batch1_k == split.effective_k


def test_resolve_effective_k_split_ties_go_to_batch0_when_sizes_match():
    split = resolve_effective_k_split(effective_k=7, n0=100, n1=100)
    assert split.batch0_k == 4
    assert split.batch1_k == 3
    assert split.batch0_k + split.batch1_k == 7


def test_resolve_effective_k_split_ties_go_to_smaller_batch():
    split = resolve_effective_k_split(effective_k=4, n0=3, n1=5)
    assert split.batch0_k == 2
    assert split.batch1_k == 2
    assert split.batch0_k + split.batch1_k == 4


@pytest.mark.parametrize("effective_k", [0, -1])
def test_resolve_effective_k_split_rejects_nonpositive_k(effective_k):
    with pytest.raises(ValueError, match="effective_k must be positive"):
        resolve_effective_k_split(effective_k=effective_k, n0=5, n1=5)


def test_resolve_effective_k_split_rejects_noninteger_k():
    with pytest.raises(ValueError, match="effective_k must be an integer"):
        resolve_effective_k_split(effective_k=3.5, n0=5, n1=5)


def test_resolve_effective_k_split_rejects_too_large_k():
    with pytest.raises(ValueError, match="exceeds the available non-self neighbors"):
        resolve_effective_k_split(effective_k=9, n0=5, n1=5)


def test_sparsify_four_block_top_k_excludes_same_batch_diagonals():
    matrix = np.arange(81, dtype=float).reshape(9, 9)
    matrix = matrix + matrix.T
    np.fill_diagonal(matrix, matrix.max() + 100.0)

    sparse_matrix, split = sparsify_four_block_top_k(
        matrix,
        n0=4,
        n1=5,
        effective_k=3,
        binarize=True,
    )

    assert split.batch0_k == 1
    assert split.batch1_k == 2
    np.testing.assert_array_equal(np.diag(sparse_matrix), np.zeros(9))
    assert set(np.unique(sparse_matrix)).issubset({0.0, 1.0})
    assert np.all(sparse_matrix[:4, :4].sum(axis=1) >= 1)
    assert np.all(sparse_matrix[:4, 4:].sum(axis=1) >= 2)
    assert np.all(sparse_matrix[4:, :4].sum(axis=1) >= 1)
    assert np.all(sparse_matrix[4:, 4:].sum(axis=1) >= 2)


def test_compute_ot_tsne_embedding_from_arrays_smoke():
    x0 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.2],
            [0.1, 1.0],
            [0.8, 0.8],
        ]
    )
    x1 = np.array(
        [
            [0.2, 0.1],
            [1.1, 0.0],
            [0.0, 1.2],
            [0.8, 0.9],
            [1.1, 1.0],
        ]
    )

    embedding, artifacts = compute_ot_tsne_embedding_from_arrays(
        x0,
        x1,
        rel_reg=1.0,
        effective_k=3,
        random_state=0,
        n_epochs=12,
        return_artifacts=True,
    )

    assert embedding.shape == (9, 2)
    assert np.isfinite(embedding).all()
    assert artifacts["ot_plan"].shape == (4, 5)
    # The graph is built block by block and returned sparse; the dense squares no longer exist.
    assert "four_block_matrix" not in artifacts
    assert "sparse_four_block_matrix" not in artifacts
    assert sp.issparse(artifacts["graph"])
    assert artifacts["graph"].shape == (9, 9)
    assert artifacts["effective_k"] == 3
    assert artifacts["batch0_k"] == 1
    assert artifacts["batch1_k"] == 2
    assert artifacts["epsilon"] > 0.0
    assert artifacts["median_squared_distance"] > 0.0
    np.testing.assert_allclose(artifacts["graph"].sum(), 1.0)


def test_within_batch_knn_block_is_binary_symmetric_and_self_free():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 6))
    k = 4

    block = _within_batch_knn_block(x, k)

    assert block.shape == (30, 30)
    assert set(np.unique(block)).issubset({0.0, 1.0})
    np.testing.assert_array_equal(block, block.T)  # OR-symmetric
    np.testing.assert_array_equal(np.diag(block), np.zeros(30))  # no self-loops
    # Every point keeps at least its own k neighbors; OR-symmetrization only adds.
    assert np.all(block.sum(axis=1) >= k)


def test_knn_diagonal_does_not_fabricate_edges_unlike_topk_binarize():
    # Insert fixed kNN edges without reselecting a binary block, which can add tied-zero edges.
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=(24, 5))
    x1 = rng.normal(size=(36, 5)) + 0.5
    plan = compute_ot_plan(x0, x1, rel_reg=1 / 10)

    _, artifacts = compute_ot_tsne_knndiag_embedding_from_plan(
        plan, x0, x1, effective_k=15, random_state=0, n_epochs=10, return_artifacts=True
    )

    split = resolve_effective_k_split(15, x0.shape[0], x1.shape[0])
    knn0 = _within_batch_knn_block(x0, split.batch0_k)
    knn1 = _within_batch_knn_block(x1, split.batch1_k)
    # The reported diagonal edge counts equal the standalone kNN graphs' — no fabrication.
    assert artifacts["diag0_edges"] == int((knn0 > 0.0).sum())
    assert artifacts["diag1_edges"] == int((knn1 > 0.0).sum())
    assert artifacts["batch0_k"] == split.batch0_k
    assert artifacts["batch1_k"] == split.batch1_k


def test_selfot_diagonal_passes_through_the_same_sparsifier():
    # Rebuild self-OT blocks with the common top-k rule and compare diagonal edge counts.
    rng = np.random.default_rng(2)
    x0 = rng.normal(size=(24, 5))
    x1 = rng.normal(size=(36, 5)) + 0.5
    plan, ot_art = compute_ot_plan(x0, x1, rel_reg=1 / 10, return_artifacts=True)
    epsilon = ot_art["epsilon"]
    self0 = compute_ot_plan(x0, x0, epsilon=epsilon)
    self1 = compute_ot_plan(x1, x1, epsilon=epsilon)
    n0, n1 = plan.shape

    _, artifacts = compute_ot_tsne_diagonal_embedding_from_plan(
        plan, self0, self1, effective_k=15, random_state=0, n_epochs=10, return_artifacts=True
    )

    matrix = build_ot_tsne_four_block_matrix(plan)
    d0 = np.array(self0, copy=True)
    d1 = np.array(self1, copy=True)
    np.fill_diagonal(d0, 0.0)
    np.fill_diagonal(d1, 0.0)
    matrix[:n0, :n0] = d0
    matrix[n0:, n0:] = d1
    sparse, _ = sparsify_four_block_top_k(matrix, n0=n0, n1=n1, effective_k=15, binarize=True)
    assert artifacts["diag0_edges"] == int((sparse[:n0, :n0] > 0.0).sum())
    assert artifacts["diag1_edges"] == int((sparse[n0:, n0:] > 0.0).sum())


def test_diagonal_variant_entry_points_shapes_and_determinism():
    rng = np.random.default_rng(3)
    x0 = rng.normal(size=(24, 5))
    x1 = rng.normal(size=(36, 5)) + 0.5

    for compute in (
        compute_ot_tsne_knndiag_embedding_from_arrays,
        compute_ot_tsne_gramdiag_embedding_from_arrays,
        compute_ot_tsne_zerodiag_embedding_from_arrays,
    ):
        first, artifacts = compute(
            x0,
            x1,
            rel_reg=1 / 10,
            effective_k=15,
            random_state=5,
            n_epochs=12,
            return_artifacts=True,
        )
        second = compute(
            x0,
            x1,
            rel_reg=1 / 10,
            effective_k=15,
            random_state=5,
            n_epochs=12,
        )
        assert first.shape == (60, 2)
        assert np.isfinite(first).all()
        np.testing.assert_array_equal(first, second)  # deterministic given seed
        assert artifacts["epsilon"] > 0.0
        # lightweight artifacts only: no (n0+n1)-square matrices are returned
        assert "four_block_matrix" not in artifacts
        assert "graph" not in artifacts


# Initialization ablation (init="random" | "spectral" | "eot")


def _init_test_graph(n0=24, n1=36, effective_k=15, seed=0):
    """A realistic normalized four-block OT graph, plus the plan it came from."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n0, 5))
    x1 = rng.normal(size=(n1, 5)) + 0.5
    plan = compute_ot_plan(x0, x1, rel_reg=1 / 10)
    matrix = build_ot_tsne_four_block_matrix(plan)
    sparse, _ = sparsify_four_block_top_k(
        matrix, n0=n0, n1=n1, effective_k=effective_k, binarize=True
    )
    return normalize_ot_tsne_graph(sparse), plan, x0, x1


def _legacy_tsne_from_graph(graph, random_state=0, n_epochs=12, init_scale=10000.0):
    """The pre-``init`` implementation, inlined so the default path can be pinned.

    A frozen golden array would be brittle across BLAS/openTSNE builds; re-deriving the old
    behavior here is not.
    """
    graph = np.asarray(graph, dtype=float).copy()
    graph = np.nan_to_num(graph, nan=0.0, posinf=0.0, neginf=0.0)
    graph[graph < 0.0] = 0.0
    np.fill_diagonal(graph, 0.0)
    graph = graph / float(graph.sum())

    n_points = graph.shape[0]
    init = np.random.RandomState(int(random_state)).randn(n_points, 2)
    init_std = float(init[:, 0].std())
    init = init / (max(init_std, 1e-12) * float(init_scale))
    n_early = n_epochs // 3
    n_late = n_epochs - n_early

    affinities = Affinities()
    affinities.P = sp.csr_matrix(graph)
    embedder = TSNE(
        n_components=2,
        random_state=int(random_state),
        verbose=False,
        early_exaggeration_iter=n_early,
        n_iter=n_late,
    )
    return np.asarray(embedder.fit(affinities=affinities, initialization=init))


def test_run_tsne_from_graph_default_init_is_byte_identical():
    """The regression guard: adding the init axis must not move any existing result."""
    graph, _, _, _ = _init_test_graph()

    expected = _legacy_tsne_from_graph(graph, random_state=0, n_epochs=12)
    legacy = dict(random_state=0, n_epochs=12, early_exaggeration_iter=None)
    np.testing.assert_array_equal(run_tsne_from_graph(graph, **legacy), expected)
    np.testing.assert_array_equal(run_tsne_from_graph(graph, init="random", **legacy), expected)


def test_within_batch_knn_indices_survive_many_identical_points():
    """More than k + 1 identical rows used to drop a point's self entry and fail to reshape."""
    from ne_aligned_utils.methods.ot_methods import within_batch_knn_indices

    rng = np.random.default_rng(0)
    x = np.vstack([np.zeros((20, 5)), rng.normal(size=(20, 5))])
    idx = within_batch_knn_indices(x, 10)
    assert idx.shape == (40, 10)
    assert not (idx == np.arange(40)[:, None]).any()
    x1 = rng.normal(size=(40, 5))
    embedding = compute_ot_tsne_knndiag_embedding_from_arrays(
        x, x1, effective_k=10, n_epochs=5, early_exaggeration_iter=0
    )
    assert embedding.shape == (80, 2)


def test_ot_tsne_forwards_n_jobs_to_open_tsne(monkeypatch):
    """The OT layout runs on the requested threads, like the common 50D t-SNE tail."""
    import ne_aligned_utils.methods.ot_graph as ot_graph

    seen = []

    class RecordingTSNE(ot_graph.TSNE):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("n_jobs"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(ot_graph, "TSNE", RecordingTSNE)
    rng = np.random.default_rng(0)
    x0, x1 = rng.normal(size=(30, 5)), rng.normal(size=(40, 5)) + 0.5
    compute_ot_tsne_embedding_from_arrays(x0, x1, n_epochs=12, n_jobs=3)
    compute_ot_tsne_zerodiag_embedding_from_arrays(x0, x1, n_epochs=12, n_jobs=2)
    run_tsne_from_graph(_init_test_graph()[0], n_epochs=12)
    assert seen == [3, 2, 1]


def test_ot_tsne_early_exaggeration_schedule(monkeypatch):
    """None keeps the old first-third split; an integer runs before n_epochs main iterations.

    The default is openTSNE's schedule, 250 exaggerated then 500 iterations.
    """
    import ne_aligned_utils.methods.ot_graph as ot_graph

    seen = []

    class RecordingTSNE(ot_graph.TSNE):
        def __init__(self, *args, **kwargs):
            seen.append((kwargs["early_exaggeration_iter"], kwargs["n_iter"]))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(ot_graph, "TSNE", RecordingTSNE)
    graph = _init_test_graph()[0]
    run_tsne_from_graph(graph, n_epochs=12, early_exaggeration_iter=None)
    run_tsne_from_graph(graph, n_epochs=12, early_exaggeration_iter=5)
    rng = np.random.default_rng(0)
    x0, x1 = rng.normal(size=(30, 5)), rng.normal(size=(40, 5)) + 0.5
    compute_ot_tsne_embedding_from_arrays(x0, x1, n_epochs=12, early_exaggeration_iter=7)
    compute_ot_tsne_embedding_from_arrays(x0, x1)
    assert seen == [(4, 8), (5, 12), (7, 12), (250, 500)]
    with pytest.raises(ValueError, match="non-negative"):
        run_tsne_from_graph(graph, n_epochs=12, early_exaggeration_iter=-1)


def test_finalize_tsne_init_contract():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(50, 2)) * 1e6
    raw[:, 0] -= raw[:, 0].max() + 1.0  # force a negative-dominant first column

    out = _finalize_tsne_init(raw, init_scale=10000.0, random_state=0)

    assert out.shape == (50, 2)
    assert np.isfinite(out).all()
    # Allow seeded jitter's finite-sample perturbation of first-axis standard deviation.
    assert out[:, 0].std() == pytest.approx(1e-4, rel=1e-2)
    # (i) deterministic signs: the largest-magnitude entry of each column is positive.
    pivots = np.argmax(np.abs(out), axis=0)
    assert np.all(out[pivots, np.arange(2)] > 0.0)
    # sign of the input must not matter
    np.testing.assert_allclose(_finalize_tsne_init(-raw, random_state=0), out, rtol=0, atol=0)
    # init_scale is a divisor, not a multiplier
    coarse = _finalize_tsne_init(raw, init_scale=100.0, random_state=0)
    assert coarse[:, 0].std() == pytest.approx(1e-2, rel=1e-2)


def test_finalize_tsne_init_rejects_unusable_input():
    """Non-finite and degenerate inputs are errors, not something to silently patch.

    Zero-filling a NaN would substitute a different geometry, and a constant first
    coordinate cannot be rescaled at all: dividing by a floor would leave axis 0 flat while
    inflating axis 1 by the floor's reciprocal, and the jitter (proportional to the axis-0
    spread) would be identically zero.
    """
    rng = np.random.default_rng(0)
    broken = rng.normal(size=(20, 2))
    broken[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _finalize_tsne_init(broken)

    degenerate = np.column_stack([np.ones(20), rng.normal(size=20)])
    with pytest.raises(ValueError, match="degenerate"):
        _finalize_tsne_init(degenerate)

    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        _finalize_tsne_init(rng.normal(size=(20, 3)))


@pytest.mark.parametrize("init", OT_TSNE_INIT_CHOICES)
def test_init_choices_shapes_and_determinism(init):
    x0 = np.random.default_rng(3).normal(size=(24, 5))
    x1 = np.random.default_rng(4).normal(size=(36, 5)) + 0.5

    kwargs = dict(rel_reg=1 / 10, effective_k=15, random_state=5, n_epochs=12, init=init)
    first = compute_ot_tsne_embedding_from_arrays(x0, x1, **kwargs)
    second = compute_ot_tsne_embedding_from_arrays(x0, x1, **kwargs)

    assert first.shape == (60, 2)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(("init", "recorded"), [("spectral", "spectral"), ("eot", "array")])
def test_ot_umap_accepts_nonrandom_initialization(init, recorded):
    """UMAP accepts its spectral initializer and EOT coordinates supplied as an array."""
    x0 = np.random.default_rng(3).normal(size=(8, 5))
    x1 = np.random.default_rng(4).normal(size=(9, 5)) + 0.5

    embedding, artifacts = compute_ot_tsne_embedding_from_arrays(
        x0,
        x1,
        rel_reg=1 / 10,
        effective_k=3,
        n_epochs=12,
        init=init,
        backend="umap",
        return_artifacts=True,
    )

    assert embedding.shape == (17, 2)
    assert np.isfinite(embedding).all()
    assert artifacts["umap_init"] == recorded


@pytest.mark.parametrize("init", ["spectral", "eot"])
def test_deterministic_inits_are_seed_sensitive(init):
    """Without the seeded jitter these arms would collapse onto one embedding per graph,
    making the seed error bars in the ablation meaningless."""
    x0 = np.random.default_rng(3).normal(size=(24, 5))
    x1 = np.random.default_rng(4).normal(size=(36, 5)) + 0.5

    kwargs = dict(rel_reg=1 / 10, effective_k=15, n_epochs=12, init=init)
    first = compute_ot_tsne_embedding_from_arrays(x0, x1, random_state=0, **kwargs)
    second = compute_ot_tsne_embedding_from_arrays(x0, x1, random_state=1, **kwargs)

    assert not np.allclose(first, second)


def test_spectral_structure_is_independent_of_the_experiment_seed():
    """The eigensolver seed is fixed, so a seed sweep perturbs only the jitter.

    With repeated eigenvalues a varying ARPACK start vector would return a different basis
    of the same eigenspace -- a rotation the sign fix cannot undo -- which would confound
    "different random perturbation" with "different structural initialization".
    """
    graph, _, _, _ = _init_test_graph()

    first, first_info = _spectral_init(graph)
    second, second_info = _spectral_init(graph)

    np.testing.assert_array_equal(first, second)
    assert _spectral_init.__code__.co_varnames[: _spectral_init.__code__.co_argcount] == ("graph",)
    assert first_info["spectral_n_components"] >= 1


def test_eot_init_is_the_plan_svd_eigenbasis():
    """``eot`` is the top-2 non-trivial stacked singular modes of the cross plan, which are
    also the leading eigenvectors of the four-block matrix (``E.T @ E == (n0 + n1) I``)."""
    _, plan, _, _ = _init_test_graph()
    n0, n1 = plan.shape

    resolved = _resolve_plan_init("eot", plan)
    assert resolved.shape == (n0 + n1, 2)

    u, _, vt = np.linalg.svd(plan, full_matrices=False)
    stacked = np.concatenate([u * np.sqrt(n0), vt.T * np.sqrt(n1)], axis=0)
    np.testing.assert_allclose(stacked.T @ stacked, (n0 + n1) * np.eye(stacked.shape[1]), atol=1e-8)
    # ...and the resolved init is columns 1:3 of that basis, up to rounding and column signs.
    flip = methods._flip_init_signs
    np.testing.assert_allclose(flip(resolved), flip(stacked[:, 1:3]), atol=1e-10)
    np.testing.assert_allclose(
        flip(resolved),
        flip(compute_eot_embedding_from_plan(plan, t=0.0, embd_dim=2, start=1)),
        atol=1e-10,
    )


def test_eot_init_without_plan_diagonals_uses_only_the_leading_modes():
    """Variants without plan diagonals compute 3 modes for the EOT init, not the full SVD.
    The init matches the full-SVD init up to rounding and is the same on every call."""
    from ne_aligned_utils.methods.ot_affinity import PlanDiagonals
    from ne_aligned_utils.methods.ot_methods import _block_graph

    _, plan, _, _ = _init_test_graph()
    flip = methods._flip_init_signs
    _, _, full_values, full_init = _block_graph(
        plan, init="eot", effective_k=15, rule="split", within_batch=PlanDiagonals()
    )
    # Plan diagonals still use the whole spectrum.
    np.testing.assert_allclose(full_values, np.linalg.svd(plan, compute_uv=False), rtol=1e-10)

    _, _, values, init = _block_graph(
        plan, init="eot", effective_k=15, rule="cross_only", within_batch=None
    )
    np.testing.assert_allclose(values, full_values[:3], rtol=1e-10)
    np.testing.assert_allclose(flip(init), flip(full_init), atol=1e-10)
    np.testing.assert_array_equal(init, _resolve_plan_init("eot", plan))
    _, _, _, again = _block_graph(
        plan, init="eot", effective_k=15, rule="cross_only", within_batch=None
    )
    np.testing.assert_array_equal(again, init)


def test_resolve_plan_init_passes_through_and_validates():
    _, plan, _, _ = _init_test_graph()
    assert _resolve_plan_init("random", plan) == "random"
    assert _resolve_plan_init("spectral", plan) == "spectral"
    assert _resolve_plan_init("pca", plan) == "pca"
    with pytest.raises(ValueError, match="init must be one of"):
        _resolve_plan_init("nope", plan)


def test_spectral_init_rejects_an_asymmetric_graph():
    """Silently symmetrizing would initialize from different affinities than t-SNE
    optimizes, so this is an error rather than a fallback."""
    graph, _, _, _ = _init_test_graph()
    graph[0, 1] = graph[1, 0] + 1.0

    with pytest.raises(ValueError, match="symmetric"):
        _spectral_init(graph)


def test_spectral_init_never_falls_back_on_degenerate_graph():
    # A graph with an isolated node: zero degree makes the degree matrix singular.
    graph = np.ones((8, 8))
    np.fill_diagonal(graph, 0.0)
    graph[7, :] = 0.0
    graph[:, 7] = 0.0
    graph = graph / graph.sum()

    candidate, info = _spectral_init(graph)
    assert candidate is None
    assert info["spectral_skipped"] == "zero_degree_row"

    # A random layout reported as spectral would make spectral-vs-random comparisons compare
    # random with random, so the run fails instead.
    with pytest.raises(ValueError, match="zero_degree_row.*no fallback"):
        run_tsne_from_graph(graph, random_state=0, n_epochs=10, init="spectral")
    with pytest.raises(ValueError, match="no fallback"):
        run_tsne_from_graph(sp.csr_matrix(graph), random_state=0, n_epochs=10, init="spectral")

    # too few points for k = n_components + 1 = 3
    tiny = np.ones((3, 3))
    np.fill_diagonal(tiny, 0.0)
    tiny = tiny / tiny.sum()
    candidate, info = _spectral_init(tiny)
    assert candidate is None
    assert info["spectral_skipped"] == "n_points<4"


def test_run_tsne_from_graph_records_init_used():
    graph, plan, _, _ = _init_test_graph()

    for init, expected in (("random", "random"), ("spectral", "spectral")):
        info = {}
        run_tsne_from_graph(graph, random_state=0, n_epochs=10, init=init, init_info=info)
        assert info["init_used"] == expected

    info = {}
    run_tsne_from_graph(
        graph,
        random_state=0,
        n_epochs=10,
        init=_resolve_plan_init("eot", plan),
        init_info=info,
    )
    assert info["init_used"] == "array"
    run_tsne_from_graph(
        graph,
        random_state=0,
        n_epochs=10,
        init=_resolve_plan_init("eot", plan),
        init_info=info,
        init_label="eot",
    )
    assert info["init_used"] == "eot"

    _, _, x0, x1 = _init_test_graph()
    info = {}
    run_tsne_from_graph(
        graph, random_state=0, n_epochs=10, init="pca", init_info=info, pca_input=np.vstack([x0, x1])
    )
    assert info["init_used"] == "pca"
    with pytest.raises(ValueError, match="needs pca_input"):
        run_tsne_from_graph(graph, n_epochs=10, init="pca")

    with pytest.raises(ValueError, match="init must be one of"):
        run_tsne_from_graph(graph, n_epochs=10, init="bogus")
    with pytest.raises(ValueError, match="precomputed init must have shape"):
        run_tsne_from_graph(graph, n_epochs=10, init=np.zeros((3, 2)))


@pytest.mark.parametrize("init", ["random", "spectral", "eot", "pca"])
def test_all_ot_tsne_variants_accept_init(init):
    """The wiring guard: catches a missed pass-through in any of the entry points, and checks
    that each records the scheme it actually used."""
    rng = np.random.default_rng(3)
    x0 = rng.normal(size=(24, 5))
    x1 = rng.normal(size=(36, 5)) + 0.5

    for compute in (
        compute_ot_tsne_embedding_from_arrays,
        compute_ot_tsne_knndiag_embedding_from_arrays,
        compute_ot_tsne_gramdiag_embedding_from_arrays,
        compute_ot_tsne_zerodiag_embedding_from_arrays,
        methods.compute_ot_tsne_selfotmedian_embedding_from_arrays,
        methods.compute_ot_tsne_zerodiag_split_embedding_from_arrays,
    ):
        embedding, artifacts = compute(
            x0,
            x1,
            rel_reg=1 / 10,
            effective_k=15,
            random_state=5,
            n_epochs=12,
            init=init,
            return_artifacts=True,
        )
        assert embedding.shape == (60, 2)
        assert np.isfinite(embedding).all()
        assert artifacts["init_used"] == init


def test_sparse_spectral_init_matches_the_dense_one():
    """The graph stays sparse; densifying it only to initialize does not fit in memory on the
    largest datasets."""
    graph, _, _, _ = _init_test_graph()
    dense, dense_info = _spectral_init(graph)
    sparse_result, sparse_info = _spectral_init(sp.csr_matrix(graph))
    np.testing.assert_allclose(sparse_result, dense, atol=1e-10)
    np.testing.assert_allclose(
        sparse_info["spectral_eigenvalues"], dense_info["spectral_eigenvalues"], atol=1e-12
    )

    asymmetric = sp.lil_matrix(graph)
    asymmetric[0, 1] = graph[1, 0] + 1.0
    with pytest.raises(ValueError, match="symmetric"):
        _spectral_init(asymmetric.tocsr())


def test_pca_init_is_openTSNE_pca_start_shared_by_every_variant():
    """``pca`` starts from the first two principal components of the joint input, rescaled
    like openTSNE's ``initialization="pca"``. It reads only the input, so every variant starts
    from the same layout; seeds differ only through the 1% jitter."""
    from openTSNE import initialization as opentsne_init

    graph, _, x0, x1 = _init_test_graph()
    x = np.vstack([x0, x1])
    captured = []

    class _Capture(Exception):
        pass

    def capture(self, affinities, initialization):
        captured.append(np.array(initialization))
        raise _Capture

    original = TSNE.fit
    TSNE.fit = capture
    try:
        for seed in (0, 0, 1):
            with pytest.raises(_Capture):
                run_tsne_from_graph(graph, random_state=seed, init="pca", pca_input=x)
    finally:
        TSNE.fit = original

    first, repeat, other_seed = captured
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, other_seed)
    # Same scale as openTSNE: first axis std 1e-4 before jitter.
    reference = opentsne_init.pca(x, random_state=0, add_jitter=False)
    unjittered = methods._finalize_tsne_init(_pca_reference(x), jitter_scale=0.0)
    assert np.isclose(unjittered[:, 0].std(), reference[:, 0].std())
    np.testing.assert_allclose(np.abs(unjittered), np.abs(reference), rtol=1e-6, atol=1e-12)
    # Jitter is small next to the layout.
    assert np.abs(first - other_seed).max() < 0.1 * np.abs(unjittered).max()


def _pca_reference(x):
    return PCA(n_components=2, svd_solver="full").fit_transform(x)


def test_eot_init_from_shared_basis_matches_a_standalone_svd():
    """The plan-level entry points reuse the four-block builder's SVD instead of paying a
    second dense one (~20 s on the largest benchmark dataset). The standalone init uses only
    the leading modes, so the two agree up to rounding and column signs."""
    _, plan, _, _ = _init_test_graph()
    _, _, eot_coordinates = _build_four_block_parts(plan, return_eot=True)

    shared = _resolve_plan_init("eot", plan, eot_coordinates)
    standalone = _resolve_plan_init("eot", plan)

    flip = methods._flip_init_signs
    np.testing.assert_allclose(flip(shared), flip(standalone), atol=1e-10)
    # EOT-2D solves for the same three leading modes as the standalone init.
    np.testing.assert_array_equal(
        standalone, compute_eot_embedding_from_plan(plan, t=0.0, embd_dim=2, start=1)
    )


def test_build_four_block_parts_agrees_with_the_public_builder():
    _, plan, _, _ = _init_test_graph()
    matrix, singular_values, _ = _build_four_block_parts(plan)

    public, public_singular = build_ot_tsne_four_block_matrix(plan, return_singular_values=True)
    np.testing.assert_array_equal(matrix, public)
    np.testing.assert_array_equal(singular_values, public_singular)
    np.testing.assert_array_equal(matrix, build_ot_tsne_four_block_matrix(plan))


def test_is_symmetric_matches_allclose_without_the_large_temporary():
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(200, 200))
    matrix = matrix + matrix.T

    assert _is_symmetric(matrix)
    assert _is_symmetric(matrix, block=7)  # block size must not change the answer
    matrix[3, 9] = matrix[9, 3] + 1.0
    assert not _is_symmetric(matrix)
    assert not _is_symmetric(matrix, block=7)


# Backend selection for the integration-only methods
#
# The representation is the controlled variable and the embedder is the only thing that varies. If
# the integration step drifted between backends the comparison would measure both at once, so the
# equality below is the whole point of the phase rather than an incidental property.


def test_scvi_adapter_defaults_match_the_central_config():
    for function in (
        methods.compute_scvi_embedding_from_arrays,
        methods.compute_scvi_then_tsne_embedding_from_arrays,
    ):
        parameters = inspect.signature(function).parameters
        assert parameters["max_epochs"].default is DEFAULT_SCVI_MAX_EPOCHS
        assert parameters["batch_size"].default == DEFAULT_SCVI_BATCH_SIZE


#: Integration methods and representation artifact keys. requires marks optional toolchains;
#: counts requires nonnegative raw counts. Joint PCA is checked separately.
BACKEND_CASES = [
    (
        "joint",
        "compute_tsne_embedding_from_arrays",
        "pca_components",
        "pca_embedding",
        {},
        None,
        False,
    ),
    (
        "ref-our",
        "compute_ref_tsne_embedding_from_arrays",
        "pca_components",
        "reference_input",
        {},
        None,
        False,
    ),
    (
        "cca-50",
        "compute_cca_then_tsne_embedding_from_arrays",
        "cca_dim",
        "cca_embedding",
        {},
        None,
        False,
    ),
    (
        "cca-50-uncentered",
        "compute_cca_then_tsne_embedding_from_arrays",
        "cca_dim",
        "cca_embedding",
        {"center": False},
        None,
        False,
    ),
    (
        "eot-50",
        "compute_eot_then_tsne_embedding_from_arrays",
        "eot_dim",
        "eot_embedding",
        {},
        None,
        False,
    ),
    (
        "harmony-50",
        "compute_harmony_then_tsne_embedding_from_arrays",
        "harmony_dim",
        "harmony_embedding",
        {},
        "harmonypy",
        False,
    ),
    (
        "scanorama-50",
        "compute_scanorama_then_tsne_embedding_from_arrays",
        "scanorama_dim",
        "scanorama_embedding",
        {},
        "scanorama",
        False,
    ),
    (
        "scvi-50",
        "compute_scvi_then_tsne_embedding_from_arrays",
        "latent_dim",
        "scvi_embedding",
        {"max_epochs": 2},
        "scvi",
        True,
    ),
    (
        "seurat-50",
        "compute_seurat_then_tsne_embedding_from_arrays",
        "seurat_dim",
        "integration_embedding",
        {},
        "R",
        True,
    ),
    (
        "seurat3-50",
        "compute_seurat3_then_tsne_embedding_from_arrays",
        "seurat_dim",
        "integration_embedding",
        {},
        "R",
        True,
    ),
    (
        "liger1-50",
        "compute_liger1_then_tsne_embedding_from_arrays",
        "liger_dim",
        "integration_embedding",
        {},
        "R",
        True,
    ),
    (
        "liger-50",
        "compute_liger_then_tsne_embedding_from_arrays",
        "liger_dim",
        "integration_embedding",
        {},
        "LIGER",
        True,
    ),
]

#: The kinds §2.3 requires covering. Asserted against config.has_backend_axis below, so a kind
#: cannot quietly go uncovered.
COVERED_BY_BACKEND_TESTS = {case[0] for case in BACKEND_CASES}


def _r_available() -> bool:
    """Return whether the explicitly configured R launcher is available to these tests."""
    import os
    from pathlib import Path

    if os.environ.get("NE_ALIGNED_RSCRIPT"):
        return True
    root = Path(os.environ.get("NE_ALIGNED_R_ROOT", Path.home() / ".local/opt/ne-aligned-r"))
    micromamba = root / "bin" / "micromamba"
    if not micromamba.is_file():
        return False
    os.environ["MAMBA_ROOT_PREFIX"] = str(root / "mmroot")
    os.environ["NE_ALIGNED_RSCRIPT"] = f"{micromamba} run -n rverify Rscript"
    return True


def _require_toolchain(requires):
    if requires is None:
        return
    if requires == "R":
        if not _r_available():
            pytest.skip("the R toolchain is absent, so seurat/liger cannot run")
        return
    if requires == "LIGER":
        import os

        if not os.environ.get("NE_ALIGNED_LIGER_RSCRIPT"):
            pytest.skip("the current LIGER R toolchain is absent")
        return
    pytest.importorskip(requires)


def _resolve_backend_case(func_name):
    """The wrapper, from whichever module defines it."""
    from ne_aligned_utils.methods.external import r

    return getattr(methods, func_name, None) or getattr(r, func_name)


def _backend_case_inputs(counts):
    """Inputs a case can actually consume.

    The raw-count kinds reject negatives, and LIGER's iNMF additionally needs genuinely variable
    genes: on pure Poisson noise ``selectGenes`` leaves fewer features than the factor count and the
    R side errors out. So the count matrices carry two populations with distinct high-expression
    blocks rather than being undifferentiated noise. Established by running it, not assumed.
    """
    rng = np.random.default_rng(0)
    if not counts:
        return rng.normal(size=(60, 40)), rng.normal(size=(55, 40)) + 0.3

    def block_counts(n, g=300):
        base = rng.poisson(1.0, size=(n, g)).astype(float)
        half = n // 2
        base[:half, : g // 3] += rng.poisson(20.0, size=(half, g // 3))
        base[half:, g // 3 : 2 * g // 3] += rng.poisson(20.0, size=(n - half, g // 3))
        return base

    return block_counts(160), block_counts(150)


@pytest.mark.parametrize(
    ("kind", "func_name", "dim_kw", "repr_key", "extra", "requires", "counts"),
    BACKEND_CASES,
    ids=[case[0] for case in BACKEND_CASES],
)
def test_representation_is_byte_identical_across_backends(
    kind, func_name, dim_kw, repr_key, extra, requires, counts
):
    """Swapping the embedder must not perturb the representation fed to it, to the byte.

    ``np.array_equal`` is too weak here: it returns True across differing dtypes, and a float32
    vs float64 representation would be a real difference in what the two arms were given.
    """
    _require_toolchain(requires)
    func = _resolve_backend_case(func_name)
    x0, x1 = _backend_case_inputs(counts)
    kwargs = {dim_kw: 8, "random_state": 0, "perplexity": 8.0, "n_iter": 50, **extra}

    _, tsne_artifacts = func(x0, x1, backend="tsne", return_artifacts=True, **kwargs)
    _, umap_artifacts = func(x0, x1, backend="umap", return_artifacts=True, **kwargs)

    a, b = tsne_artifacts[repr_key], umap_artifacts[repr_key]
    assert a.dtype == b.dtype, f"{repr_key} dtype differs between backends"
    assert a.shape == b.shape
    assert a.tobytes() == b.tobytes(), f"{repr_key} is not byte-identical across backends"


@pytest.mark.parametrize(
    ("kind", "func_name", "dim_kw", "repr_key", "extra", "requires", "counts"),
    BACKEND_CASES,
    ids=[case[0] for case in BACKEND_CASES],
)
def test_backends_are_recorded_and_produce_different_layouts(
    kind, func_name, dim_kw, repr_key, extra, requires, counts
):
    """Each arm records its own embedder parameters, and the two actually differ."""
    _require_toolchain(requires)
    func = _resolve_backend_case(func_name)
    x0, x1 = _backend_case_inputs(counts)
    kwargs = {dim_kw: 8, "random_state": 0, "perplexity": 8.0, "n_iter": 50, **extra}

    e_tsne, a_tsne = func(x0, x1, backend="tsne", return_artifacts=True, **kwargs)
    e_umap, a_umap = func(x0, x1, backend="umap", return_artifacts=True, **kwargs)

    assert a_tsne["embedding_backend"] == "tsne"
    assert a_umap["embedding_backend"] == "umap"

    # Check backend-specific concepts, allowing prefixed and unprefixed artifact keys.
    def perplexity_keys(artifacts):
        return {k for k in artifacts if "perplexity" in k}

    def neighbour_keys(artifacts):
        # Joint stores unprefixed artifacts; integration wrappers prefix backend-specific keys.
        return {k for k in artifacts if "n_neighbors" in k}

    assert perplexity_keys(a_tsne), f"the t-SNE arm recorded no perplexity: {sorted(a_tsne)}"
    assert not neighbour_keys(a_tsne), "the t-SNE arm recorded a UMAP neighbourhood"
    assert neighbour_keys(a_umap), f"the UMAP arm recorded no n_neighbors: {sorted(a_umap)}"
    assert not perplexity_keys(a_umap), "the UMAP arm recorded a perplexity"

    assert not np.allclose(e_tsne, e_umap), "the two backends produced the same layout"


def test_umap_neighbourhood_default_is_independent_of_tsne_perplexity(monkeypatch):
    """UMAP has no perplexity; its default is the explicit benchmark neighborhood."""
    seen = {}

    class FakeUMAP:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self._a = 1.0
            self._b = 1.0

        def fit_transform(self, x):
            return np.zeros((len(x), 2))

    monkeypatch.setattr(methods.umap, "UMAP", FakeUMAP)
    methods.embed_2d(np.ones((20, 3)), "umap", perplexity=50.0)
    assert seen["n_neighbors"] == 15


def test_early_exaggeration_is_refused_by_the_umap_backend():
    """A t-SNE-only parameter must not be silently dropped when the backend changes."""
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="no UMAP equivalent"):
        methods.embed_2d(rng.normal(size=(30, 5)), "umap", early_exaggeration_iter=10, n_iter=50)


def test_backend_tests_cover_every_integration_only_kind():
    """Require backend-comparison coverage for every separable integration method.

    Exclude EPS_K_METHODS, whose graph interfaces are tested separately; EOT-50 remains
    an ordinary representation followed by a backend and must be covered.
    """
    from ne_aligned_utils.benchmark.sweep import EPS_K_METHODS, TASK_KINDS
    from ne_aligned_utils.config import has_backend_axis

    integration_only = {k for k in TASK_KINDS if has_backend_axis(k) and k not in EPS_K_METHODS}
    assert integration_only == COVERED_BY_BACKEND_TESTS, (
        f"uncovered: {sorted(integration_only - COVERED_BY_BACKEND_TESTS)}, "
        f"stale: {sorted(COVERED_BY_BACKEND_TESTS - integration_only)}"
    )


@pytest.mark.parametrize("backend", ["tsne", "umap"])
def test_ref_embedding_query_does_not_move_the_reference(backend):
    """The property that makes ref-* a reference-atlas method rather than a joint one.

    Batch 0 is embedded on its own and frozen; batch 1 is projected into that fixed space by an
    out-of-sample transform. So the reference half of the output must be *unchanged* by whatever the
    query contains -- otherwise the method is a joint embedding wearing a different name, and the
    UMAP arm in particular could quietly have become one by calling fit on the concatenation.
    """
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(120, 30))
    x1_a = rng.normal(size=(90, 30)) + 0.3
    x1_b = rng.normal(size=(60, 30)) * 3.0 - 5.0  # a very different query

    kwargs = {
        "pca_components": 8,
        "random_state": 0,
        "perplexity": 10.0,
        "n_iter": 80,
        "backend": backend,
        "return_artifacts": True,
    }
    _, first = methods.compute_ref_tsne_embedding_from_arrays(x0, x1_a, **kwargs)
    _, second = methods.compute_ref_tsne_embedding_from_arrays(x0, x1_b, **kwargs)

    assert first["left_embedding"].shape == second["left_embedding"].shape == (120, 2)
    assert first["right_embedding"].shape == (90, 2)
    assert second["right_embedding"].shape == (60, 2)
    assert first["left_embedding"].tobytes() == second["left_embedding"].tobytes(), (
        f"the {backend} reference embedding changed when only the query changed, "
        "so the query is moving the reference"
    )


# The directed neighbor list the UMAP arm embeds


@pytest.mark.parametrize("rel_reg", [1.0, 1 / 10, 1 / 100, 1 / 300])
def test_directed_knn_union_equals_the_sparsifier_mask(rel_reg):
    """Require directed-neighbor union to equal the graph passed to t-SNE.

    Sweep regularization to expose near-tie rounding differences; symmetrize scores
    before row and column ranking.
    """
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(50, 12))
    x1 = rng.normal(size=(40, 12)) + 0.4
    n0, n1, k = 50, 40, 15
    matrix, _, _ = methods._build_four_block_parts(methods.compute_ot_plan(x0, x1, rel_reg=rel_reg))
    mask, split = methods.sparsify_four_block_top_k(
        matrix, n0=n0, n1=n1, effective_k=k, binarize=True
    )
    directed, split2 = methods.four_block_directed_knn(matrix, n0=n0, n1=n1, effective_k=k)

    assert split == split2
    assert directed.shape == (n0 + n1, k)
    adjacency = np.zeros((n0 + n1, n0 + n1), dtype=bool)
    adjacency[np.arange(n0 + n1)[:, None], directed] = True
    union = adjacency | adjacency.T
    np.fill_diagonal(union, False)
    assert np.array_equal(union, mask > 0)


def test_directed_knn_rows_are_exactly_the_neighbour_budget_and_never_repeat():
    """Every row holds exactly k0 + k1 distinct neighbors, none of them itself.

    UMAP truncates a precomputed neighbor list to its first n_neighbors columns, and these
    distances carry no ordering, so a short or padded row would silently drop whichever block's
    neighbors sat in the trailing columns.
    """
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=(30, 8))
    x1 = rng.normal(size=(25, 8)) + 0.4
    matrix, _, _ = methods._build_four_block_parts(methods.compute_ot_plan(x0, x1, rel_reg=1 / 10))
    directed, split = methods.four_block_directed_knn(matrix, n0=30, n1=25, effective_k=12)

    assert directed.shape[1] == split.effective_k == split.batch0_k + split.batch1_k
    for row, targets in enumerate(directed):
        assert len(set(targets.tolist())) == len(targets), f"row {row} repeats a neighbour"
        assert row not in targets.tolist(), f"row {row} is its own neighbour"
    # Batch 0 targets live in the left columns, batch 1 in the right, both globally indexed.
    assert directed[:, : split.batch0_k].max() < 30
    assert directed[:, split.batch0_k :].min() >= 30


def test_knndiag_within_batch_indices_replace_the_right_two_corners():
    """knndiag takes its within-batch neighbors from the data, not from the transport plan.

    "Within-batch" is a different pair of columns per batch: a batch-0 point's own batch is the
    left half of the row and a batch-1 point's own batch is the right half, so the replacement is
    two rectangles on opposite corners. The other two corners are cross-batch and must be left as
    the transport plan chose them.
    """
    rng = np.random.default_rng(2)
    n0, n1, k = 40, 30, 12
    x0 = rng.normal(size=(n0, 10))
    x1 = rng.normal(size=(n1, 10)) + 0.4
    matrix, _, _ = methods._build_four_block_parts(methods.compute_ot_plan(x0, x1, rel_reg=1 / 10))
    split = methods.resolve_effective_k_split(effective_k=k, n0=n0, n1=n1)
    idx0 = methods.within_batch_knn_indices(x0, split.batch0_k)
    idx1 = methods.within_batch_knn_indices(x1, split.batch1_k)

    plain, _ = methods.four_block_directed_knn(matrix, n0=n0, n1=n1, effective_k=k)
    patched, _ = methods.four_block_directed_knn(
        matrix, n0=n0, n1=n1, effective_k=k, within_batch_indices=(idx0, idx1)
    )

    # The two within-batch corners now hold the data kNN, with batch 1 shifted to global indices.
    assert np.array_equal(patched[:n0, : split.batch0_k], idx0)
    assert np.array_equal(patched[n0:, split.batch0_k :], idx1 + n0)
    # The two cross-batch corners are untouched.
    assert np.array_equal(patched[:n0, split.batch0_k :], plain[:n0, split.batch0_k :])
    assert np.array_equal(patched[n0:, : split.batch0_k], plain[n0:, : split.batch0_k])

    # And the union still reproduces the graph the knndiag t-SNE arm embeds.
    mask, _ = methods.sparsify_four_block_top_k(matrix, n0=n0, n1=n1, effective_k=k, binarize=True)
    graph = mask.copy()
    graph[:n0, :n0] = methods._within_batch_knn_block(x0, split.batch0_k)
    graph[n0:, n0:] = methods._within_batch_knn_block(x1, split.batch1_k)
    adjacency = np.zeros_like(mask, dtype=bool)
    adjacency[np.arange(n0 + n1)[:, None], patched] = True
    union = adjacency | adjacency.T
    np.fill_diagonal(union, False)
    assert np.array_equal(union, graph > 0)


def test_within_batch_indices_shape_is_checked():
    """A width mismatch would silently misalign the corners, so it must raise."""
    rng = np.random.default_rng(3)
    matrix, _, _ = methods._build_four_block_parts(
        methods.compute_ot_plan(rng.normal(size=(20, 6)), rng.normal(size=(18, 6)), rel_reg=1 / 10)
    )
    with pytest.raises(ValueError, match="within-batch indices for batch 0"):
        methods.four_block_directed_knn(
            matrix,
            n0=20,
            n1=18,
            effective_k=10,
            within_batch_indices=(np.zeros((20, 2), int), np.zeros((18, 5), int)),
        )


def test_four_block_selection_is_invariant_to_transposing_the_input():
    """Selection must not depend on the summation order of the matrix product.

    Which of two entries a hair apart wins used to depend on it, which made the graph a property
    of the maths library rather than the data. Feeding the transpose is a cheap proxy: in exact
    arithmetic the matrix equals its transpose, so the same graph must come back.
    """
    rng = np.random.default_rng(4)
    matrix, _, _ = methods._build_four_block_parts(
        methods.compute_ot_plan(
            rng.normal(size=(40, 10)), rng.normal(size=(35, 10)) + 0.4, rel_reg=1 / 300
        )
    )
    a, _ = methods.four_block_directed_knn(matrix, n0=40, n1=35, effective_k=15)
    b, _ = methods.four_block_directed_knn(matrix.T, n0=40, n1=35, effective_k=15)
    assert np.array_equal(np.sort(a, axis=1), np.sort(b, axis=1))


@pytest.mark.parametrize(
    ("rel_reg", "expect_identical"),
    [(1.0, False), (1 / 300, True)],
    ids=["strong-reg-distinct", "weak-reg-collapses"],
)
def test_knndiag_and_selfotmedian_collapse_at_low_epsilon(rel_reg, expect_identical):
    """Check graph equality of kNN and self-OT diagonals at low regularization.

    Self-OT scores contain ``u_i u_j exp(-d_ij**2 / epsilon)``. A sharp kernel can
    make distance dominate scaling, producing the same selected neighbors as kNN.
    """
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(90, 12))
    x1 = rng.normal(size=(75, 12)) + 0.5
    kwargs = dict(effective_k=12, random_state=0, n_epochs=80, rel_reg=rel_reg)

    knndiag = methods.compute_ot_tsne_knndiag_embedding_from_arrays(x0, x1, **kwargs)
    selfot = methods.compute_ot_tsne_selfotmedian_embedding_from_arrays(x0, x1, **kwargs)

    identical = knndiag.tobytes() == selfot.tobytes()
    assert identical is expect_identical, (
        f"at rel_reg={rel_reg} the two variants are "
        f"{'identical' if identical else 'distinct'}, expected "
        f"{'identical' if expect_identical else 'distinct'}"
    )


def test_binarize_flag_does_not_reach_the_embedded_graph():
    """Documents a known wart so a future fix has something to change deliberately.

    normalize_ot_tsne_graph binarises unconditionally, so the binarize flag threaded through the OT
    entry points does not affect the graph that gets embedded -- it only changes an artifact that is
    never persisted. This asserts the *current* behavior, not the desired one; when the fix in
    OPEN_ITEMS.md lands, this test should be inverted rather than deleted.
    """
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=(60, 10))
    x1 = rng.normal(size=(50, 10)) + 0.4
    kwargs = dict(effective_k=10, random_state=0, n_epochs=60, rel_reg=1 / 10)

    on = methods.compute_ot_tsne_embedding_from_arrays(x0, x1, binarize=True, **kwargs)
    off = methods.compute_ot_tsne_embedding_from_arrays(x0, x1, binarize=False, **kwargs)
    assert on.tobytes() == off.tobytes(), (
        "binarize now changes the embedded graph -- if that is the intended fix, invert this test"
    )


# Unbalanced OT


def _uot_pair(seed=0, n0=40, n1=28):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n0, 5)), rng.normal(size=(n1, 5)) + 0.6


def test_balanced_transport_is_the_default_and_the_untouched_sinkhorn_call():
    """The whole change rests on this: nothing existing moved.

    Asserted against the POT call the old code made, not against a stored constant, so it keeps
    meaning if the test data changes. Bit-identical rather than close -- the balanced branch is
    literally the same call, and `sinkhorn_unbalanced(reg_m=inf)` is deliberately *not* used for
    it precisely because that would only be close.
    """
    import ot as pot

    x0, x1 = _uot_pair()
    dist = pot.dist(x0, x1)
    epsilon = float(np.median(dist)) * 0.1
    expected = pot.sinkhorn(
        np.ones(len(x0)) / len(x0),
        np.ones(len(x1)) / len(x1),
        dist,
        reg=epsilon,
        numItermax=int(1e4),
        stopThr=1e-12,
    )

    plan, artifacts = compute_ot_plan(x0, x1, rel_reg=0.1, return_artifacts=True)

    assert np.array_equal(plan, expected)
    assert artifacts["transport"] == methods.BALANCED_TRANSPORT
    # The unbalanced diagnostics must not appear on a balanced run, so a reader cannot mistake a
    # balanced cache for a relaxed one.
    assert "reg_m" not in artifacts and "plan_total_mass" not in artifacts


def test_unbalanced_plan_recovers_the_balanced_one_as_the_relaxation_vanishes():
    x0, x1 = _uot_pair()
    balanced = compute_ot_plan(x0, x1, rel_reg=0.1)
    relaxed = compute_ot_plan(x0, x1, rel_reg=0.1, transport="unbalanced", rel_reg_m=float("inf"))
    np.testing.assert_allclose(relaxed, balanced, atol=1e-12)


def test_unbalanced_plan_loses_mass_monotonically_as_the_relaxation_tightens():
    """The defining behavior: a finite reg_m lets the solver decline to transport."""
    x0, x1 = _uot_pair()
    masses = []
    for rel_reg_m in (10.0, 1.0, 0.1):
        _, artifacts = compute_ot_plan(
            x0,
            x1,
            rel_reg=0.1,
            transport="unbalanced",
            rel_reg_m=rel_reg_m,
            return_artifacts=True,
        )
        masses.append(artifacts["plan_total_mass"])
        assert artifacts["plan_row_mass"].shape == (len(x0),)
        assert artifacts["plan_col_mass"].shape == (len(x1),)
        # reg_m is absolute and rel_reg_m is the multiplier that produced it, exactly as
        # epsilon is to rel_reg.
        assert artifacts["reg_m"] == pytest.approx(artifacts["median_squared_distance"] * rel_reg_m)
    assert masses == sorted(masses, reverse=True), masses
    assert masses[0] < 1.0


def test_reg_m_overrides_rel_reg_m_the_way_epsilon_overrides_rel_reg():
    x0, x1 = _uot_pair()
    _, from_multiplier = compute_ot_plan(
        x0, x1, rel_reg=0.1, transport="unbalanced", rel_reg_m=0.5, return_artifacts=True
    )
    _, from_absolute = compute_ot_plan(
        x0,
        x1,
        rel_reg=0.1,
        transport="unbalanced",
        reg_m=from_multiplier["reg_m"],
        return_artifacts=True,
    )
    assert from_absolute["reg_m"] == from_multiplier["reg_m"]
    assert from_absolute["rel_reg_m"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"transport": "semi"}, "transport must be one of"),
        ({"transport": "unbalanced", "rel_reg_m": 0.0}, "rel_reg_m must be positive"),
        ({"transport": "unbalanced", "reg_m": -1.0}, "reg_m must be positive"),
    ],
)
def test_unbalanced_arguments_are_validated(kwargs, message):
    x0, x1 = _uot_pair()
    with pytest.raises(ValueError, match=message):
        compute_ot_plan(x0, x1, rel_reg=0.1, **kwargs)


def test_a_collapsed_unbalanced_plan_raises_instead_of_embedding_noise():
    """An underflowed kernel gives an all-zero plan, which the sparsifier would happily fill.

    The four-block matrix of a zero plan is all zeros, so top-k selects whatever argpartition
    breaks ties to and the embedding looks ordinary while meaning nothing. This is the guard that
    turns that into a failure.
    """
    x0, x1 = _uot_pair()
    with pytest.raises(ValueError, match="underflowed"):
        # Small enough that exp(-M / epsilon) flushes to zero across most of the cost matrix.
        compute_ot_plan(x0, x1, epsilon=1e-3, transport="unbalanced", rel_reg_m=1.0)
    # And a legitimately light plan is not mistaken for a collapsed one: its mass is far below 1
    # while every row and column still carries structure.
    _, artifacts = compute_ot_plan(
        x0, x1, rel_reg=1.0, transport="unbalanced", rel_reg_m=0.01, return_artifacts=True
    )
    assert artifacts["plan_total_mass"] < 1.0


def test_mass_aware_sparsifier_reduces_to_the_fixed_budget_one():
    """Uniform budgets must reproduce ``sparsify_four_block_top_k`` exactly, ties included.

    Exactly, not approximately: the candidate set is taken with the same ``argpartition`` call, so
    the two cannot disagree on the many exact zeros a clipped four-block matrix carries.
    """
    x0, x1 = _uot_pair()
    plan = compute_ot_plan(x0, x1, rel_reg=0.1)
    matrix = build_ot_tsne_four_block_matrix(plan)
    n0, n1 = plan.shape
    split = resolve_effective_k_split(15, n0, n1)

    expected, _ = sparsify_four_block_top_k(matrix, n0=n0, n1=n1, effective_k=15, binarize=True)
    got, _ = sparsify_four_block_top_k_mass_aware(
        matrix,
        n0=n0,
        n1=n1,
        effective_k=15,
        row_budgets=np.full(n0, split.batch1_k),
        col_budgets=np.full(n1, split.batch0_k),
        binarize=True,
    )
    assert np.array_equal(expected, got)


def test_mass_aware_sparsifier_cuts_cross_batch_edges_for_a_starved_group():
    """Require reduced cross-batch connectivity for a low-mass group and fixed within-batch edges.

    Test a group because row/column union permits other cells to select a cell even
    when its own budget is reduced.
    """
    x0, x1 = _uot_pair()
    plan = compute_ot_plan(x0, x1, rel_reg=0.1)
    matrix = build_ot_tsne_four_block_matrix(plan)
    n0, n1 = plan.shape
    split = resolve_effective_k_split(15, n0, n1)

    full, _ = sparsify_four_block_top_k_mass_aware(
        matrix,
        n0=n0,
        n1=n1,
        effective_k=15,
        row_budgets=np.full(n0, split.batch1_k),
        col_budgets=np.full(n1, split.batch0_k),
        binarize=True,
    )
    # Starve the first half of batch 1, as an unbalanced plan starves a whole cell type.
    starved_group = np.arange(n1) < n1 // 2
    starved_col = np.where(starved_group, 1, split.batch0_k)
    starved, _ = sparsify_four_block_top_k_mass_aware(
        matrix,
        n0=n0,
        n1=n1,
        effective_k=15,
        row_budgets=np.full(n0, split.batch1_k),
        col_budgets=starved_col,
        binarize=True,
    )

    # Batch-1 cells live at joint rows n0:, and their cross-batch edges are columns :n0.
    full_degree = (full[n0:, :n0] > 0).sum(axis=1)
    starved_degree = (starved[n0:, :n0] > 0).sum(axis=1)
    assert starved_degree[starved_group].mean() < full_degree[starved_group].mean()
    # And the cells left alone keep theirs, so the effect is targeted rather than global.
    assert starved_degree[~starved_group].mean() >= full_degree[~starved_group].mean()
    # The within-batch blocks keep the fixed quota, so they must not have moved.
    assert np.array_equal(starved[:n0, :n0], full[:n0, :n0])
    assert np.array_equal(starved[n0:, n0:], full[n0:, n0:])


def test_mass_budgets_track_the_share_of_mass_a_cell_kept():
    """Ratio 1.0 is the uniform share and earns the full budget; below it the budget falls.

    Four cells holding 0.4/0.3/0.2/0.1 of the mass against a uniform share of 0.25, so the ratios
    are 1.6/1.2/0.8/0.4 and the budgets 8 (capped), 8 (capped), 6, 3.
    """
    mass = np.array([0.4, 0.3, 0.2, 0.1])
    budgets = resolve_mass_budgets(mass, k=8, k_min=1)
    np.testing.assert_array_equal(budgets, [8, 8, 6, 3])
    # Scale-free: halving every entry changes nothing, because only the redistribution matters.
    np.testing.assert_array_equal(resolve_mass_budgets(mass / 2.0, k=8, k_min=1), budgets)
    # A cell with no mass is floored rather than isolated.
    assert resolve_mass_budgets(np.array([1.0, 0.0]), k=8, k_min=1)[1] == 1


def test_uot_topk_at_the_balanced_limit_reproduces_ot_tsne():
    """The sweep's validation anchor, end to end rather than at the plan alone.

    A tolerance rather than bit-equality, and the reason is worth stating: `ot.sinkhorn` and
    `sinkhorn_unbalanced(reg_m=inf)` solve the same problem by different iterations and agree to
    about 1e-13. The graph is a rank-based selection so that normally vanishes entirely, but on a
    flat enough plan it can flip a near-tied edge -- measured at 16 of 88,502 on `hematopoietic` at
    `rel_reg` 1/300, which t-SNE then amplifies. See METHODS.md.
    """
    x0, x1 = _uot_pair()
    shared = dict(rel_reg=0.1, effective_k=15, random_state=0, n_epochs=60)
    balanced = compute_ot_tsne_embedding_from_arrays(x0, x1, **shared)
    relaxed = compute_ot_tsne_uot_topk_embedding_from_arrays(
        x0, x1, rel_reg_m=float("inf"), **shared
    )
    np.testing.assert_allclose(relaxed, balanced, atol=1e-8)


def test_uot_variants_return_the_diagnostics_the_experiment_records():
    x0, x1 = _uot_pair()
    shared = dict(rel_reg=0.1, rel_reg_m=0.1, effective_k=15, random_state=0, n_epochs=60)
    _, topk = compute_ot_tsne_uot_topk_embedding_from_arrays(
        x0, x1, return_artifacts=True, **shared
    )
    _, mass = compute_ot_tsne_uot_mass_embedding_from_arrays(
        x0, x1, return_artifacts=True, **shared
    )
    for artifacts in (topk, mass):
        assert artifacts["transport"] == methods.UNBALANCED_TRANSPORT
        assert 0.0 < artifacts["plan_total_mass"] < 1.0
        assert artifacts["rel_reg_m"] == pytest.approx(0.1)
        assert artifacts["plan_row_mass"].shape == (len(x0),)
    # Only the mass-aware arm carries budgets, and they must be within the resolved quotas.
    assert "batch0_mass_budgets" not in topk
    split = resolve_effective_k_split(15, len(x0), len(x1))
    assert mass["batch0_mass_budgets"].max() <= split.batch1_k
    assert mass["batch1_mass_budgets"].max() <= split.batch0_k


def test_mass_aware_budgets_are_refused_by_the_umap_backend():
    """UMAP's neighbor list is a fixed-width array; variable degrees cannot be expressed in it."""
    x0, x1 = _uot_pair()
    with pytest.raises(ValueError, match="only supported by the t-SNE backend"):
        compute_ot_tsne_uot_mass_embedding_from_arrays(
            x0, x1, rel_reg=0.1, rel_reg_m=0.1, effective_k=15, n_epochs=60, backend="umap"
        )


# ot-zerodiag: identically zero within-batch blocks, and the pins that the three existing
# diagonal variants are untouched by it.

ZERODIAG_N0, ZERODIAG_N1, ZERODIAG_K = 24, 36, 15
EXISTING_DIAGONAL_ARMS = ("ot", "ot-knndiag", "ot-gramdiag", "ot-selfotmedian")
SHARED_REL_REG = (1.0, 1 / 10, 1 / 100, 1 / 300)


def _zerodiag_inputs(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(ZERODIAG_N0, 5)), rng.normal(size=(ZERODIAG_N1, 5)) + 0.5


def _reference_variant_graph(method, x0, x1, rel_reg, effective_k=ZERODIAG_K):
    """Rebuild a variant's documented block assembly, quotas, and sparsification.

    Use production selection primitives for implementation-defined argpartition ties;
    this reference checks composition rather than imposing another tie-breaking rule.
    """
    plan = compute_ot_plan(x0, x1, rel_reg=rel_reg)
    matrix, _, _ = methods._build_four_block_parts(plan)
    n0, n1 = plan.shape
    # The cross block is the plan itself, not its SVD rebuild. At small epsilon most plan entries
    # are far below the rebuild's absolute rounding error, so the two rank them differently.
    matrix[:n0, n0:] = plan
    matrix[n0:, :n0] = plan.T

    if method in {"ot-gramdiag", "ot-selfotmedian"}:
        # Weighted diagonals, written BEFORE the sparsifier so the whole matrix passes through
        # the one selection -- METHODS.md, "the three diagonal variants".
        if method == "ot-gramdiag":
            diag0, diag1 = plan @ plan.T, plan.T @ plan
        else:
            diag0 = compute_ot_plan(x0, x0, rel_reg=rel_reg)
            diag1 = compute_ot_plan(x1, x1, rel_reg=rel_reg)
        diag0 = np.array(diag0, dtype=float, copy=True)
        diag1 = np.array(diag1, dtype=float, copy=True)
        np.fill_diagonal(diag0, 0.0)
        np.fill_diagonal(diag1, 0.0)
        matrix[:n0, :n0] = diag0
        matrix[n0:, n0:] = diag1

    sparse, split = sparsify_four_block_top_k(
        matrix, n0=n0, n1=n1, effective_k=effective_k, binarize=True
    )

    if method == "ot-knndiag":
        # Binary diagonals, written AFTER the sparsifier, which would otherwise fabricate edges
        # on a binary block. Same per-batch budget the split allocated.
        sparse[:n0, :n0] = methods._adjacency_from_indices(
            methods.within_batch_knn_indices(x0, split.batch0_k), n0
        )
        sparse[n0:, n0:] = methods._adjacency_from_indices(
            methods.within_batch_knn_indices(x1, split.batch1_k), n1
        )

    return normalize_ot_tsne_graph(sparse)


def test_zerodiag_four_block_has_exactly_zero_diagonal_blocks():
    """The block form A = [[0, P], [P.T, 0]] must hold before sparsification, exactly.

    Exact equality, not a tolerance: the blocks are assigned zero, not computed small. And the
    two cross blocks must be bit-identical to the default builder's, so the only difference
    between `ot` and `ot-zerodiag` at this stage is the removal.
    """
    x0, x1 = _zerodiag_inputs()
    plan = compute_ot_plan(x0, x1, rel_reg=1 / 10)
    n0, n1 = plan.shape

    zerodiag = build_ot_tsne_zerodiag_four_block_matrix(plan)
    default = build_ot_tsne_four_block_matrix(plan)

    np.testing.assert_array_equal(zerodiag[:n0, :n0], np.zeros((n0, n0)))
    np.testing.assert_array_equal(zerodiag[n0:, n0:], np.zeros((n1, n1)))
    np.testing.assert_array_equal(zerodiag[:n0, n0:], default[:n0, n0:])
    np.testing.assert_array_equal(zerodiag[n0:, :n0], default[n0:, :n0])
    # The default builder's diagonal is genuinely populated, so the test above is not vacuous.
    assert (default[:n0, :n0] > 0.0).any()
    assert (default[n0:, n0:] > 0.0).any()


def test_cross_batch_only_declines_the_diagonal_where_the_generic_sparsifier_fabricates():
    """Check that cross-only selection excludes zero blocks that generic top-k would populate.

    Binarization turns selected tied zeros into edges, so zero-valued blocks must be
    excluded from selection rather than merely assigned zero scores.
    """
    x0, x1 = _zerodiag_inputs()
    plan = compute_ot_plan(x0, x1, rel_reg=1 / 10)
    n0, n1 = plan.shape
    matrix = build_ot_tsne_zerodiag_four_block_matrix(plan)

    generic, _ = sparsify_four_block_top_k(
        matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True
    )
    fabricated = int((generic[:n0, :n0] > 0.0).sum() + (generic[n0:, n0:] > 0.0).sum())
    assert fabricated > 0, (
        "the generic sparsifier no longer fabricates edges on an all-zero block; if that is a "
        "deliberate fix, this test should be inverted rather than deleted"
    )

    cross_only = sparsify_four_block_cross_batch_only(
        matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True
    )
    assert int((cross_only[:n0, :n0] > 0.0).sum()) == 0
    assert int((cross_only[n0:, n0:] > 0.0).sum()) == 0
    # The cross blocks are not empty, so "no within-batch edges" is not "no edges".
    assert (cross_only[:n0, n0:] > 0.0).any()


def test_zerodiag_graph_degree_is_all_cross_batch():
    """Every point keeps effective_k neighbors, every one of them in the other batch.

    The degree floor is `>=`, not `==`, for the same reason it is in the base method: the mask is
    the union of a row pass and a column pass, so a point chosen by others gains edges.
    """
    x0, x1 = _zerodiag_inputs()
    plan = compute_ot_plan(x0, x1, rel_reg=1 / 10)
    n0, n1 = plan.shape
    matrix = build_ot_tsne_zerodiag_four_block_matrix(plan)
    graph = normalize_ot_tsne_graph(
        sparsify_four_block_cross_batch_only(
            matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True
        )
    )

    adjacency = graph > 0.0
    assert not adjacency[:n0, :n0].any()
    assert not adjacency[n0:, n0:].any()
    degrees = adjacency.sum(axis=1)
    assert degrees.min() >= ZERODIAG_K
    # Each row's whole degree sits in the opposite batch's columns.
    np.testing.assert_array_equal(adjacency[:n0, n0:].sum(axis=1), degrees[:n0])
    np.testing.assert_array_equal(adjacency[n0:, :n0].sum(axis=1), degrees[n0:])
    np.testing.assert_allclose(graph.sum(), 1.0)


def test_zerodiag_rejects_effective_k_above_the_smaller_batch():
    """The ceiling is min(n0, n1), not the two capacities summed.

    A cross-batch-only selection draws a point's whole budget from one block, whose width is the
    size of the opposite batch. `resolve_effective_k_split`'s `(n0-1) + (n1-1)` would admit a
    budget no batch-0 row could fill.
    """
    assert resolve_cross_batch_k(effective_k=24, n0=24, n1=36) == 24
    with pytest.raises(ValueError, match="exceeds the cross-batch neighbours"):
        resolve_cross_batch_k(effective_k=25, n0=24, n1=36)
    # The split-based ceiling would have allowed it, which is the point of checking separately.
    assert resolve_effective_k_split(effective_k=25, n0=24, n1=36).effective_k == 25
    with pytest.raises(ValueError, match="effective_k must be positive"):
        resolve_cross_batch_k(effective_k=0, n0=24, n1=36)


@pytest.mark.parametrize("rel_reg", SHARED_REL_REG)
def test_cross_batch_directed_knn_union_equals_the_cross_only_mask(rel_reg):
    """The UMAP arm must embed the same graph the t-SNE arm does.

    `sparsify_four_block_cross_batch_only` builds the union of a row and a column pass;
    `cross_batch_directed_knn` returns only the row pass, because UMAP wants a fixed-width
    neighbor list. Its fuzzy union reconstructs the mask only if D | D.T equals it, and that is
    not guaranteed a priori: the two are separate argpartition calls over a matrix clipped at
    zero, so they could break ties differently. Both go through `_row_topk_indices` to stop that.
    """
    x0, x1 = _zerodiag_inputs()
    plan = compute_ot_plan(x0, x1, rel_reg=rel_reg)
    n0, n1 = plan.shape
    matrix = build_ot_tsne_zerodiag_four_block_matrix(plan)

    mask = sparsify_four_block_cross_batch_only(
        matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True
    )
    directed = methods.cross_batch_directed_knn(matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K)
    assert directed.shape == (n0 + n1, ZERODIAG_K)
    # Batch-0 rows point only into batch 1 and vice versa, so no point is ever its own neighbor.
    assert (directed[:n0] >= n0).all()
    assert (directed[n0:] < n0).all()

    adjacency = np.zeros((n0 + n1, n0 + n1), dtype=bool)
    adjacency[np.arange(n0 + n1)[:, None], directed] = True
    union = adjacency | adjacency.T
    np.fill_diagonal(union, False)
    assert np.array_equal(union, mask > 0.0)


@pytest.mark.parametrize("method", EXISTING_DIAGONAL_ARMS)
@pytest.mark.parametrize("rel_reg", SHARED_REL_REG)
def test_reference_rebuild_matches_each_existing_variant_graph(method, rel_reg):
    """Require each public variant to match its reference assembly under identical embedding
    settings.
    """
    entry = {
        "ot": compute_ot_tsne_embedding_from_arrays,
        "ot-knndiag": compute_ot_tsne_knndiag_embedding_from_arrays,
        "ot-gramdiag": compute_ot_tsne_gramdiag_embedding_from_arrays,
        "ot-selfotmedian": methods.compute_ot_tsne_selfotmedian_embedding_from_arrays,
    }[method]
    x0, x1 = _zerodiag_inputs()

    embedding, artifacts = entry(
        x0,
        x1,
        rel_reg=rel_reg,
        effective_k=ZERODIAG_K,
        random_state=5,
        n_epochs=25,
        return_artifacts=True,
    )
    reference_graph = _reference_variant_graph(method, x0, x1, rel_reg)
    # The entry points hand t-SNE a sparse graph. Converting the dense reference to sparse puts
    # both through the same path, so equal bytes mean equal edges and equal weights (1 / |E|).
    reference = run_tsne_from_graph(
        sp.csr_matrix(reference_graph),
        random_state=5,
        n_epochs=25,
        init_scale=10000.0,
        init="random",
    )
    assert embedding.tobytes() == reference.tobytes()
    # `ot` is the one arm that still returns its graph, so compare that directly too.
    if method == "ot":
        np.testing.assert_array_equal(artifacts["graph"].toarray(), reference_graph)


@pytest.mark.parametrize("method", EXISTING_DIAGONAL_ARMS)
def test_the_existing_variants_never_hand_an_all_zero_block_to_the_sparsifier(method):
    """The guaranteed-fabrication case is reachable only through ot-zerodiag.

    Stated carefully, because the mechanism is neither new nor exclusive: the four-block matrix is
    clipped at zero, so a sparse row can already hold more exact ties than its budget, and these
    four arms can already select tied zeros. What an *identically* zero block changes is that the
    whole block is ties and fabrication becomes certain. This asserts the four existing arms never
    reach that state -- both diagonal quadrants carry real values going in, and real edges coming
    out -- which is why they need no guard and get none.
    """
    x0, x1 = _zerodiag_inputs()
    graph = _reference_variant_graph(method, x0, x1, rel_reg=1 / 10)
    n0 = ZERODIAG_N0
    assert (graph[:n0, :n0] > 0.0).any(), f"{method} lost its batch-0 within-batch edges"
    assert (graph[n0:, n0:] > 0.0).any(), f"{method} lost its batch-1 within-batch edges"

    zerodiag_graph = normalize_ot_tsne_graph(
        sparsify_four_block_cross_batch_only(
            build_ot_tsne_zerodiag_four_block_matrix(compute_ot_plan(x0, x1, rel_reg=1 / 10)),
            n0=n0,
            n1=ZERODIAG_N1,
            effective_k=ZERODIAG_K,
            binarize=True,
        )
    )
    assert not (zerodiag_graph[:n0, :n0] > 0.0).any()
    assert not (zerodiag_graph[n0:, n0:] > 0.0).any()


def test_the_generic_sparsifier_output_is_unchanged():
    """A direct pin on `sparsify_four_block_top_k`, independent of any OT machinery.

    Two halves. On a tie-free matrix an argsort reference is genuinely independent of the
    production argpartition path, so the mask can be compared entry for entry. On a tie-heavy
    matrix -- a diagonal block clipped at zero, which is what the real pipeline produces -- the
    selection is only defined up to tie-breaking, so what is pinned there is the structural
    property that survives any tie choice: each row keeps at least its block's budget.
    """
    size, n0, n1, k = 9, 4, 5, 3
    # Use symmetric scores distinct within each row and column so reference ranking has no ties.
    upper = np.triu(np.arange(1, size * size + 1, dtype=float).reshape(size, size), 1)
    matrix = upper + upper.T
    np.fill_diagonal(matrix, matrix.max() + 100.0)
    for index in range(size):
        row = np.delete(matrix[index], index)
        assert len(np.unique(row)) == row.size, "the tie-free half needs distinct values per row"

    produced, split = sparsify_four_block_top_k(matrix, n0=n0, n1=n1, effective_k=k, binarize=True)

    # Independent reference: full argsort per row and per column of each block, no argpartition.
    symmetric = (matrix + matrix.T) / 2.0
    batch_ks = (split.batch0_k, split.batch1_k)
    slices = (slice(0, n0), slice(n0, n0 + n1))
    expected = np.zeros_like(matrix, dtype=bool)
    for row_batch, row_slice in enumerate(slices):
        for col_batch, col_slice in enumerate(slices):
            block = np.array(symmetric[row_slice, col_slice], copy=True)
            if row_batch == col_batch:
                np.fill_diagonal(block, -np.inf)
            row_k, col_k = batch_ks[col_batch], batch_ks[row_batch]
            block_mask = np.zeros(block.shape, dtype=bool)
            if row_k:
                for i in range(block.shape[0]):
                    block_mask[i, np.argsort(-block[i])[:row_k]] = True
            if col_k:
                for j in range(block.shape[1]):
                    block_mask[np.argsort(-block[:, j])[:col_k], j] = True
            if row_batch == col_batch:
                np.fill_diagonal(block_mask, False)
            expected[row_slice, col_slice] = block_mask
    np.fill_diagonal(expected, False)
    np.testing.assert_array_equal(produced > 0.0, expected)

    # Small regularization produces clipped-zero ties; verify budgets independently of tied choices.
    x0, x1 = _zerodiag_inputs()
    clipped = build_ot_tsne_four_block_matrix(compute_ot_plan(x0, x1, rel_reg=1 / 100))
    off_diagonal_zeros = int((clipped == 0.0).sum()) - clipped.shape[0]
    assert off_diagonal_zeros > ZERODIAG_K, (
        f"expected a genuinely clipped matrix, found {off_diagonal_zeros} clipped entries"
    )
    sparse, real_split = sparsify_four_block_top_k(
        clipped, n0=ZERODIAG_N0, n1=ZERODIAG_N1, effective_k=ZERODIAG_K, binarize=True
    )
    adjacency = sparse > 0.0
    assert np.all(adjacency[:, :ZERODIAG_N0].sum(axis=1) >= real_split.batch0_k)
    assert np.all(adjacency[:, ZERODIAG_N0:].sum(axis=1) >= real_split.batch1_k)


# GPU / torch-backend execution of the POT solver

CUDA_AVAILABLE = False
try:  # pragma: no cover - depends on the machine
    import torch as _torch

    CUDA_AVAILABLE = bool(_torch.cuda.is_available())
except Exception:  # pragma: no cover - torch is a hard dependency, but never fail collection
    pass

requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="no CUDA device on this machine")

#: Balanced and unbalanced, so both POT solvers are covered by every agreement check below.
TRANSPORT_CASES = (
    pytest.param({}, id="balanced"),
    pytest.param({"transport": "unbalanced", "rel_reg_m": 0.1}, id="unbalanced"),
)


def _device_pair(x0, x1, device, rel_reg, **kwargs):
    """The same plan computed on numpy and on ``device``, plus the graph each one produces.

    The graph is the point. The plan feeds a rank-based top-k selection that is then binarised, so
    agreement on the plan is necessary and agreement on the *selected edges* is what actually says
    the two devices produce the same science.
    """
    reference = compute_ot_plan(x0, x1, rel_reg=rel_reg, **kwargs)
    candidate = compute_ot_plan(x0, x1, rel_reg=rel_reg, device=device, **kwargs)
    n0, n1 = reference.shape
    graphs = [
        sparsify_four_block_top_k(
            build_ot_tsne_four_block_matrix(plan), n0=n0, n1=n1, effective_k=15, binarize=True
        )[0]
        for plan in (reference, candidate)
    ]
    return reference, candidate, graphs[0], graphs[1]


@pytest.mark.parametrize("kwargs", TRANSPORT_CASES)
@pytest.mark.parametrize("rel_reg", [1.0, 0.1, 0.01, 1 / 300])
def test_torch_backend_reproduces_the_numpy_plan(kwargs, rel_reg):
    """Compare NumPy and torch-cpu plans across regularization and both solvers.

    The CPU tensor path exercises backend dispatch and conversion without requiring CUDA.
    Tolerance allows floating-point variation while detecting precision loss.
    """
    x0, x1 = _uot_pair(n0=60, n1=45)
    reference, candidate, _, _ = _device_pair(x0, x1, "torch-cpu", rel_reg, **kwargs)

    assert isinstance(candidate, np.ndarray) and candidate.dtype == np.float64
    np.testing.assert_allclose(candidate, reference, rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("kwargs", TRANSPORT_CASES)
@pytest.mark.parametrize("rel_reg", [1.0, 0.1])
def test_torch_backend_reproduces_the_numpy_graph(kwargs, rel_reg):
    """Require identical selected edges from NumPy and torch-cpu where rankings are well determined.

    Exclude small-epsilon cases dominated by clipped-zero ties, where backend rounding
    can legitimately change implementation-defined argpartition choices.
    """
    x0, x1 = _uot_pair(n0=60, n1=45)
    _, _, graph_reference, graph_candidate = _device_pair(x0, x1, "torch-cpu", rel_reg, **kwargs)
    assert np.array_equal(graph_candidate, graph_reference)


@pytest.mark.parametrize("kwargs", TRANSPORT_CASES)
@requires_cuda
def test_cuda_reproduces_the_numpy_plan_and_graph(kwargs):  # pragma: no cover - needs a GPU
    """The same assertion against a real GPU, wherever one exists."""
    x0, x1 = _uot_pair(n0=60, n1=45)
    # rel_reg 0.1: the plan is compared at a tight tolerance and the graph exactly, for the reason
    # given on the torch-cpu graph test above -- at smaller epsilon the selection is tie-dominated
    # on this input and would be testing argpartition rather than the device.
    reference, candidate, graph_reference, graph_candidate = _device_pair(
        x0, x1, "cuda", 0.1, **kwargs
    )
    assert isinstance(candidate, np.ndarray) and candidate.dtype == np.float64
    np.testing.assert_allclose(candidate, reference, rtol=1e-9, atol=0.0)
    assert np.array_equal(graph_candidate, graph_reference)


def test_the_default_device_keeps_numpy_off_the_torch_backend():
    """The default must be the historical path, not a torch path that lands on the CPU."""
    x0, x1 = _uot_pair()
    _, artifacts = compute_ot_plan(x0, x1, rel_reg=0.1, return_artifacts=True)
    assert artifacts["ot_device"] == methods.CPU_DEVICE == "cpu"
    assert isinstance(artifacts["cost_matrix"], np.ndarray)


@pytest.mark.skipif(CUDA_AVAILABLE, reason="this asserts the no-GPU behaviour")
def test_cuda_without_a_device_raises_rather_than_falling_back():
    """Silence here would be the worst outcome: the run would be recorded as a GPU run."""
    x0, x1 = _uot_pair()
    with pytest.raises(RuntimeError, match="no CUDA device"):
        compute_ot_plan(x0, x1, rel_reg=0.1, device="cuda")


def test_mps_is_refused_and_says_why():
    """MPS cannot hold float64, and float32 changes the plan completely at small epsilon."""
    x0, x1 = _uot_pair()
    with pytest.raises(ValueError, match="float64"):
        compute_ot_plan(x0, x1, rel_reg=0.1, device="mps")


def test_an_unknown_device_is_refused():
    x0, x1 = _uot_pair()
    with pytest.raises(ValueError, match="device must be one of"):
        compute_ot_plan(x0, x1, rel_reg=0.1, device="gpu")


def test_the_on_device_median_matches_numpys_definition():
    """``torch.median`` is not ``np.median``: it takes the lower middle value, numpy averages.

    The median sets ``epsilon``, so taking the torch one would move the regularization on every
    cost matrix with an even number of entries -- which is most of them. This pins the definition
    rather than the implementation, by checking a case where the two genuinely disagree.
    """
    import ot as pot
    import torch

    rng = np.random.default_rng(0)
    # 40 x 30 = 1200 entries, an even count, which is where the definitions part company.
    dist = pot.dist(rng.normal(size=(40, 4)), rng.normal(size=(30, 4)))
    tensor = torch.as_tensor(dist, dtype=torch.float64)

    assert methods._median_squared_distance(dist, None) == np.median(dist)
    assert methods._median_squared_distance(tensor, torch) == np.median(dist)
    assert float(tensor.median()) != np.median(dist), (
        "picked a case where they agree; choose another"
    )


@pytest.mark.parametrize(
    "entry_point, extra",
    [
        ("compute_ot_tsne_embedding_from_arrays", {}),
        ("compute_ot_tsne_knndiag_embedding_from_arrays", {}),
        ("compute_ot_tsne_selfotmedian_embedding_from_arrays", {}),
        ("compute_ot_tsne_gramdiag_embedding_from_arrays", {}),
        ("compute_ot_tsne_uot_topk_embedding_from_arrays", {"rel_reg_m": 0.1}),
        ("compute_ot_tsne_uot_mass_embedding_from_arrays", {"rel_reg_m": 0.1}),
        # Both zero-diagonal variants must resolve and record the OT device.
        ("compute_ot_tsne_zerodiag_embedding_from_arrays", {}),
        ("compute_ot_tsne_zerodiag_split_embedding_from_arrays", {}),
    ],
)
def test_every_ot_entry_point_takes_the_device_and_records_it(entry_point, extra):
    """The flag has to reach the experiments without anyone editing implementation code.

    ``torch_device`` is the key ``run_task`` copies into the task file's ``device`` field, and
    ``paper/runtime.py`` refuses to pool runtime rows that disagree on it, so a GPU run that failed
    to record itself would be silently averaged against CPU runs.
    """
    x0, x1 = _uot_pair(n0=50, n1=40)
    shared = dict(rel_reg=0.1, effective_k=15, random_state=0, n_epochs=60, **extra)
    function = getattr(methods, entry_point)

    reference = function(x0, x1, **shared)
    candidate, artifacts = function(x0, x1, device="torch-cpu", return_artifacts=True, **shared)

    assert artifacts["ot_device"] == "torch-cpu"
    assert artifacts["torch_device"] == "torch-cpu"
    np.testing.assert_allclose(candidate, reference, rtol=1e-6, atol=1e-6)


# OR vs AND symmetrization: shared kNN against mutual kNN, on the same directed selection


def _directed_adjacency(indices, n):
    """The directed selection as a boolean square, self-loops removed."""
    adjacency = np.zeros((n, n), dtype=bool)
    adjacency[np.arange(n)[:, None], indices] = True
    np.fill_diagonal(adjacency, False)
    return adjacency


def test_one_way_edges_survive_or_and_are_dropped_by_and():
    """Check union and mutual selection on symmetric, untied scores.

    At k=1, choices are 0->1, 1->0, 2->0, 3->2: only {0, 1} is reciprocal.
    """
    block = np.array(
        [
            [0.0, 9.0, 5.0, 0.1],
            [9.0, 0.0, 0.1, 0.1],
            [5.0, 0.1, 0.0, 0.2],
            [0.1, 0.1, 0.2, 0.0],
        ]
    )
    or_mask = methods._row_or_col_topk_mask(block, row_k=1, col_k=1, exclude_diagonal=True)
    and_mask = methods._mutual_topk_mask(block, row_k=1, col_k=1, exclude_diagonal=True)

    # Reciprocated: both rules keep it.
    assert or_mask[0, 1] and or_mask[1, 0]
    assert and_mask[0, 1] and and_mask[1, 0]

    # One-way. 2 picks 0, but 0 picks 1. OR keeps the pair, AND drops it -- the whole distinction.
    assert or_mask[2, 0] and or_mask[0, 2]
    assert not and_mask[2, 0] and not and_mask[0, 2]

    # One-way the other way round, to show the rule is not about which index is larger.
    assert or_mask[3, 2] and or_mask[2, 3]
    assert not and_mask[3, 2] and not and_mask[2, 3]

    # AND kept exactly the reciprocated pair and nothing else.
    assert and_mask.sum() == 2
    assert np.array_equal(and_mask, and_mask.T)


def test_mutual_mask_is_symmetric_and_a_subset_of_the_or_mask():
    """Three structural properties, over a range of tie densities.

    The tie sweep matters: the four-block matrix is clipped at zero, so real blocks hold far more
    exact ties than their budget, and an intersection is where differing tie-breaks would show.
    """
    for trial in range(12):
        rng = np.random.default_rng(trial)
        block = rng.random((18, 14))
        # 0.0, ~0.5 and ~0.9 of the block exactly zero.
        block[block < (trial % 3) * 0.45] = 0.0

        or_mask = methods._row_or_col_topk_mask(block, row_k=4, col_k=5)
        and_mask = methods._mutual_topk_mask(block, row_k=4, col_k=5)

        assert np.array_equal(and_mask, and_mask & or_mask), "AND is not a subset of OR"
        assert and_mask.sum() <= or_mask.sum()
        # Symmetry is a statement about the square graph, not about a rectangular block, so it is
        # asserted on the assembled four-block mask in the test below.


@pytest.mark.parametrize("rel_reg", SHARED_REL_REG)
def test_mutual_mask_equals_the_directed_selection_intersected_with_its_transpose(rel_reg):
    """Require mutual masks to equal ``D & D.T`` for the same directed selections.

    Sweep regularization to expose inconsistent tie-breaking near flat scores.
    """
    x0, x1 = _zerodiag_inputs()
    n0, n1 = x0.shape[0], x1.shape[0]
    matrix, _, _ = _build_four_block_parts(compute_ot_plan(x0, x1, rel_reg=rel_reg))

    or_mask, or_split = sparsify_four_block_top_k(
        matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True
    )
    and_mask, and_split = sparsify_four_block_top_k(
        matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True, graph_symmetrization="and"
    )
    directed, directed_split = methods.four_block_directed_knn(
        matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K
    )
    adjacency = _directed_adjacency(directed, n0 + n1)

    # The neighbor budget is untouched by the switch: same split, from the same resolver.
    assert or_split == and_split == directed_split

    or_mask = or_mask > 0
    and_mask = and_mask > 0
    expected = adjacency & adjacency.T
    np.fill_diagonal(expected, False)
    np.testing.assert_array_equal(and_mask, expected)
    np.testing.assert_array_equal(or_mask, adjacency | adjacency.T)

    # The three properties that make it a mutual graph rather than merely a smaller one.
    assert np.array_equal(and_mask, and_mask.T), "the mutual mask must be symmetric"
    assert not (and_mask & ~or_mask).any(), "AND created an edge OR does not have"
    assert not (and_mask & ~(adjacency | adjacency.T)).any(), "AND created an undirected edge"


@pytest.mark.parametrize("rel_reg", SHARED_REL_REG)
def test_cross_batch_only_mutual_mask_equals_its_directed_intersection(rel_reg):
    """The same pair of invariants for the ot-zerodiag sparsifier.

    Adding the switch here changes no neighbor budget: ``resolve_cross_batch_k`` validates a
    ceiling, and whether ``effective_k`` fits in the opposite batch does not depend on how the two
    directed selections are later combined. Asserted below by comparing the resolved budget.
    """
    x0, x1 = _zerodiag_inputs()
    n0, n1 = x0.shape[0], x1.shape[0]
    matrix = build_ot_tsne_zerodiag_four_block_matrix(compute_ot_plan(x0, x1, rel_reg=rel_reg))

    or_mask = (
        sparsify_four_block_cross_batch_only(
            matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True
        )
        > 0
    )
    and_mask = (
        sparsify_four_block_cross_batch_only(
            matrix,
            n0=n0,
            n1=n1,
            effective_k=ZERODIAG_K,
            binarize=True,
            graph_symmetrization="and",
        )
        > 0
    )
    adjacency = _directed_adjacency(
        methods.cross_batch_directed_knn(matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K), n0 + n1
    )

    expected = adjacency & adjacency.T
    np.fill_diagonal(expected, False)
    np.testing.assert_array_equal(and_mask, expected)
    np.testing.assert_array_equal(or_mask, adjacency | adjacency.T)
    assert np.array_equal(and_mask, and_mask.T)
    assert not (and_mask & ~or_mask).any()
    # Still no within-batch edges: the switch changes the combination, not which blocks are offered.
    assert not and_mask[:n0, :n0].any() and not and_mask[n0:, n0:].any()
    # The budget is unchanged -- same ceiling, same resolved value.
    assert resolve_cross_batch_k(effective_k=ZERODIAG_K, n0=n0, n1=n1) == ZERODIAG_K


def test_a_mutual_graph_survives_normalization_instead_of_being_re_ored():
    """``normalize_ot_tsne_graph`` ends with ``graph | graph.T``, which would undo a mutual mask.

    It is a no-op on a symmetric input, which is why the mutual mask is built to be symmetric by
    construction rather than by luck. The second half of this test is the counterfactual: an
    asymmetric mask really is widened back towards the union, silently, so "symmetric" is the
    property the whole feature rests on and not a detail.
    """
    x0, x1 = _zerodiag_inputs()
    n0, n1 = x0.shape[0], x1.shape[0]
    matrix, _, _ = _build_four_block_parts(compute_ot_plan(x0, x1, rel_reg=1.0))
    and_mask, _ = sparsify_four_block_top_k(
        matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True, graph_symmetrization="and"
    )

    graph = normalize_ot_tsne_graph(and_mask)
    np.testing.assert_array_equal(graph > 0.0, and_mask > 0.0)

    # The counterfactual. Break symmetry in one triangle and the normalizer restores the union.
    lopsided = np.array(and_mask, dtype=float, copy=True)
    or_mask, _ = sparsify_four_block_top_k(
        matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True
    )
    upper = np.triu(or_mask > 0.0)
    lopsided[upper] = 1.0
    assert int((normalize_ot_tsne_graph(lopsided) > 0.0).sum()) == int((or_mask > 0.0).sum()), (
        "an asymmetric mask is re-ORed by the normalizer; if that ever stops being true this "
        "test should be inverted rather than deleted"
    )


def test_mutual_selection_keeps_the_matrix_values_when_not_binarized():
    """Weight semantics for a surviving edge are the existing ones, not a new rule.

    The switch decides *which* edges survive. What a survivor is worth is still the entry of the
    four-block matrix, exactly as under "or".
    """
    x0, x1 = _zerodiag_inputs()
    n0, n1 = x0.shape[0], x1.shape[0]
    matrix, _, _ = _build_four_block_parts(compute_ot_plan(x0, x1, rel_reg=1 / 10))

    weighted, _ = sparsify_four_block_top_k(
        matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=False, graph_symmetrization="and"
    )
    binary, _ = sparsify_four_block_top_k(
        matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True, graph_symmetrization="and"
    )

    kept = binary > 0.0
    # `sparsify_four_block_top_k` symmetrises before selecting, so compare against that same matrix.
    np.testing.assert_array_equal(weighted[kept], methods._symmetrised(matrix)[kept])
    assert not weighted[~kept].any()


def test_mutual_kNN_diagonal_is_the_intersection_of_the_euclidean_knn():
    """The second switch, on ot-knndiag's Euclidean diagonal.

    Independent of the OT one: this block never passes through the four-block sparsifier, it is
    written in afterwards.
    """
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 4))
    indices = methods.within_batch_knn_indices(x, 5)
    directed = _directed_adjacency(indices, 30)

    shared = methods._adjacency_from_indices(indices, 30)
    mutual = methods._adjacency_from_indices(indices, 30, symmetrization="and")

    np.testing.assert_array_equal(shared > 0.0, directed | directed.T)
    expected = directed & directed.T
    np.fill_diagonal(expected, False)
    np.testing.assert_array_equal(mutual > 0.0, expected)
    assert np.array_equal(mutual, mutual.T)
    assert (mutual > 0.0).sum() < (shared > 0.0).sum()


@pytest.mark.parametrize("bad", ["OR", "and ", "mutual", "", None])
def test_an_unknown_symmetrization_is_refused(bad):
    """A typo must fail loudly rather than fall through to the historical default."""
    x0, x1 = _zerodiag_inputs()
    n0, n1 = x0.shape[0], x1.shape[0]
    matrix, _, _ = _build_four_block_parts(compute_ot_plan(x0, x1, rel_reg=1.0))
    with pytest.raises(ValueError, match="graph_symmetrization must be one of"):
        sparsify_four_block_top_k(
            matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K, graph_symmetrization=bad
        )


@pytest.mark.parametrize(
    "method",
    ["ot", "ot-knndiag", "ot-gramdiag", "ot-selfotmedian", "ot-zerodiag"],
)
def test_omitting_the_switch_reproduces_the_or_run_exactly(method):
    """Every entry point: no argument and ``"or"`` must be the same bytes.

    This is the guarantee the whole change rests on -- that nothing on disk moved -- expressed
    where a future edit would trip it, rather than only in a one-off digest comparison.
    """
    entry = {
        "ot": compute_ot_tsne_embedding_from_arrays,
        "ot-knndiag": compute_ot_tsne_knndiag_embedding_from_arrays,
        "ot-gramdiag": compute_ot_tsne_gramdiag_embedding_from_arrays,
        "ot-selfotmedian": methods.compute_ot_tsne_selfotmedian_embedding_from_arrays,
        "ot-zerodiag": compute_ot_tsne_zerodiag_embedding_from_arrays,
    }[method]
    x0, x1 = _zerodiag_inputs()
    shared = dict(rel_reg=1 / 10, effective_k=ZERODIAG_K, random_state=5, n_epochs=25)

    implicit = entry(x0, x1, **shared)
    explicit = entry(x0, x1, graph_symmetrization="or", **shared)
    assert implicit.tobytes() == explicit.tobytes()

    mutual = entry(x0, x1, graph_symmetrization="and", **shared)
    assert mutual.shape == implicit.shape
    # The ablation has to actually do something, or the tests above are checking nothing.
    assert mutual.tobytes() != implicit.tobytes()


def test_knndiag_takes_its_two_switches_independently():
    """Four combinations, all distinct: the OT selection and the kNN diagonal vary separately."""
    x0, x1 = _zerodiag_inputs()
    shared = dict(rel_reg=1 / 10, effective_k=ZERODIAG_K, random_state=5, n_epochs=25)

    embeddings = {}
    for graph_mode in ("or", "and"):
        for knn_mode in ("or", "and"):
            embeddings[(graph_mode, knn_mode)] = compute_ot_tsne_knndiag_embedding_from_arrays(
                x0,
                x1,
                graph_symmetrization=graph_mode,
                knndiag_symmetrization=knn_mode,
                **shared,
            ).tobytes()

    assert (
        embeddings[("or", "or")]
        == compute_ot_tsne_knndiag_embedding_from_arrays(x0, x1, **shared).tobytes()
    ), "the default pair must be the historical run"
    assert len(set(embeddings.values())) == 4, "the two switches are not independent"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"backend": "umap"}, "only supported by the t-SNE backend"),
        ({"percentage": 0.1, "effective_k": None}, "percentage sparsification"),
    ],
)
def test_a_mutual_graph_is_refused_where_it_cannot_be_expressed(kwargs, match):
    """Refusals, not silent no-ops.

    The UMAP arm embeds a directed neighbor list whose fuzzy union rebuilds the union, and the
    percentage sparsifier has no mutual mode. Either would return an "or" graph labelled "and",
    which is the one result an ablation of this distinction cannot tolerate.
    """
    x0, x1 = _zerodiag_inputs()
    plan = compute_ot_plan(x0, x1, rel_reg=1.0)
    call = dict(effective_k=ZERODIAG_K, random_state=0, n_epochs=25, x_joint=np.vstack([x0, x1]))
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        methods.compute_ot_tsne_embedding_from_plan(plan, graph_symmetrization="and", **call)


def test_a_mutual_graph_is_refused_with_mass_aware_budgets():
    """The mass-aware sparsifier has no mutual mode, so the unbalanced arms must refuse rather
    than quietly drop the request."""
    x0, x1 = _zerodiag_inputs()
    plan = compute_ot_plan(x0, x1, rel_reg=1.0)
    budgets = (np.full(x0.shape[0], 7), np.full(x1.shape[0], 7))
    with pytest.raises(ValueError, match="mass-aware budgets"):
        methods.compute_ot_tsne_embedding_from_plan(
            plan,
            effective_k=ZERODIAG_K,
            random_state=0,
            n_epochs=25,
            mass_budgets=budgets,
            graph_symmetrization="and",
        )


def test_the_mutual_graph_reports_its_coverage_shortfall():
    """A mutual graph can leave points with no edges at all, and the artifacts must let a caller
    see that.

    This is the failure mode the switch is most likely to be misread through: purity measured on a
    graph where most rows are empty looks excellent and means very little. The budget is unchanged
    -- ``effective_k`` still resolves to the same split -- so degree is the only thing that moves,
    and that is what this pins.
    """
    x0, x1 = _zerodiag_inputs()
    n0, n1 = x0.shape[0], x1.shape[0]
    matrix, _, _ = _build_four_block_parts(compute_ot_plan(x0, x1, rel_reg=1.0))

    or_mask, or_split = sparsify_four_block_top_k(
        matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True
    )
    and_mask, and_split = sparsify_four_block_top_k(
        matrix, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True, graph_symmetrization="and"
    )

    assert or_split == and_split, "the neighbour budget must not move with the symmetrization"
    or_degree = (or_mask > 0).sum(axis=1)
    and_degree = (and_mask > 0).sum(axis=1)
    # Under OR a row's degree is at least the budget; under AND that floor is gone. This is the
    # documented consequence of the rule, not a bug.
    assert or_degree.min() >= or_split.effective_k
    assert and_degree.sum() < or_degree.sum()
    assert (and_degree <= or_degree).all()


# ot-zerodiag-split: the same removal under the proportional split, so the surviving cross-batch
# edges are the base method's exactly and only the deletion differs.


def test_zerodiag_split_keeps_the_base_methods_cross_edges_exactly():
    """Require split-budget diagonal removal to preserve the base cross-batch mask exactly.

    Each block is ranked independently, so removing within-batch blocks cannot change
    cross-block selections. Full-budget zero-diagonal selection has different quotas.
    """
    x0, x1 = _zerodiag_inputs()
    plan = compute_ot_plan(x0, x1, rel_reg=1 / 10)
    n0, n1 = plan.shape

    base = build_ot_tsne_four_block_matrix(plan)
    zeroed = build_ot_tsne_zerodiag_four_block_matrix(plan)
    reference, base_split = sparsify_four_block_top_k(
        base, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True
    )
    produced, split = sparsify_four_block_cross_batch_split(
        zeroed, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True
    )

    assert split == base_split, "the split arm must use the base method's allocation unchanged"
    np.testing.assert_array_equal(produced[:n0, n0:], reference[:n0, n0:])
    np.testing.assert_array_equal(produced[n0:, :n0], reference[n0:, :n0])
    # And the diagonal is gone, where the base method's is populated.
    assert int((produced[:n0, :n0] > 0.0).sum()) == 0
    assert int((produced[n0:, n0:] > 0.0).sum()) == 0
    assert (reference[:n0, :n0] > 0.0).any()
    assert (reference[n0:, n0:] > 0.0).any()


def test_the_two_zerodiag_arms_differ_only_in_the_budget():
    """Both delete the diagonal; they part company on what replaces the freed quota.

    Pinning the relationship rather than either arm alone, because the pair is what separates
    "no within-batch information" from "different graph density". The full-budget arm must be a
    strict superset of the split arm's cross-batch edges and must keep the base method's degree;
    the split arm must keep the base method's edges and lose the degree.
    """
    x0, x1 = _zerodiag_inputs()
    plan = compute_ot_plan(x0, x1, rel_reg=1 / 10)
    n0, n1 = plan.shape
    zeroed = build_ot_tsne_zerodiag_four_block_matrix(plan)

    full = (
        sparsify_four_block_cross_batch_only(
            zeroed, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True
        )
        > 0.0
    )
    split, _ = sparsify_four_block_cross_batch_split(
        zeroed, n0=n0, n1=n1, effective_k=ZERODIAG_K, binarize=True
    )
    split = split > 0.0
    base = (
        sparsify_four_block_top_k(
            build_ot_tsne_four_block_matrix(plan),
            n0=n0,
            n1=n1,
            effective_k=ZERODIAG_K,
            binarize=True,
        )[0]
        > 0.0
    )

    # Superset: every edge the split arm keeps, the full-budget arm keeps too.
    assert np.array_equal(full & split, split)
    assert full.sum() > split.sum(), "the full budget must add edges, or the arms are the same"
    # Degree: the full arm matches the base method, the split arm falls short of it by the
    # within-batch share it no longer has.
    assert full.sum(axis=1).mean() == pytest.approx(base.sum(axis=1).mean(), rel=0.1)
    assert split.sum(axis=1).mean() < base.sum(axis=1).mean() * 0.75


def test_zerodiag_split_refuses_umap_rather_than_lying_about_the_graph():
    """Two row widths cannot fit a dense (N, k) neighbor list, so the combination raises.

    Under the split a batch-0 point draws ``batch1_k`` neighbors and a batch-1 point ``batch0_k``.
    Embedding a fixed-width list anyway would hand UMAP a graph that is not the one the sparsifier
    built. Same refusal, and the same reason, as ``ot-uot-mass``.
    """
    x0, x1 = _zerodiag_inputs()
    with pytest.raises(ValueError, match="only supported by the t-SNE backend"):
        compute_ot_tsne_zerodiag_split_embedding_from_arrays(
            x0, x1, rel_reg=1 / 10, effective_k=ZERODIAG_K, backend="umap"
        )
    # The full-budget arm has no such restriction: its rows are all effective_k wide.
    embedding = compute_ot_tsne_zerodiag_embedding_from_arrays(
        x0, x1, rel_reg=1 / 10, effective_k=ZERODIAG_K, random_state=0, n_epochs=25, backend="umap"
    )
    assert embedding.shape == (ZERODIAG_N0 + ZERODIAG_N1, 2)
    assert np.isfinite(embedding).all()


def test_zerodiag_split_graph_is_all_cross_batch_and_deterministic():
    """The end-to-end shape, artifact contract and seed determinism of the second arm."""
    x0, x1 = _zerodiag_inputs()
    kwargs = dict(rel_reg=1 / 10, effective_k=ZERODIAG_K, random_state=5, n_epochs=25)

    first, artifacts = compute_ot_tsne_zerodiag_split_embedding_from_arrays(
        x0, x1, return_artifacts=True, **kwargs
    )
    second = compute_ot_tsne_zerodiag_split_embedding_from_arrays(x0, x1, **kwargs)

    assert first.shape == (ZERODIAG_N0 + ZERODIAG_N1, 2)
    assert np.isfinite(first).all()
    assert first.tobytes() == second.tobytes()
    assert artifacts["diag0_edges"] == 0
    assert artifacts["diag1_edges"] == 0
    assert artifacts["within_batch_k"] == 0
    # Unlike the full-budget arm, the proportional split is meaningful here and is reported under
    # the same keys the base method uses.
    split = resolve_effective_k_split(ZERODIAG_K, ZERODIAG_N0, ZERODIAG_N1)
    assert artifacts["batch0_k"] == split.batch0_k
    assert artifacts["batch1_k"] == split.batch1_k
    assert artifacts["epsilon"] > 0.0
    # t-SNE only, so the dense square is never retained.
    assert "four_block_matrix" not in artifacts
