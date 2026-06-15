import numpy as np
import pytest

from ne_aligned_utils.methods import (
    compute_eot_embedding_from_arrays,
    compute_eot_embedding_from_plan,
    compute_eot_then_tsne_embedding_from_arrays,
    compute_ot_plan,
    compute_ot_tsne_embedding_from_arrays,
    compute_pca_embedding_from_arrays,
    compute_seurat_embedding_from_arrays,
    compute_tsne_embedding_from_arrays,
    resolve_effective_k_split,
    sparsify_four_block_top_k,
)
from sklearn.decomposition import PCA


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

    expected = PCA(n_components=2, random_state=0).fit_transform(
        np.concatenate([x0, x1], axis=0)
    )
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


def test_compute_seurat_embedding_from_arrays_matches_cross_product_svd():
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

    embedding, artifacts = compute_seurat_embedding_from_arrays(
        x0,
        x1,
        embd_dim=2,
        return_artifacts=True,
    )

    u, singular_values, vt = np.linalg.svd(x0 @ x1.T, full_matrices=False)
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


def test_compute_seurat_embedding_from_arrays_clips_dimension_to_available_rank():
    x0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    x1 = np.array([[1.0, 1.0], [1.0, -1.0], [0.5, 0.25]])

    embedding, artifacts = compute_seurat_embedding_from_arrays(
        x0,
        x1,
        embd_dim=10,
        return_artifacts=True,
    )

    assert embedding.shape == (x0.shape[0] + x1.shape[0], min(x0.shape[0], x1.shape[0]))
    assert artifacts["embd_dim"] == min(x0.shape[0], x1.shape[0])
    assert artifacts["embd_dim_requested"] == 10


def test_compute_seurat_embedding_from_arrays_validates_inputs():
    with pytest.raises(ValueError, match="2D arrays"):
        compute_seurat_embedding_from_arrays(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError, match="same number of features"):
        compute_seurat_embedding_from_arrays(np.zeros((2, 2)), np.zeros((3, 3)))

    with pytest.raises(ValueError, match="positive"):
        compute_seurat_embedding_from_arrays(np.zeros((2, 2)), np.zeros((3, 2)), embd_dim=0)


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
        eps_times=1.0,
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
        eps_times=1.0,
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
        eps_times=1.0,
        effective_k=3,
        random_state=0,
        n_epochs=12,
        return_artifacts=True,
    )

    assert embedding.shape == (9, 2)
    assert np.isfinite(embedding).all()
    assert artifacts["ot_plan"].shape == (4, 5)
    assert artifacts["four_block_matrix"].shape == (9, 9)
    assert artifacts["sparse_four_block_matrix"].shape == (9, 9)
    assert artifacts["graph"].shape == (9, 9)
    assert artifacts["effective_k"] == 3
    assert artifacts["batch0_k"] == 1
    assert artifacts["batch1_k"] == 2
    assert artifacts["epsilon"] > 0.0
    assert artifacts["median_squared_distance"] > 0.0
    np.testing.assert_allclose(artifacts["graph"].sum(), 1.0)
