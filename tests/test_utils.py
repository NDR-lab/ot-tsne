from pathlib import Path

import numpy as np
import pytest

from ne_aligned_utils.methods import compute_seurat_embedding_from_arrays
from ne_aligned_utils.utils import get_path, kNN_graph, load_dict, save_dict


def _align_svd_column_signs(actual, expected):
    aligned = np.array(actual, copy=True)
    for col in range(expected.shape[1]):
        pivot = int(np.argmax(np.abs(expected[:, col])))
        if aligned[pivot, col] * expected[pivot, col] < 0:
            aligned[:, col] *= -1.0
    return aligned


def test_get_seurat_matches_cross_product_svd():
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

    u, expected_singular_values, vt = np.linalg.svd(x0 @ x1.T, full_matrices=False)
    expected_embedding = np.concatenate([u[:, :2], vt.T[:, :2]], axis=0)
    aligned_embedding = _align_svd_column_signs(embedding, expected_embedding)

    np.testing.assert_allclose(aligned_embedding, expected_embedding)
    np.testing.assert_allclose(artifacts["singular_values"], expected_singular_values)


def test_get_seurat_clips_embedding_dim_to_available_rank():
    x0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    x1 = np.array([[1.0, 1.0], [1.0, -1.0], [0.5, 0.25]])

    embedding = compute_seurat_embedding_from_arrays(x0, x1, embd_dim=10)

    assert embedding.shape == (x0.shape[0] + x1.shape[0], min(x0.shape[0], x1.shape[0]))


def test_get_seurat_validates_inputs():
    with pytest.raises(ValueError, match="2D arrays"):
        compute_seurat_embedding_from_arrays(np.array([1.0, 2.0]), np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError, match="same number of features"):
        compute_seurat_embedding_from_arrays(np.zeros((2, 2)), np.zeros((3, 3)))

    with pytest.raises(ValueError, match="positive"):
        compute_seurat_embedding_from_arrays(np.zeros((2, 2)), np.zeros((3, 2)), embd_dim=0)


def test_get_path_resolves_repo_relative_paths():
    expected = (Path(__file__).resolve().parents[1] / "data").resolve()
    assert Path(get_path("data")) == expected


def test_save_load_dict_round_trip(tmp_path):
    payload = {"array": np.array([1, 2, 3]), "label": "yao"}
    file_path = tmp_path / "nested" / "payload.pkl"

    save_dict(payload, file_path)
    loaded = load_dict(file_path)

    np.testing.assert_array_equal(loaded["array"], payload["array"])
    assert loaded["label"] == payload["label"]


def test_knn_graph_returns_neighbor_indices_without_self():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])

    indices = kNN_graph(x, 1).numpy()

    assert indices.shape == (3, 1)
    np.testing.assert_array_equal(indices[:, 0], np.array([1, 0, 1]))
