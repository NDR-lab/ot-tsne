import numpy as np

from ne_aligned_utils.methods import compute_eot_embedding_from_arrays, compute_ot_plan


def _align_svd_column_signs(actual, expected):
    aligned = np.array(actual, copy=True)
    for col in range(expected.shape[1]):
        pivot = int(np.argmax(np.abs(expected[:, col])))
        if aligned[pivot, col] * expected[pivot, col] < 0:
            aligned[:, col] *= -1.0
    return aligned


def test_compute_eot_embedding_from_arrays_matches_ot_plan_svd():
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

    plan = compute_ot_plan(x0, x1, eps_times=1.0)
    u, singular_values, vt = np.linalg.svd(plan, full_matrices=False)
    expected_left = u[:, 1:3] * np.sqrt(x0.shape[0])
    expected_right = vt.T[:, 1:3] * np.sqrt(x1.shape[0])
    expected_joint = np.concatenate([expected_left, expected_right], axis=0)

    aligned_left = _align_svd_column_signs(artifacts["left_embedding"], expected_left)
    aligned_right = _align_svd_column_signs(artifacts["right_embedding"], expected_right)
    aligned_joint = _align_svd_column_signs(embedding, expected_joint)

    np.testing.assert_allclose(aligned_left, expected_left)
    np.testing.assert_allclose(aligned_right, expected_right)
    np.testing.assert_allclose(aligned_joint, expected_joint)
    np.testing.assert_array_less(0.0, singular_values[:2] + 1e-12)
