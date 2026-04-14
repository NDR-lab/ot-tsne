import numpy as np
import pytest

from ne_aligned_utils.metrics import (
    compute_concordance,
    compute_cross_concordance,
    compute_foscttm,
    compute_knn_other_batch_fraction,
    compute_knn_same_cell_type_fraction,
)


def make_ring(n, radius=1.0, radial_noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = radius + rng.normal(0.0, radial_noise, size=n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)


def rotate(points, deg):
    rad = np.deg2rad(deg)
    rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return points @ rot.T


def shift_x(points, dx):
    out = points.copy()
    out[:, 0] += dx
    return out


def scale_radius(points, factor):
    return points * factor


def squash_y(points, factor):
    out = points.copy()
    out[:, 1] *= factor
    return out


def compute_foscttm_proxy_latent_nn(x0_ref, x1_ref, y0, y1):
    """
    FOSCTTM for unequal sizes:
    true match is nearest neighbor in latent/clean space.
    Computed in both directions and averaged.
    """
    return compute_foscttm(x0_ref, x1_ref, y0, y1)


def compute_metric_row(x0_ref, x1_ref, y0, y1, k=50):
    k_cross = min(k, x0_ref.shape[0], x1_ref.shape[0], y0.shape[0], y1.shape[0])
    x_ref_all = np.concatenate([x0_ref, x1_ref], axis=0)
    y_all = np.concatenate([y0, y1], axis=0)
    k_conc = min(k, x_ref_all.shape[0])

    row = {
        "concordance": compute_concordance(x_ref_all, y_all, k=k_conc),
        "cross_concordance": compute_cross_concordance(x0_ref, x1_ref, y0, y1, k=k_cross),
    }
    row["foscttm"] = compute_foscttm(x0_ref, x1_ref, y0, y1)
    row["foscttm_mode"] = "proxy_latent_nn"

    return row


def select_steps_for_row(steps, n_cols=5):
    steps = list(steps)
    if len(steps) <= n_cols:
        return steps
    idx = np.linspace(0, len(steps) - 1, n_cols, dtype=int)
    return [steps[i] for i in idx]


def format_ring_step_label(family, value):
    if family == "rotation_deg":
        return f"deg={value}"
    if family == "shift_right":
        return f"dx={value}"
    if family == "radius_growth":
        return f"scale={value}"
    if family == "oval_squash":
        return f"y-scale={value}"
    return str(value)


def make_corresponded_gmm(n0=600, n1=600, q=6, mean_scale=5.0, base_std=0.6, seed=0):
    rng = np.random.default_rng(seed)

    # 2D cluster means on a circle for clear visualization.
    angles = np.linspace(0, 2 * np.pi, q, endpoint=False)
    means = np.stack([mean_scale * np.cos(angles), mean_scale * np.sin(angles)], axis=1)

    if n0 == n1:
        # Equal-size mode keeps exact pointwise correspondence.
        labels = rng.integers(0, q, size=n0)
        base_points = means[labels] + rng.normal(scale=base_std, size=(n0, 2))
        x0_ref = base_points.copy()
        x1_ref = base_points.copy()
        labels0 = labels.copy()
        labels1 = labels.copy()
    else:
        # Unequal-size mode samples each modality independently from same GMM.
        labels0 = rng.integers(0, q, size=n0)
        labels1 = rng.integers(0, q, size=n1)
        x0_ref = means[labels0] + rng.normal(scale=base_std, size=(n0, 2))
        x1_ref = means[labels1] + rng.normal(scale=base_std, size=(n1, 2))

    return x0_ref, x1_ref, labels0, labels1, means


def gmm_shift_x(points, dx):
    out = points.copy()
    out[:, 0] += dx
    return out


def gmm_scatter(means, labels, scatter_std, seed):
    """
    Sample around fixed cluster means with a chosen std.
    """
    rng = np.random.default_rng(seed)
    centers = means[labels]
    noise = rng.normal(scale=scatter_std, size=centers.shape)
    return centers + noise


def format_gmm_step_label(family, value):
    if family == "shift_right":
        return f"dx={value}"
    if family == "scatter":
        return f"std={value}"
    return str(value)


def test_compute_knn_other_batch_fraction_excludes_self():
    embd = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 10.0],
            [10.0, 10.0],
        ]
    )
    batch_ids = np.array([0, 1, 0, 1])

    score = compute_knn_other_batch_fraction(embd, batch_ids, k=1)

    assert score == pytest.approx(1.0)


def test_compute_knn_same_cell_type_fraction_detects_pure_clusters():
    embd = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])

    score = compute_knn_same_cell_type_fraction(embd, labels, k=2)

    assert score == pytest.approx(1.0)


def test_compute_knn_other_batch_fraction_detects_poor_integration():
    embd = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ]
    )
    batch_ids = np.array([0, 0, 0, 1, 1, 1])

    score = compute_knn_other_batch_fraction(embd, batch_ids, k=2)

    assert score == pytest.approx(0.0)


def test_knn_fraction_metrics_clip_k_to_available_neighbors():
    embd = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    labels = np.array([3, 3])
    batch_ids = np.array([0, 1])

    same_type_score = compute_knn_same_cell_type_fraction(embd, labels, k=10)
    other_batch_score = compute_knn_other_batch_fraction(embd, batch_ids, k=10)

    assert same_type_score == pytest.approx(1.0)
    assert other_batch_score == pytest.approx(1.0)


def test_knn_fraction_metrics_return_zero_for_tiny_inputs():
    embd = np.array([[0.0, 0.0]])
    labels = np.array([0])
    batch_ids = np.array([0])

    assert compute_knn_same_cell_type_fraction(embd, labels, k=1) == pytest.approx(0.0)
    assert compute_knn_other_batch_fraction(embd, batch_ids, k=1) == pytest.approx(0.0)


def test_knn_fraction_metrics_validate_inputs():
    embd = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="same number of samples"):
        compute_knn_same_cell_type_fraction(embd, np.array([0]), k=1)

    with pytest.raises(ValueError, match="positive integer"):
        compute_knn_other_batch_fraction(embd, np.array([0, 1]), k=0)
