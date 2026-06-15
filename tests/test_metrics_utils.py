import numpy as np
import pytest

from ne_aligned_utils.metrics import (
    compute_concordance,
    compute_cross_concordance,
    compute_foscttm,
    compute_ilisi,
    compute_kbet,
    compute_knn_other_batch_fraction,
    compute_knn_same_cell_type_fraction,
    compute_neighbor_purity,
    compute_silhouette_index,
    compute_silhouette_score,
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


def test_compute_knn_other_batch_fraction_hits_expected_midpoint_for_well_mixed_shared_type():
    embd = np.zeros((4, 2), dtype=float)
    labels = np.zeros(4, dtype=int)
    batch_ids = np.array([0, 0, 1, 1], dtype=int)

    score = compute_knn_other_batch_fraction(
        embd,
        batch_ids,
        labels,
        k=10,
        min_cells_per_batch=1,
    )

    assert score == pytest.approx(0.5)


def test_compute_neighbor_purity_matches_same_cell_type_fraction_alias():
    embd = np.array(
        [[float(i), 0.0] for i in range(10)]
        + [[100.0 + i, 0.0] for i in range(10)]
    )
    labels = np.array([0] * 10 + [1] * 10)

    purity = compute_neighbor_purity(embd, labels, k=50)
    same_type = compute_knn_same_cell_type_fraction(embd, labels, k=50)

    assert purity == pytest.approx(1.0)
    assert purity == pytest.approx(same_type)


def test_compute_neighbor_purity_uses_macro_fixed_knn_label_mean():
    n_dims = 7
    pure_label = np.zeros((20, n_dims), dtype=float)
    pure_label[:, 0] = 100.0 + np.arange(20) * 0.01
    mixed_label = np.eye(5, n_dims, dtype=float)
    skipped_decoys = np.zeros((4, n_dims), dtype=float)

    embd = np.vstack([pure_label, mixed_label, skipped_decoys])
    labels = np.array([0] * 20 + [1] * 5 + [2] * 4)

    score = compute_neighbor_purity(embd, labels, k=50)

    assert score == pytest.approx((1.0 + 0.0) / 2.0)


def test_compute_neighbor_purity_returns_zero_when_all_labels_are_skipped():
    embd = np.array([[float(i), 0.0] for i in range(8)])
    labels = np.array([0] * 4 + [1] * 4)

    score = compute_neighbor_purity(embd, labels, k=50)

    assert score == pytest.approx(0.0)


def test_compute_silhouette_index_detects_well_separated_clusters():
    embd = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [4.0, 4.0],
            [4.2, 4.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])

    score = compute_silhouette_index(embd, labels)

    assert score > 0.9


def test_compute_silhouette_index_returns_zero_for_degenerate_labels():
    embd = np.array([[0.0, 0.0], [1.0, 1.0]])

    assert compute_silhouette_index(embd, np.array([0, 0])) == pytest.approx(0.0)
    assert compute_silhouette_index(embd, np.array([0, 1])) == pytest.approx(0.0)


def test_compute_silhouette_score_matches_paper_rescaling():
    embd = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [4.0, 4.0],
            [4.2, 4.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])

    raw = compute_silhouette_index(embd, labels)
    scaled = compute_silhouette_score(embd, labels)

    assert scaled == pytest.approx((raw + 1.0) / 2.0)
    assert 0.0 <= scaled <= 1.0


def test_compute_ilisi_detects_balanced_local_batch_mixing():
    embd = np.repeat(np.arange(20, dtype=float)[:, None], 2, axis=0)
    embd = np.c_[embd, np.zeros(embd.shape[0])]
    batch_ids = np.tile([0, 1], 20)

    score = compute_ilisi(embd, batch_ids, k=30)

    assert score > 0.8


def test_compute_ilisi_detects_separated_batches():
    embd = np.array(
        [[float(i), 0.0] for i in range(10)]
        + [[100.0 + i, 0.0] for i in range(10)]
    )
    batch_ids = np.array([0] * 10 + [1] * 10)

    score = compute_ilisi(embd, batch_ids, k=6)

    assert score < 0.1


def test_compute_kbet_detects_balanced_local_batch_mixing():
    embd = np.array([[float(i), 0.0] for i in range(90)])
    batch_ids = np.array([0, 1] * 45)
    labels = np.zeros(90, dtype=int)

    score = compute_kbet(
        embd,
        batch_ids,
        labels,
    )

    assert score > 0.9


def test_compute_kbet_detects_local_batch_imbalance():
    embd = np.array(
        [[float(i), 0.0] for i in range(45)]
        + [[100.0 + i, 0.0] for i in range(45)]
    )
    batch_ids = np.array([0] * 45 + [1] * 45)
    labels = np.zeros(90, dtype=int)

    score = compute_kbet(
        embd,
        batch_ids,
        labels,
    )

    assert score < 0.1


def test_compute_knn_same_cell_type_fraction_detects_pure_clusters():
    embd = np.array(
        [[float(i), 0.0] for i in range(10)]
        + [[100.0 + i, 0.0] for i in range(10)]
    )
    labels = np.array([0] * 10 + [1] * 10)

    score = compute_knn_same_cell_type_fraction(embd, labels, k=50)

    assert score == pytest.approx(1.0)


def test_compute_knn_other_batch_fraction_detects_poor_integration():
    embd = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ]
    )
    labels = np.zeros(4, dtype=int)
    batch_ids = np.array([0, 0, 1, 1], dtype=int)

    score = compute_knn_other_batch_fraction(
        embd,
        batch_ids,
        labels,
        k=1,
        min_cells_per_batch=1,
    )

    assert score == pytest.approx(0.0)


def test_compute_knn_other_batch_fraction_detects_overmixing():
    embd = np.array([[0.0], [2.0], [4.0], [1.0], [3.0], [5.0]], dtype=float)
    labels = np.zeros(6, dtype=int)
    batch_ids = np.array([0, 0, 0, 1, 1, 1], dtype=int)

    score = compute_knn_other_batch_fraction(
        embd,
        batch_ids,
        labels,
        k=1,
        min_cells_per_batch=1,
    )

    assert score == pytest.approx(1.0)


def test_compute_knn_other_batch_fraction_skips_low_support_cell_types():
    embd = np.zeros((41, 2), dtype=float)
    labels = np.array([0] * 20 + [1] * 21, dtype=int)
    batch_ids = np.array([0] * 10 + [1] * 10 + [0] * 9 + [1] * 12, dtype=int)

    score = compute_knn_other_batch_fraction(embd, batch_ids, labels, k=50)

    assert score == pytest.approx(0.5)


def test_knn_fraction_metrics_return_zero_for_tiny_inputs():
    embd = np.array([[0.0, 0.0], [1.0, 1.0]])
    labels = np.array([0, 0])
    batch_ids = np.array([0, 1])

    assert compute_knn_same_cell_type_fraction(embd, labels, k=1) == pytest.approx(0.0)
    score = compute_knn_other_batch_fraction(embd, batch_ids, labels, k=1)
    assert np.isnan(score)


def test_knn_fraction_metrics_validate_inputs():
    embd = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError, match="same number of samples"):
        compute_knn_same_cell_type_fraction(embd, np.array([0]), k=1)

    with pytest.raises(ValueError, match="same number of samples"):
        compute_neighbor_purity(embd, np.array([0]), k=1)

    with pytest.raises(ValueError, match="same number of samples"):
        compute_knn_other_batch_fraction(embd, np.array([0, 1]), np.array([0]), k=1)

    with pytest.raises(ValueError, match="same number of samples"):
        compute_silhouette_index(embd, np.array([0]))

    with pytest.raises(ValueError, match="positive integer"):
        compute_knn_other_batch_fraction(embd, np.array([0, 1]), np.array([0, 1]), k=0)

    with pytest.raises(ValueError, match="exactly two batches"):
        compute_knn_other_batch_fraction(
            np.zeros((3, 2)),
            np.array([0, 1, 2]),
            np.zeros(3, dtype=int),
            k=1,
            min_cells_per_batch=1,
        )
