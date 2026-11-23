"""The plan SVD behind OT-tSNE's spectral within-batch blocks: solver and directions."""

import numpy as np
import pytest

from ne_aligned_utils.benchmark.pipeline import (
    liger_cores_task_args,
    ot_svd_task_args,
    scientific_method_settings,
)
from ne_aligned_utils.config import DEFAULT_OT_SVD_DIRECTIONS, DEFAULT_OT_SVD_SOLVER
from ne_aligned_utils.methods.ot_affinity import PlanSpectrum, resolve_plan_svd
from ne_aligned_utils.methods.ot_cache import cache_key
from ne_aligned_utils.methods.ot_methods import compute_ot_tsne_embedding_from_plan
from ne_aligned_utils.methods.ot_transport import compute_ot_plan


def _plan(n0=90, n1=70, seed=0):
    rng = np.random.default_rng(seed)
    return compute_ot_plan(rng.normal(size=(n0, 6)), rng.normal(size=(n1, 6)) + 0.3)


def test_defaults_are_randomized_with_500_directions():
    assert DEFAULT_OT_SVD_SOLVER == "randomized"
    assert DEFAULT_OT_SVD_DIRECTIONS == 500


@pytest.mark.parametrize(
    "solver, directions, expected",
    [
        ("full", 10, ("full", 70)),
        ("randomized", 10, ("randomized", 10)),
        ("arpack", 10, ("arpack", 10)),
        # Half the directions or more (35 of 70) is the full SVD: faster there, and exact.
        ("randomized", 34, ("randomized", 34)),
        ("randomized", 35, ("full", 70)),
        ("arpack", 35, ("full", 70)),
        ("randomized", 70, ("full", 70)),
        ("arpack", 1000, ("full", 70)),
        ("randomized", None, ("full", 70)),
    ],
)
def test_resolve_plan_svd(solver, directions, expected):
    assert resolve_plan_svd(solver, directions, 90, 70) == expected


@pytest.mark.parametrize(
    "smaller_batch, expected",
    # The default 500 directions on the benchmark's smaller-batch sizes (Table R).
    [
        (999, "full"),
        (1000, "full"),
        (1001, "randomized"),
        (1240, "randomized"),
        (2715, "randomized"),
        (11352, "randomized"),
    ],
)
def test_default_directions_switch_to_randomized_above_1000_cells(smaller_batch, expected):
    used, _ = resolve_plan_svd(
        DEFAULT_OT_SVD_SOLVER, DEFAULT_OT_SVD_DIRECTIONS, smaller_batch + 500, smaller_batch
    )
    assert used == expected


@pytest.mark.parametrize("directions", [0, -1, 2.5, True])
def test_resolve_plan_svd_rejects_bad_directions(directions):
    with pytest.raises(ValueError, match="positive integer"):
        resolve_plan_svd("randomized", directions, 90, 70)


def test_resolve_plan_svd_rejects_unknown_solver():
    with pytest.raises(ValueError, match="svd solver"):
        resolve_plan_svd("lobpcg", 10, 90, 70)


@pytest.mark.parametrize("solver", ["randomized", "arpack"])
def test_truncated_spectrum_matches_leading_full_directions(solver):
    """Truncated solvers return the leading singular values and directions of the full SVD."""
    plan = _plan()
    full = PlanSpectrum.from_plan(plan, solver="full")
    q = 8
    spectrum = PlanSpectrum.from_plan(plan, solver=solver, n_directions=q)

    assert spectrum.singular_values.shape == (q,)
    assert spectrum.eot.shape == (plan.shape[0] + plan.shape[1], q)
    np.testing.assert_allclose(spectrum.singular_values, full.singular_values[:q], rtol=1e-6)
    # Directions agree up to sign: compare the projectors they span, one direction at a time.
    for column in range(q):
        a, b = spectrum.eot[:, column], full.eot[:, column]
        np.testing.assert_allclose(np.outer(a, a), np.outer(b, b), atol=1e-5)


def test_full_directions_reproduce_the_full_spectrum():
    plan = _plan()
    full = PlanSpectrum.from_plan(plan, solver="full")
    asked_for_all = PlanSpectrum.from_plan(plan, solver="randomized", n_directions=1000)
    np.testing.assert_array_equal(asked_for_all.singular_values, full.singular_values)
    np.testing.assert_array_equal(asked_for_all.eot, full.eot)


def test_randomized_spectrum_does_not_depend_on_the_call():
    plan = _plan()
    first = PlanSpectrum.from_plan(plan, solver="randomized", n_directions=8)
    second = PlanSpectrum.from_plan(plan, solver="randomized", n_directions=8)
    np.testing.assert_array_equal(first.eot, second.eot)


def test_embedding_records_the_solver_and_directions():
    plan = _plan()
    _, truncated = compute_ot_tsne_embedding_from_plan(
        plan,
        n_epochs=5,
        early_exaggeration_iter=5,
        svd_solver="randomized",
        svd_directions=8,
        return_artifacts=True,
    )
    assert truncated["svd_solver"] == "randomized"
    assert truncated["svd_directions"] == 8
    _, full = compute_ot_tsne_embedding_from_plan(
        plan, n_epochs=5, early_exaggeration_iter=5, svd_solver="full", return_artifacts=True
    )
    assert full["svd_solver"] == "full"
    assert full["svd_directions"] == 70
    # Only the within-batch edges can change; the cross-batch block is the plan itself.
    n0 = plan.shape[0]
    rows, cols = truncated["graph"].nonzero()
    cross_truncated = {(r, c) for r, c in zip(rows, cols) if (r < n0) != (c < n0)}
    rows, cols = full["graph"].nonzero()
    cross_full = {(r, c) for r, c in zip(rows, cols) if (r < n0) != (c < n0)}
    assert cross_truncated == cross_full


def test_cache_key_keeps_full_svd_keys_and_separates_truncated_ones():
    rng = np.random.default_rng(1)
    x0, x1 = rng.normal(size=(30, 4)), rng.normal(size=(20, 4))
    common = dict(rel_reg=1.0, tol=1e-12, max_iter=100, device="cpu", threads=1)
    legacy = cache_key(x0, x1, **common)
    assert cache_key(x0, x1, svd_solver="full", **common) == legacy
    # Asking for every direction is the full SVD, so it shares the full SVD's cache file.
    assert cache_key(x0, x1, svd_solver="randomized", svd_directions=50, **common) == legacy
    randomized = cache_key(x0, x1, svd_solver="randomized", svd_directions=5, **common)
    assert randomized != legacy
    assert randomized != cache_key(x0, x1, svd_solver="arpack", svd_directions=5, **common)
    assert randomized != cache_key(x0, x1, svd_solver="randomized", svd_directions=6, **common)


def _manifest_settings(**extra):
    settings = {
        "embedding_dim": 2,
        "visualization_epochs": 500,
        "perplexity": 30.0,
        "ot_init": "eot",
        "ot_tolerance": 1e-12,
        "ot_max_iterations": 10000,
    }
    settings.update(extra)
    return {"settings": settings}


def test_manifests_without_the_setting_run_the_full_svd():
    """Older manifests ran every direction; they must still, and keep their run ids."""
    manifest = _manifest_settings()
    assert ot_svd_task_args(manifest["settings"]) == ["--svd-solver", "full"]
    assert "ot_svd_solver" not in scientific_method_settings(manifest, "ot")


def test_manifests_with_the_setting_pass_and_record_it():
    manifest = _manifest_settings(ot_svd_solver="randomized", ot_svd_directions=1000)
    assert ot_svd_task_args(manifest["settings"]) == [
        "--svd-solver",
        "randomized",
        "--svd-directions",
        "1000",
    ]
    settings = scientific_method_settings(manifest, "ot")
    assert settings["ot_svd_solver"] == "randomized"
    assert settings["ot_svd_directions"] == 1000
    # Variants without spectral blocks do not depend on it, so their run ids do not move.
    assert "ot_svd_solver" not in scientific_method_settings(manifest, "ot-knndiag")


def test_manifests_without_liger_cores_run_rligers_default():
    """Older manifests ran LIGER at rliger's default of 2 cores and keep their run ids."""
    manifest = _manifest_settings()
    assert liger_cores_task_args("liger-50", manifest["settings"]) == []
    assert "liger_n_cores" not in scientific_method_settings(manifest, "liger-50")


def test_manifests_with_liger_cores_pass_and_record_it():
    manifest = _manifest_settings(liger_n_cores=4)
    for method in ("liger-2d", "liger-50"):
        assert liger_cores_task_args(method, manifest["settings"]) == ["--liger-n-cores", "4"]
        assert scientific_method_settings(manifest, method)["liger_n_cores"] == 4
    # Other methods, including the historical liger1, do not take it.
    for method in ("liger1-50", "seurat-50", "ot"):
        assert liger_cores_task_args(method, manifest["settings"]) == []
        assert "liger_n_cores" not in scientific_method_settings(manifest, method)
