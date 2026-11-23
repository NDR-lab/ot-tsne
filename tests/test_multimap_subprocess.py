"""Check isolated MultiMAP input/output, parameters, timing, and failure handling."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ne_aligned_utils.methods.external.multimap import (
    MULTIMAP_PYTHON_ENV_VAR,
    MultiMapSubprocessError,
    compute_multimap_embedding_from_arrays,
    multimap_available,
    read_multimap_output,
    run_multimap_subprocess,
    write_multimap_input,
)

MOCK_MULTIMAP_PYTHON = Path(__file__).resolve().parent / "mock_multimap_python.py"


@pytest.fixture
def mock_multimap(monkeypatch):
    """Point the MultiMAP launcher at the mock interpreter for the duration of a test."""
    monkeypatch.setenv(MULTIMAP_PYTHON_ENV_VAR, f"{sys.executable} {MOCK_MULTIMAP_PYTHON}")


def test_write_input_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(6, 4))
    x1 = rng.normal(size=(9, 4))
    batch_ids = np.concatenate([np.zeros(6, int), np.ones(9, int)])
    path = tmp_path / "input.npz"

    write_multimap_input(
        path, x0, x1, batch_ids, {"method": "multimap", "n_components": 2, "seed": 3}
    )

    with np.load(path, allow_pickle=False) as handle:
        np.testing.assert_allclose(np.asarray(handle["x0"]), x0)
        np.testing.assert_allclose(np.asarray(handle["x1"]), x1)
        np.testing.assert_array_equal(np.asarray(handle["batch_ids"]), batch_ids)
        params = json.loads(str(handle["params_json"]))
    assert params == {"method": "multimap", "n_components": 2, "seed": 3}


def test_read_output_rejects_wrong_row_count(tmp_path):
    path = tmp_path / "output.npz"
    np.savez_compressed(path, embedding=np.zeros((11, 2)))
    with pytest.raises(MultiMapSubprocessError, match="rows"):
        read_multimap_output(path, n_total=15, n_components=2)


def test_read_output_rejects_wrong_width(tmp_path):
    path = tmp_path / "output.npz"
    np.savez_compressed(path, embedding=np.zeros((15, 3)))
    with pytest.raises(MultiMapSubprocessError, match="width"):
        read_multimap_output(path, n_total=15, n_components=2)


def test_read_output_rejects_non_finite(tmp_path):
    path = tmp_path / "output.npz"
    embedding = np.zeros((15, 2))
    embedding[3, 1] = np.nan
    np.savez_compressed(path, embedding=embedding)
    with pytest.raises(MultiMapSubprocessError, match="non-finite"):
        read_multimap_output(path, n_total=15, n_components=2)


def test_run_multimap_subprocess_with_mock(mock_multimap):
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=(10, 6))
    x1 = rng.normal(size=(14, 6))
    batch_ids = np.concatenate([np.zeros(10, int), np.ones(14, int)])

    embedding, artifacts = run_multimap_subprocess(
        x0, x1, batch_ids, {"method": "multimap", "n_components": 2, "seed": 0}
    )
    assert embedding.shape == (24, 2)
    assert np.isfinite(embedding).all()
    assert artifacts["multimap_runner"] == "multimap_runner.py"


def test_run_multimap_subprocess_does_not_inherit_notebook_backend(monkeypatch):
    monkeypatch.setenv(MULTIMAP_PYTHON_ENV_VAR, "mock-python")
    monkeypatch.setenv("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    observed = {}

    def fake_run(command, **kwargs):
        observed["backend"] = kwargs["env"]["MPLBACKEND"]
        input_path, output_path = Path(command[-2]), Path(command[-1])
        with np.load(input_path, allow_pickle=False) as handle:
            n_total = len(handle["x0"]) + len(handle["x1"])
        np.savez_compressed(output_path, embedding=np.zeros((n_total, 2)))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("ne_aligned_utils.methods.external.multimap.subprocess.run", fake_run)
    run_multimap_subprocess(
        np.zeros((3, 2)),
        np.zeros((4, 2)),
        np.array([0, 0, 0, 1, 1, 1, 1]),
        {"method": "multimap", "n_components": 2, "seed": 0},
    )
    assert observed["backend"] == "Agg"


def test_run_multimap_subprocess_raises_without_interpreter(monkeypatch):
    monkeypatch.delenv(MULTIMAP_PYTHON_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "ne_aligned_utils.methods.external.multimap.DEFAULT_MULTIMAP_PYTHON",
        Path("/nonexistent/python"),
    )
    with pytest.raises(MultiMapSubprocessError, match="numpy<2"):
        run_multimap_subprocess(
            np.zeros((3, 2)),
            np.zeros((3, 2)),
            np.array([0, 0, 0, 1, 1, 1]),
            {"method": "multimap", "n_components": 2, "seed": 0},
        )


def test_compute_multimap_via_mock(mock_multimap):
    rng = np.random.default_rng(2)
    x0 = rng.normal(size=(12, 8))
    x1 = rng.normal(size=(15, 8))
    embedding, artifacts = compute_multimap_embedding_from_arrays(
        x0, x1, embd_dim=2, random_state=0, return_artifacts=True
    )
    assert embedding.shape == (27, 2)
    assert np.isfinite(embedding).all()
    assert artifacts["multimap_dim"] == 2
    assert artifacts["multimap_n_neighbors"] == "native"
    assert artifacts["multimap_min_dist"] == "native"
    assert artifacts["multimap_spread"] == "native"
    assert artifacts["multimap_n_epochs"] == "native"
    assert artifacts["multimap_init"] == "native"
    assert artifacts["multimap_use_native_scaling"] is True
    assert artifacts["left_embedding"].shape == (12, 2)
    assert artifacts["right_embedding"].shape == (15, 2)


def test_compute_multimap_retains_an_explicit_controlled_profile(mock_multimap):
    x0 = np.zeros((5, 4))
    x1 = np.zeros((6, 4))
    _, artifacts = compute_multimap_embedding_from_arrays(
        x0,
        x1,
        n_neighbors=15,
        min_dist=0.1,
        spread=1.0,
        n_epochs=200,
        init="spectral",
        return_artifacts=True,
    )
    assert artifacts["multimap_n_neighbors"] == 15
    assert artifacts["multimap_min_dist"] == 0.1
    assert artifacts["multimap_spread"] == 1.0
    assert artifacts["multimap_n_epochs"] == 200
    assert artifacts["multimap_init"] == "spectral"


def test_compute_multimap_rejects_mismatched_features(mock_multimap):
    with pytest.raises(ValueError):
        compute_multimap_embedding_from_arrays(np.zeros((5, 4)), np.zeros((6, 3)))


def test_compute_multimap_rejects_non_finite_input(mock_multimap):
    x0 = np.zeros((5, 4))
    x1 = np.zeros((6, 4))
    x0[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compute_multimap_embedding_from_arrays(x0, x1)


def test_guarded_scaling_is_an_explicit_compatibility_mode(mock_multimap):
    x0 = np.zeros((5, 4))
    x1 = np.zeros((6, 4))
    _, artifacts = compute_multimap_embedding_from_arrays(
        x0, x1, use_native_scaling=False, return_artifacts=True
    )
    assert artifacts["multimap_use_native_scaling"] is False

    with pytest.raises(ValueError, match="requires scale=True"):
        compute_multimap_embedding_from_arrays(x0, x1, scale=False, use_native_scaling=True)


def test_compute_multimap_rejects_invalid_layout_parameters(mock_multimap):
    x0 = np.zeros((5, 4))
    x1 = np.zeros((6, 4))
    with pytest.raises(ValueError, match="must not exceed spread"):
        compute_multimap_embedding_from_arrays(x0, x1, min_dist=1.1, spread=1.0)
    with pytest.raises(ValueError, match="n_epochs must be positive"):
        compute_multimap_embedding_from_arrays(x0, x1, n_epochs=0)


def test_multimap_available_checks_override(monkeypatch):
    monkeypatch.setenv(MULTIMAP_PYTHON_ENV_VAR, "some-launcher")
    assert multimap_available() is False

    monkeypatch.setenv(MULTIMAP_PYTHON_ENV_VAR, sys.executable)
    assert multimap_available() is True
    monkeypatch.delenv(MULTIMAP_PYTHON_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "ne_aligned_utils.methods.external.multimap.DEFAULT_MULTIMAP_PYTHON",
        Path("/nonexistent/python"),
    )
    assert multimap_available() is False


# --- Opt-in real-MultiMAP integration ------------------------------------------
# Skipped unless NE_ALIGNED_TEST_REAL_MULTIMAP=1 and an isolated numpy<2 interpreter
# with MultiMAP + scanpy is reachable (see the external/README.md setup guide):
#   NE_ALIGNED_TEST_REAL_MULTIMAP=1 pytest -k real_multimap

real_multimap = pytest.mark.skipif(
    os.environ.get("NE_ALIGNED_TEST_REAL_MULTIMAP") != "1",
    reason="set NE_ALIGNED_TEST_REAL_MULTIMAP=1 to run the real-MultiMAP integration",
)


@real_multimap
def test_real_multimap_survives_constant_features():
    """Constant features made scanpy's per-batch scaling emit NaN (MNIST border pixels).

    They must be scaled to zero and kept, not dropped, and the embedding must stay finite.
    """
    rng = np.random.default_rng(1)
    n0, n1, n_features = 100, 120, 50
    x0 = rng.normal(size=(n0, n_features))
    x1 = rng.normal(size=(n1, n_features)) + 0.4
    x0[:, :8] = 0.0
    x1[:, :8] = 0.0
    x0[:, 8] = 3.0  # constant in one batch only
    x0[:, 9] = x1[:, 9] = 1e-38  # nonzero constants can also suffer cancellation
    x0[:, 10] *= 1e-38  # float32 squaring underflow must not create NaNs
    x1[:, 10] *= 1e-38
    embedding, artifacts = compute_multimap_embedding_from_arrays(
        x0, x1, embd_dim=2, random_state=0, return_artifacts=True
    )
    assert np.asarray(embedding).shape == (n0 + n1, 2)
    assert np.isfinite(embedding).all()
    assert artifacts["n_constant_features"] == 10
    assert artifacts["n_features_used"] == n_features  # kept, not dropped


@real_multimap
def test_real_multimap_embeds_two_batches():
    rng = np.random.default_rng(0)
    n0, n1, n_features = 120, 140, 60
    x0 = rng.normal(size=(n0, n_features))
    x1 = rng.normal(size=(n1, n_features)) + 0.5
    embedding = np.asarray(
        compute_multimap_embedding_from_arrays(x0, x1, embd_dim=2, random_state=0)
    )
    assert embedding.shape == (n0 + n1, 2)
    assert np.isfinite(embedding).all()


def test_native_scaling_preserves_tiny_mnist_values():
    """Float32 squaring underflow made native Scanpy scaling feed NaNs to PCA."""
    from ne_aligned_utils.methods.external.py_scripts.multimap_runner import _prepare_batches

    rng = np.random.default_rng(5)
    x0 = rng.normal(size=(100, 8))
    x1 = rng.normal(size=(120, 8))
    x0[:, 0] = x1[:, 0] = 0
    x0[:, 1] = x1[:, 1] = 1e-38
    x0[:, 2] = (1 + 0.1 * rng.normal(size=100)) * 1e-38
    x1[:, 2] = (1 + 0.1 * rng.normal(size=120)) * 1e-38
    joint, counts = _prepare_batches(x0, x1, scale=True, use_native_scaling=True)
    assert joint.dtype == np.float64
    np.testing.assert_array_equal(joint[:, 2:], np.vstack([x0, x1])[:, 2:])
    np.testing.assert_array_equal(joint[:, :2], 0)
    assert counts == (2, 2)

    import scanpy as sc

    for block in (joint[:100], joint[100:]):
        scaled = sc.pp.scale(block.copy())
        assert np.isfinite(scaled).all()
        np.testing.assert_allclose(scaled[:, 0], 0)
        np.testing.assert_allclose(scaled[:, 2].std(ddof=1), 1)


def test_guarded_scaling_keeps_compatibility_precision():
    from ne_aligned_utils.methods.external.py_scripts.multimap_runner import _prepare_batches

    rng = np.random.default_rng(10)
    x0, x1 = rng.normal(size=(30, 6)), rng.normal(size=(40, 6))
    x0[:, 0] = x1[:, 0] = 0
    joint, counts = _prepare_batches(x0, x1, scale=True, use_native_scaling=False)
    assert joint.dtype == np.float32
    assert counts == (1, 1)
    assert np.isfinite(joint).all()
