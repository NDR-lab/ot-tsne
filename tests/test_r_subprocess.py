"""Check R adapter serialization, method parameters, timing, and failure handling."""

import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from ne_aligned_utils.methods.external.r import (
    LIGER_RSCRIPT_ENV_VAR,
    RSCRIPT_ENV_VAR,
    STANDARD_TIMEOUT_SECONDS,
    RSubprocessError,
    compute_bc_tsne_embedding_from_arrays,
    compute_liger1_embedding_from_arrays,
    compute_liger1_then_tsne_embedding_from_arrays,
    compute_liger_embedding_from_arrays,
    compute_liger_then_tsne_embedding_from_arrays,
    compute_seurat3_embedding_from_arrays,
    compute_seurat_embedding_from_arrays,
    compute_seurat_then_tsne_embedding_from_arrays,
    r_available,
    read_r_output,
    rscript_command,
    run_r_method,
    write_r_input,
)

MOCK_RSCRIPT = Path(__file__).resolve().parent / "mock_rscript.py"
SEURAT_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src/ne_aligned_utils/methods/external/r_scripts/seurat.R"
)
LIGER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "src/ne_aligned_utils/methods/external/r_scripts/liger.R"
)


def test_default_timeout_covers_full_yao_runs():
    assert STANDARD_TIMEOUT_SECONDS == 14400


@pytest.fixture
def mock_r(monkeypatch):
    """Point the R launcher at the mock Rscript for the duration of a test."""
    monkeypatch.setenv(RSCRIPT_ENV_VAR, f"{sys.executable} {MOCK_RSCRIPT}")
    monkeypatch.setenv(LIGER_RSCRIPT_ENV_VAR, f"{sys.executable} {MOCK_RSCRIPT}")


def test_write_and_read_input_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(6, 4))
    x1 = rng.normal(size=(9, 4))
    batch_ids = np.concatenate([np.zeros(6, int), np.ones(9, int)])
    path = tmp_path / "input.h5"

    write_r_input(path, x0, x1, batch_ids, {"method": "seurat", "dim": 2, "seed": 3})

    with h5py.File(path, "r") as handle:
        np.testing.assert_allclose(np.asarray(handle["x0"]), x0)
        np.testing.assert_allclose(np.asarray(handle["x1"]), x1)
        np.testing.assert_array_equal(np.asarray(handle["batch_ids"]), batch_ids)
        attrs = handle["params"].attrs
        assert attrs["method"] == "seurat"
        assert int(attrs["dim"]) == 2
        assert int(attrs["seed"]) == 3
        assert int(attrs["n0"]) == 6 and int(attrs["n1"]) == 9 and int(attrs["n_features"]) == 4


def test_read_r_output_handles_r_transpose(tmp_path):
    path = tmp_path / "output.h5"
    # R/column-major can hand back a (dim, n_total) array; read_r_output must fix it.
    stored = np.arange(2 * 15, dtype=float).reshape(2, 15)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("embedding", data=stored)

    embedding = read_r_output(path, n_total=15)
    assert embedding.shape == (15, 2)
    np.testing.assert_allclose(embedding, stored.T)


def test_read_r_output_rejects_wrong_row_count(tmp_path):
    path = tmp_path / "output.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("embedding", data=np.zeros((11, 2)))
    with pytest.raises(RSubprocessError):
        read_r_output(path, n_total=15)


def test_run_r_method_with_mock_returns_aligned_embedding(mock_r):
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=(10, 6))
    x1 = rng.normal(size=(14, 6))
    batch_ids = np.concatenate([np.zeros(10, int), np.ones(14, int)])

    embedding, artifacts = run_r_method(
        "seurat.R", x0, x1, batch_ids, {"method": "seurat", "dim": 4, "seed": 0}
    )
    assert embedding.shape == (24, 4)
    assert np.isfinite(embedding).all()
    assert artifacts["r_script"] == "seurat.R"


def test_run_r_method_raises_without_r(monkeypatch):
    monkeypatch.delenv(RSCRIPT_ENV_VAR, raising=False)
    with pytest.raises(RSubprocessError, match=RSCRIPT_ENV_VAR):
        run_r_method(
            "seurat.R",
            np.zeros((3, 2)),
            np.zeros((3, 2)),
            np.array([0, 0, 0, 1, 1, 1]),
            {"method": "seurat", "dim": 2, "seed": 0},
        )


def test_compute_seurat_2d_via_mock(mock_r):
    rng = np.random.default_rng(2)
    x0 = rng.normal(size=(12, 8))
    x1 = rng.normal(size=(15, 8))
    embedding, artifacts = compute_seurat_embedding_from_arrays(
        x0, x1, embd_dim=2, random_state=0, return_artifacts=True
    )
    assert embedding.shape == (27, 2)
    assert np.isfinite(embedding).all()
    assert artifacts["seurat_dim"] == 2
    assert artifacts["left_embedding"].shape == (12, 2)
    assert artifacts["right_embedding"].shape == (15, 2)
    assert artifacts["workflow"] == "v5_cca"
    assert artifacts["preselected_features"] is True


def test_compute_seurat_accepts_preselected_feature_panel(mock_r):
    rng = np.random.default_rng(20)
    x0 = rng.poisson(2.0, size=(12, 8)).astype(float)
    x1 = rng.poisson(2.0, size=(15, 8)).astype(float)
    embedding, artifacts = compute_seurat_embedding_from_arrays(
        x0,
        x1,
        embd_dim=2,
        random_state=0,
        preselected_features=True,
        return_artifacts=True,
    )
    assert embedding.shape == (27, 2)
    assert artifacts["preselected_features"] is True


def test_compute_seurat3_preserves_the_historical_workflow(mock_r):
    rng = np.random.default_rng(21)
    x0 = rng.poisson(2.0, size=(12, 8)).astype(float)
    x1 = rng.poisson(2.0, size=(15, 8)).astype(float)
    embedding, artifacts = compute_seurat3_embedding_from_arrays(
        x0,
        x1,
        embd_dim=2,
        random_state=0,
        return_artifacts=True,
    )
    assert embedding.shape == (27, 2)
    assert artifacts["workflow"] == "legacy"
    assert artifacts["preselected_features"] is False


def test_compute_seurat_rejects_unknown_workflow(mock_r):
    with pytest.raises(ValueError, match="workflow"):
        compute_seurat_embedding_from_arrays(np.ones((3, 4)), np.ones((3, 4)), workflow="unknown")


def test_seurat_pca_uses_the_benchmark_seed():
    source = SEURAT_SCRIPT.read_text()
    assert source.count("seed.use = seed") == 3
    assert "set.seed(seed)" in source


def test_compute_seurat_50_tsne_applies_python_tail(mock_r):
    rng = np.random.default_rng(3)
    x0 = rng.normal(size=(20, 10))
    x1 = rng.normal(size=(25, 10))
    embedding, artifacts = compute_seurat_then_tsne_embedding_from_arrays(
        x0, x1, seurat_dim=8, random_state=0, perplexity=5.0, n_iter=12, return_artifacts=True
    )
    assert embedding.shape == (45, 2)  # R gives 8D, Python t-SNE -> 2D
    assert np.isfinite(embedding).all()
    assert artifacts["integration_embedding"].shape == (45, 8)
    assert artifacts["seurat_dim"] == 8


def test_compute_liger1_variants_via_mock(mock_r):
    rng = np.random.default_rng(4)
    x0 = rng.normal(size=(14, 9))
    x1 = rng.normal(size=(18, 9))

    direct = compute_liger1_embedding_from_arrays(x0, x1, embd_dim=2, random_state=0)
    assert direct.shape == (32, 2)

    tsne = compute_liger1_then_tsne_embedding_from_arrays(
        x0, x1, liger_dim=6, random_state=0, perplexity=5.0, n_iter=12
    )
    assert tsne.shape == (32, 2)
    assert np.isfinite(tsne).all()


def test_compute_liger_variants_via_mock(mock_r):
    rng = np.random.default_rng(24)
    x0 = rng.poisson(2.0, size=(14, 9)).astype(float)
    x1 = rng.poisson(2.0, size=(18, 9)).astype(float)

    direct, artifacts = compute_liger_embedding_from_arrays(
        x0, x1, embd_dim=2, random_state=7, return_artifacts=True
    )
    assert direct.shape == (32, 2)
    assert artifacts["preselected_features"] is True
    assert artifacts["alignment"] == "quantileNorm"
    assert artifacts["random_state"] == 7
    assert artifacts["n_cores"] == 2

    _, artifacts = compute_liger_embedding_from_arrays(
        x0, x1, embd_dim=2, random_state=7, return_artifacts=True, n_cores=4
    )
    assert artifacts["n_cores"] == 4

    embedded = compute_liger_then_tsne_embedding_from_arrays(
        x0, x1, liger_dim=6, random_state=7, perplexity=5.0, n_iter=12
    )
    assert embedded.shape == (32, 2)
    assert np.isfinite(embedded).all()


@pytest.mark.parametrize("n_cores", [0, -1, 1.5, True])
def test_compute_liger_rejects_invalid_core_counts(mock_r, n_cores):
    x0 = np.ones((6, 4))
    x1 = np.ones((5, 4))
    with pytest.raises(ValueError, match="n_cores"):
        compute_liger_embedding_from_arrays(x0, x1, n_cores=n_cores)
    with pytest.raises(ValueError, match="n_cores"):
        compute_liger_then_tsne_embedding_from_arrays(x0, x1, n_cores=n_cores)


def test_current_liger_only_overrides_features_dimension_seed_and_cores():
    source = LIGER_SCRIPT.read_text()
    assert "varFeatures(liger_object) <- rownames(data_list$b0)" in source
    assert "liger_object, k = k, seed = seed, nCores = n_cores, verbose = FALSE" in source
    # Callers that send no core count get rliger's own default.
    assert "n_cores <- 2L" in source
    assert "alignFactors(liger_object, seed = seed)" in source
    assert "removeMissing =" not in source
    assert "method =" not in source
    assert "lambda =" not in source


def test_liger1_seed_controls_both_stochastic_stages_by_default(mock_r):
    rng = np.random.default_rng(23)
    x0 = rng.poisson(2.0, size=(14, 9)).astype(float)
    x1 = rng.poisson(2.0, size=(18, 9)).astype(float)

    _, artifacts = compute_liger1_embedding_from_arrays(
        x0, x1, embd_dim=2, random_state=7, return_artifacts=True
    )

    assert artifacts["random_state"] == 7
    assert artifacts["couple_quantile_seed"] is True


def test_compute_liger1_accepts_opt_in_feature_and_seed_controls(mock_r):
    rng = np.random.default_rng(22)
    x0 = rng.poisson(2.0, size=(14, 9)).astype(float)
    x1 = rng.poisson(2.0, size=(18, 9)).astype(float)
    embedding, artifacts = compute_liger1_embedding_from_arrays(
        x0,
        x1,
        embd_dim=2,
        random_state=3,
        preselected_features=True,
        couple_quantile_seed=True,
        return_artifacts=True,
    )
    assert embedding.shape == (32, 2)
    assert artifacts["preselected_features"] is True
    assert artifacts["couple_quantile_seed"] is True


def test_compute_liger1_rejects_conflicting_feature_controls(mock_r):
    with pytest.raises(ValueError, match="cannot both"):
        compute_liger1_embedding_from_arrays(
            np.ones((3, 4)),
            np.ones((3, 4)),
            preselected_features=True,
            requested_features=3,
        )


def test_compute_bc_tsne_via_mock(mock_r):
    rng = np.random.default_rng(5)
    x0 = rng.normal(size=(11, 7))
    x1 = rng.normal(size=(13, 7))
    embedding, artifacts = compute_bc_tsne_embedding_from_arrays(
        x0, x1, random_state=0, perplexity=5.0, return_artifacts=True
    )
    assert embedding.shape == (24, 2)
    assert np.isfinite(embedding).all()
    assert artifacts["perplexity"] == 5.0


def test_r_available_reflects_override(monkeypatch):
    monkeypatch.setenv(RSCRIPT_ENV_VAR, "some-launcher")
    assert r_available() is True
    monkeypatch.delenv(RSCRIPT_ENV_VAR, raising=False)
    assert r_available() is False
    # Blank, or whitespace, names nothing. Both used to reach the PATH fallback.
    monkeypatch.setenv(RSCRIPT_ENV_VAR, "   ")
    assert r_available() is False


def test_no_launcher_means_no_r_even_when_rscript_is_on_path(monkeypatch, tmp_path):
    """The regression this file exists to prevent.

    `rscript_command` used to return ``["Rscript"]`` when the variable was unset, so a machine
    with any R on PATH ran the R methods against it silently -- the wrong package versions, or a
    plain R with no Seurat at all, and no way to tell from the output afterwards. PATH is now
    consulted only when the caller names it.
    """
    stub = tmp_path / "Rscript"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.delenv(RSCRIPT_ENV_VAR, raising=False)

    msg = "the stub has to be findable, or this test would pass for the wrong reason"
    assert shutil.which("Rscript") == str(stub), msg
    assert r_available() is False
    with pytest.raises(RSubprocessError, match="no R launcher has been named"):
        rscript_command()

    # And naming it is all it takes to opt back in.
    monkeypatch.setenv(RSCRIPT_ENV_VAR, "Rscript")
    assert r_available() is True
    assert rscript_command() == ["Rscript"]


# --- Opt-in real-R integration -------------------------------------------------
# Skipped unless NE_ALIGNED_TEST_REAL_R=1 and NE_ALIGNED_RSCRIPT points at an R
# launcher with Seurat / rliger (1.0.x) / bcTSNE / hdf5r installed (see
# src/ne_aligned_utils/methods/external/README.md). Lets a real R setup be re-verified:
#   NE_ALIGNED_TEST_REAL_R=1 NE_ALIGNED_RSCRIPT="Rscript" pytest -k real_r
import os

real_r = pytest.mark.skipif(
    os.environ.get("NE_ALIGNED_TEST_REAL_R") != "1",
    reason="set NE_ALIGNED_TEST_REAL_R=1 (and NE_ALIGNED_RSCRIPT) to run real-R integration",
)


def _raw_counts(seed):
    rng = np.random.default_rng(seed)
    n0, n1, g = 120, 140, 400
    base = rng.gamma(2.0, 1.0, size=(1, g))
    raw0 = rng.poisson(rng.gamma(2.0, 1.0, size=(n0, g)) * base).astype(float)
    raw1 = rng.poisson(
        rng.gamma(2.0, 1.0, size=(n1, g)) * base * rng.uniform(0.5, 1.5, (1, g))
    ).astype(float)
    return raw0, raw1


@real_r
def test_real_r_seurat_and_liger():
    raw0, raw1 = _raw_counts(0)
    n_total = raw0.shape[0] + raw1.shape[0]
    for embedding in (
        compute_seurat_embedding_from_arrays(raw0, raw1, embd_dim=2, random_state=0),
        compute_seurat_then_tsne_embedding_from_arrays(
            raw0, raw1, seurat_dim=30, random_state=0, perplexity=15.0, n_iter=250
        ),
        compute_liger1_embedding_from_arrays(raw0, raw1, embd_dim=2, random_state=0),
        compute_liger1_then_tsne_embedding_from_arrays(
            raw0, raw1, liger_dim=20, random_state=0, perplexity=15.0, n_iter=250
        ),
    ):
        embedding = np.asarray(embedding)
        assert embedding.shape == (n_total, 2)
        assert np.isfinite(embedding).all()


@real_r
def test_real_r_bc_tsne():
    raw0, raw1 = _raw_counts(1)
    # bc-tsne consumes processed features
    proc0 = np.log1p(raw0 / (raw0.sum(1, keepdims=True) + 1e-9) * 1e4)
    proc1 = np.log1p(raw1 / (raw1.sum(1, keepdims=True) + 1e-9) * 1e4)
    embedding = np.asarray(
        compute_bc_tsne_embedding_from_arrays(proc0, proc1, random_state=0, perplexity=15.0)
    )
    assert embedding.shape == (raw0.shape[0] + raw1.shape[0], 2)
    assert np.isfinite(embedding).all()
