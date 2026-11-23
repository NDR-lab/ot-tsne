"""Check the rc-tsne binary contract and conditioning-parameter mapping.

Use mock_rctsne_binary unless real_rctsne is requested, parsing serialized inputs
to verify what the driver passes to the executable.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from ne_aligned_utils.config import DEFAULT_RC_TSNE_BETA, RC_TSNE_MAX_ITER, RC_TSNE_THETA
from ne_aligned_utils.methods import compute_tsne_embedding_from_arrays
from ne_aligned_utils.methods.external.rctsne import (
    RCTSNE_BIN_ENV_VAR,
    RcTsneSubprocessError,
    _joint_pca,
    check_split_neighbour_budget,
    compute_rc_tsne_embedding_from_arrays,
    rctsne_available,
    rctsne_command,
    resolve_conditioning,
)

MOCK_BINARY = Path(__file__).resolve().parent / "mock_rctsne_binary.py"

real_rctsne = pytest.mark.skipif(
    not os.environ.get("NE_ALIGNED_TEST_REAL_RCTSNE"),
    reason="set NE_ALIGNED_TEST_REAL_RCTSNE=1 to run against the compiled rc-tsne binary",
)


@pytest.fixture
def mock_rctsne(monkeypatch):
    """Point the rc-tsne launcher at the mock binary for the duration of a test."""
    monkeypatch.setenv(RCTSNE_BIN_ENV_VAR, f"{sys.executable} {MOCK_BINARY}")


def _inputs(n0=30, n1=20, n_features=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n0, n_features)), rng.normal(size=(n1, n_features)) + 0.5


# Locating the binary


def test_command_honours_a_multi_token_override(monkeypatch):
    """Shell-style tokenisation is what lets a mock stand in for a binary."""
    monkeypatch.setenv(RCTSNE_BIN_ENV_VAR, "python /tmp/mock.py")
    assert rctsne_command() == ["python", "/tmp/mock.py"]


def test_command_rejects_an_empty_override(monkeypatch):
    monkeypatch.setenv(RCTSNE_BIN_ENV_VAR, "   ")
    with pytest.raises(RcTsneSubprocessError, match="set but empty"):
        rctsne_command()


def test_available_is_false_when_the_binary_is_missing(monkeypatch):
    monkeypatch.setenv(RCTSNE_BIN_ENV_VAR, "/nonexistent/fast_ctsne")
    assert rctsne_available() is False


def test_a_missing_binary_names_the_build_script(monkeypatch):
    """The failure a new machine hits first, so it has to say what to do about it."""
    monkeypatch.setenv(RCTSNE_BIN_ENV_VAR, "/nonexistent/fast_ctsne")
    x0, x1 = _inputs()
    with pytest.raises(RcTsneSubprocessError, match="build_rctsne.sh"):
        compute_rc_tsne_embedding_from_arrays(x0, x1, perplexity=5)


# The alpha / beta inversion


def test_alpha_is_derived_the_way_their_paper_defines_it():
    """Pins the convention in executable form, because prose about it is easy to misread.

    ``beta`` is the paper's: the small number that suppresses same-label affinity. ``alpha`` is the
    opposite-label multiplier, solving ``beta * ratio + alpha * (1 - ratio) = 1``.
    """
    labels = np.array([0] * 600 + [1] * 900)
    n = labels.size
    ratio_expected = (600 * 599 + 900 * 899) / (n * (n - 1))
    beta, alpha, ratio = resolve_conditioning(labels, 1e-4)

    assert beta == 1e-4
    assert ratio == pytest.approx(ratio_expected)
    assert alpha == pytest.approx((1 - 1e-4 * ratio_expected) / (1 - ratio_expected))
    # The suppression that beta = 1e-4 actually buys, which is the number worth knowing.
    assert alpha / beta == pytest.approx(2.08e4, rel=0.05)


def test_beta_one_is_ordinary_tsne():
    """Their own documented special case: at beta = 1 both multipliers are 1."""
    labels = np.array([0] * 5 + [1] * 7)
    beta, alpha, _ = resolve_conditioning(labels, 1.0)
    assert (beta, alpha) == (1.0, 1.0)


@pytest.mark.parametrize("bad", [0.0, -1e-6, 1.5, 2.0])
def test_beta_outside_the_upstream_bound_is_refused(bad):
    """Their driver asserts ``0 < alpha <= 1``. Zero is invalid there, not a limiting case."""
    labels = np.array([0] * 5 + [1] * 5)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        resolve_conditioning(labels, bad)


def test_conditioning_needs_two_labels():
    with pytest.raises(ValueError, match="at least two label values"):
        resolve_conditioning(np.zeros(10, dtype=int), 1e-4)


def test_the_binary_receives_beta_as_its_alpha_field(mock_rctsne, tmp_path, monkeypatch):
    """The end-to-end inversion check, read off the real .dat rather than the call site.

    If these two were ever swapped, rc-tsne would *raise* same-batch affinity by four orders of
    magnitude and still produce a plausible-looking picture. Nothing else in the suite would catch
    it, which is why this test reads the bytes the binary was given.
    """
    dump = tmp_path / "handed.json"
    monkeypatch.setenv("NE_ALIGNED_MOCK_RCTSNE_DUMP", str(dump))
    x0, x1 = _inputs(n0=30, n1=20)
    compute_rc_tsne_embedding_from_arrays(x0, x1, perplexity=5, beta=1e-4)

    handed = json.loads(dump.read_text())
    _, alpha_expected, _ = resolve_conditioning(
        np.concatenate([np.zeros(30, int), np.ones(20, int)]), 1e-4
    )
    assert handed["alpha"] == pytest.approx(1e-4), "their `alpha` must carry the paper's beta"
    assert handed["beta"] == pytest.approx(alpha_expected), "their `beta` is the derived multiplier"
    assert handed["num_label_vals"] == 2
    assert handed["labels"] == [0] * 30 + [1] * 20
    # The prespecified settings, asserted where they actually land.
    assert handed["max_iter"] == RC_TSNE_MAX_ITER
    assert handed["theta"] == pytest.approx(RC_TSNE_THETA)
    assert handed["perplexity"] == pytest.approx(5.0)
    assert handed["map_dims"] == 2


# The split neighbor budget


def test_split_budget_refusal_names_both_batches():
    """Their driver checks only the joint size; the method needs half the budget per batch."""
    with pytest.raises(ValueError, match=r"45 same-batch neighbours.*20 and 900"):
        check_split_neighbour_budget(20, 900, 30)


def test_split_budget_allows_a_feasible_perplexity():
    assert check_split_neighbour_budget(3000, 6000, 30) is None


def test_split_budget_is_checked_before_the_binary_runs(monkeypatch):
    """No binary needed: the guard must fire first, so it works on a machine with no build."""
    monkeypatch.setenv(RCTSNE_BIN_ENV_VAR, "/nonexistent/fast_ctsne")
    x0, x1 = _inputs(n0=10, n1=10)
    with pytest.raises(ValueError, match="same-batch neighbours"):
        compute_rc_tsne_embedding_from_arrays(x0, x1, perplexity=30)


# The PCA is benchmark-standardised, and must match joint exactly


def test_joint_pca_is_byte_identical_to_joint_tsne():
    """``_joint_pca`` repeats the rule rather than sharing it, so this is what keeps them equal."""
    x0, x1 = _inputs(n0=60, n1=40, n_features=80)
    ours, meta = _joint_pca(np.concatenate([x0, x1], axis=0), 50, 0)
    _, artifacts = compute_tsne_embedding_from_arrays(
        x0, x1, random_state=0, perplexity=5, return_artifacts=True
    )
    np.testing.assert_array_equal(ours, np.asarray(artifacts["pca_embedding"], dtype=float))
    assert meta["pca_components"] == 50


def test_joint_pca_skips_when_the_input_is_already_narrow():
    """The yao case: 50 columns in, 50 asked for, so the step is a no-op there as for joint."""
    joint = np.concatenate(_inputs(n0=60, n1=40, n_features=50), axis=0)
    reduced, meta = _joint_pca(joint, 50, 0)
    assert meta["pca_components"] is None
    np.testing.assert_array_equal(reduced, joint)


def test_joint_pca_rejects_a_fractional_width():
    joint = np.concatenate(_inputs(), axis=0)
    with pytest.raises(ValueError, match="integer or None"):
        _joint_pca(joint, 12.5, 0)


# Running, and the ways a run can go wrong


def test_embeds_two_batches_through_the_mock(mock_rctsne):
    x0, x1 = _inputs(n0=30, n1=20)
    embedding, artifacts = compute_rc_tsne_embedding_from_arrays(
        x0, x1, perplexity=5, return_artifacts=True
    )
    assert embedding.shape == (50, 2)
    assert np.isfinite(embedding).all()
    # The mock writes (row index, label), so row order survived the round trip and the split is
    # batch 0 first -- the contract every method in this repo shares.
    np.testing.assert_array_equal(embedding[:, 1], np.concatenate([np.zeros(30), np.ones(20)]))
    assert artifacts["left_embedding"].shape == (30, 2)
    assert artifacts["right_embedding"].shape == (20, 2)
    assert artifacts["rc_beta"] == DEFAULT_RC_TSNE_BETA
    assert artifacts["rc_loss"].shape == (RC_TSNE_MAX_ITER,)
    assert artifacts["embedding_backend"] == "rctsne"


def test_a_failing_binary_raises_with_its_stderr(mock_rctsne, monkeypatch):
    monkeypatch.setenv("NE_ALIGNED_MOCK_RCTSNE_FAIL", "1")
    x0, x1 = _inputs()
    with pytest.raises(RcTsneSubprocessError, match="exited 3"):
        compute_rc_tsne_embedding_from_arrays(x0, x1, perplexity=5)


def test_a_binary_that_exits_zero_without_writing_output_raises(mock_rctsne, monkeypatch):
    """The real binary does exactly this when it cannot read its input, so it must not pass.

    Upstream would have surfaced it as a bare FileNotFoundError from an open() call; vendor patch 3
    turns it into a named error. See vendor/rctsne/VENDOR.md.
    """
    monkeypatch.setenv("NE_ALIGNED_MOCK_RCTSNE_SILENT", "1")
    x0, x1 = _inputs()
    with pytest.raises(RcTsneSubprocessError, match="wrote no output"):
        compute_rc_tsne_embedding_from_arrays(x0, x1, perplexity=5)


@pytest.mark.parametrize(
    "env,value,match",
    [
        ("NE_ALIGNED_MOCK_RCTSNE_WIDTH", "3", "width 3"),
        ("NE_ALIGNED_MOCK_RCTSNE_ROWS", "17", "17 rows"),
        ("NE_ALIGNED_MOCK_RCTSNE_NAN", "1", "non-finite"),
    ],
)
def test_malformed_output_is_rejected(mock_rctsne, monkeypatch, env, value, match):
    monkeypatch.setenv(env, value)
    x0, x1 = _inputs()
    with pytest.raises(RcTsneSubprocessError, match=match):
        compute_rc_tsne_embedding_from_arrays(x0, x1, perplexity=5)


def test_mismatched_feature_counts_are_rejected(mock_rctsne):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        compute_rc_tsne_embedding_from_arrays(
            rng.normal(size=(20, 10)), rng.normal(size=(20, 11)), perplexity=5
        )


def test_temporary_files_do_not_land_in_the_working_directory(mock_rctsne, tmp_path, monkeypatch):
    """Vendor patch 1. Upstream wrote data_*.dat into os.getcwd(), so concurrent sweep tasks
    collided and a sweep left debris in the repository root."""
    monkeypatch.chdir(tmp_path)
    x0, x1 = _inputs()
    compute_rc_tsne_embedding_from_arrays(x0, x1, perplexity=5)
    assert list(tmp_path.iterdir()) == []


# Against the compiled binary


@real_rctsne
def test_real_binary_is_deterministic_per_seed():
    x0, x1 = _inputs(n0=120, n1=150, n_features=40)
    first = compute_rc_tsne_embedding_from_arrays(x0, x1, random_state=0, perplexity=10)
    second = compute_rc_tsne_embedding_from_arrays(x0, x1, random_state=0, perplexity=10)
    np.testing.assert_array_equal(first, second)
    other = compute_rc_tsne_embedding_from_arrays(x0, x1, random_state=7, perplexity=10)
    assert not np.allclose(first, other)


@real_rctsne
def test_the_patched_driver_matches_the_unpatched_one(tmp_path):
    """Proves patches 1 to 3 are plumbing and nothing else.

    It says nothing about the binary itself -- both drivers invoke the same one. The uninitialised
    ``sum_P`` question is settled in VENDOR.md, not here.
    """
    import importlib.util

    from ne_aligned_utils.methods.external.rctsne import VENDOR_DIR, rctsne_command, run_rctsne

    x0, x1 = _inputs(n0=80, n1=90, n_features=20)
    joint = np.concatenate([x0, x1], axis=0)
    labels = np.concatenate([np.zeros(80, np.int32), np.ones(90, np.int32)])

    patched, _ = run_rctsne(joint, labels, perplexity=10, random_state=0, return_loss=True)

    # Upstream resolves its binary from its own directory, so it is copied next to a bin/ that
    # holds the same binary rather than being pointed at one.
    work = tmp_path / "upstream"
    (work / "bin").mkdir(parents=True)
    shutil.copy(VENDOR_DIR / "fast_ctsne_upstream.py", work / "fast_ctsne.py")
    shutil.copy(rctsne_command()[0], work / "bin" / "fast_ctsne")
    spec = importlib.util.spec_from_file_location("upstream_fc", work / "fast_ctsne.py")
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)

    cwd = os.getcwd()
    os.chdir(work)
    try:
        reference = upstream.fast_ctsne(
            joint,
            labels,
            alpha=DEFAULT_RC_TSNE_BETA,
            perplexity=10,
            theta=RC_TSNE_THETA,
            max_iter=RC_TSNE_MAX_ITER,
            stop_early_exag_iter=250,
            keep_nonzero=3,
            initialization="pca",
            fixed_perp=True,
            seed=0,
            nthreads=1,
            print_output=False,
        )
    finally:
        os.chdir(cwd)

    np.testing.assert_array_equal(np.asarray(patched), np.asarray(reference))
