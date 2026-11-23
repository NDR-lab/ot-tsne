"""Check timing schemas, stage ownership, reuse semantics, and transparent instrumentation.

Assert measured-field contracts rather than elapsed values or additive identities:
nested wall-clock stages overlap and include measurement overhead. Wrappers must
preserve callable signatures and numerical results.
"""

from __future__ import annotations

import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from ne_aligned_utils import timing
from ne_aligned_utils.benchmark.sweep import TASK_KINDS, run_task, save_npz
from ne_aligned_utils.methods.external.r import RSCRIPT_ENV_VAR

MOCK_RSCRIPT = Path(__file__).resolve().parent / "mock_rscript.py"

#: A tolerance for "a child stage does not exceed its parent". Generous on purpose: the point is to
#: catch a stage attributed to the wrong parent, not to measure timer overhead.
TOLERANCE_SECONDS = 0.5


# The schema itself


def test_every_task_kind_has_a_schema():
    """A new method must be classified, not silently defaulted into a shape that misdescribes it."""
    assert set(TASK_KINDS) == set(timing.TASK_SCHEMAS)
    assert set(timing.TASK_SCHEMAS.values()) == {"ot", "two_stage", "fused"}


def test_an_unknown_task_kind_is_refused():
    with pytest.raises(KeyError, match="has no timing schema"):
        timing.resolve_schema("not-a-method")


def test_fused_methods_report_a_total_and_nothing_else():
    """The point of the fused schema: no invented stages.

    ``joint-pca`` is a single PCA call and ``rc-tsne`` is an opaque binary. Neither has a boundary
    to report, and a ``0.0`` in an optimization field would read as "measured, took no time".
    """
    timer = timing.StageTimer()
    # Even if a stage were somehow recorded, the schema must drop it.
    timer.record("time_pre_embedding", 1.0)
    timer.record("time_embedding_optimization", 2.0)
    fields = timing.finalise(timer, schema="fused", wall_outer=3.0, task_kind="joint-pca")
    assert fields["time_total"] == 3.0
    for name in (*timing.STAGE_FIELDS, *timing.OT_FIELDS, *timing.EMBEDDING_DIAGNOSTIC_FIELDS):
        assert math.isnan(fields[name]), name


def test_non_applicable_stages_are_nan_and_never_zero():
    """NaN and 0.0 are different claims, and both occur, so they must stay distinguishable."""
    timer = timing.StageTimer()
    timer.record("time_pre_embedding", 0.0)  # a real, genuinely instantaneous measurement
    fields = timing.finalise(timer, schema="two_stage", wall_outer=0.5, task_kind="cca-50")
    assert fields["time_pre_embedding"] == 0.0
    assert math.isnan(fields["time_embedding"])


def test_reuse_forfeits_the_total_but_keeps_the_embedding():
    """A reuse run loaded a representation that normally has to be computed.

    So it is not an observation of the canonical total, and a number there would be a fake. The
    embedding stage did genuinely run, so that one stays -- the row is honest about being useful
    for a backend-only comparison and useless for a total.
    """
    timer = timing.StageTimer()
    timer.record("time_pre_embedding", 0.1)
    timer.record("time_embedding", 0.2)
    fields = timing.finalise(
        timer,
        schema="two_stage",
        wall_outer=0.3,
        task_kind="cca-50",
        representation_reused=True,
    )
    assert math.isnan(fields["time_total"])
    assert math.isnan(fields["time_pre_embedding"])
    assert fields["time_embedding"] == 0.2


def test_child_timed_methods_take_their_total_from_the_child():
    """Ownership per the table: the parent's number would include R startup and library import."""
    timer = timing.StageTimer()
    timer.set("time_pre_embedding", 1.0)
    timer.record("time_embedding", 0.5)
    fields = timing.finalise(timer, schema="two_stage", wall_outer=9.0, task_kind="seurat-50")
    assert fields["time_total"] == pytest.approx(1.5)
    assert fields["time_wall_outer"] == 9.0
    assert fields["time_excluded_overhead"] == pytest.approx(7.5)


@pytest.mark.parametrize(
    "task_kind, reported",
    [
        ("seurat-2d", "time_pre_embedding"),
        ("liger1-2d", "time_pre_embedding"),
        ("liger-2d", "time_pre_embedding"),
        ("bc-tsne", "time_total"),
        ("multimap", "time_total"),
    ],
)
def test_fused_children_are_credited_whichever_field_they_report(task_kind, reported):
    """The R scripts do not all name their one number the same thing.

    ``seurat.R`` and ``liger.R`` serve both the 2D and the 50-dimensional form, so they report
    ``time_pre_embedding`` either way -- and for the 2D form, with no Python tail, that number *is*
    the total. Reading only ``time_total`` left both 2D methods silently falling back to the
    parent's measurement, which includes R's startup and ``library(Seurat)``. That is the one
    failure child timing exists to prevent, so it is pinned here.
    """
    timer = timing.StageTimer()
    timer.set(reported, 500.0)
    fields = timing.finalise(timer, schema="fused", wall_outer=3.0, task_kind=task_kind)
    assert fields["time_total"] == pytest.approx(500.0)
    assert fields["time_wall_outer"] == 3.0


def test_a_child_that_reports_nothing_falls_back_to_the_parent():
    """A mock interpreter, or a build predating the child timer.

    Falling back keeps the total rather than losing it; the NaN excluded-overhead is what marks the
    row as un-separated.
    """
    fields = timing.finalise(
        timing.StageTimer(), schema="fused", wall_outer=4.0, task_kind="bc-tsne"
    )
    assert fields["time_total"] == 4.0
    assert math.isnan(fields["time_excluded_overhead"])


def test_rc_tsne_is_parent_timed_and_says_so():
    """The documented exception: its binary exposes no algorithm-only boundary."""
    assert "rc-tsne" not in timing.CHILD_TIMED_METHODS
    fields = timing.finalise(
        timing.StageTimer(), schema="fused", wall_outer=2.0, task_kind="rc-tsne"
    )
    assert fields["time_total"] == fields["time_wall_outer"] == 2.0


# The instrumentation changes nothing


def test_instrumentation_leaves_both_libraries_as_it_found_them():
    from openTSNE import affinity, tsne as tsne_module
    from umap import umap_

    watched = [
        (affinity.MultiscaleMixture, "__init__"),
        (affinity.MultiscaleMixture, "to_new"),
        (tsne_module.TSNEEmbedding, "optimize"),
        (tsne_module.PartialTSNEEmbedding, "optimize"),
        (tsne_module.initialization_scheme, "pca"),
        (umap_, "nearest_neighbors"),
        (umap_, "fuzzy_simplicial_set"),
        (umap_, "spectral_layout"),
        (umap_, "optimize_layout_euclidean"),
    ]
    before = [getattr(owner, name) for owner, name in watched]
    with timing.instrument_embedders(timing.StageTimer()):
        pass
    after = [getattr(owner, name) for owner, name in watched]
    assert all(a is b for a, b in zip(before, after))


def test_instrumentation_is_transparent_to_introspection():
    """openTSNE's ``transform`` reads ``to_new``'s signature to decide if it can run.

    A ``(*args, **kwargs)`` wrapper hides the ``perplexity`` parameter it looks for, and ``ref``
    then fails with "transform currently does not support MultiscaleMixture affinities". This is
    that regression, pinned.
    """
    import inspect

    from openTSNE import affinity

    outside = inspect.signature(affinity.MultiscaleMixture.to_new)
    with timing.instrument_embedders(timing.StageTimer()):
        inside = inspect.signature(affinity.MultiscaleMixture.to_new)
    assert inside == outside
    assert "perplexities" in inside.parameters or "perplexity" in inside.parameters


@pytest.mark.parametrize("backend", ["tsne", "umap"])
def test_an_embedding_is_bit_identical_inside_the_instrumentation(backend):
    from ne_aligned_utils.methods import embed_2d

    rng = np.random.RandomState(0)
    x = rng.randn(120, 6)
    kwargs = dict(random_state=0, perplexity=10.0, n_neighbors=8, n_iter=30, n_jobs=1)

    plain = np.asarray(embed_2d(x, backend, **kwargs))
    timer = timing.StageTimer()
    with timing.instrument_embedders(timer):
        timed = np.asarray(embed_2d(x, backend, timer=timer, **kwargs))

    assert np.array_equal(plain, timed)
    assert timer.get("time_embedding") >= 0.0


# End to end, through the ordinary run-task path


@pytest.fixture
def task_args(tmp_path):
    """A small on-disk task namespace, exactly what ``run-task`` builds for itself."""

    dataset_dir = tmp_path / "data" / "nCoV"
    dataset_dir.mkdir(parents=True)
    rng = np.random.RandomState(0)
    x0 = rng.normal(size=(60, 12))
    x1 = rng.normal(size=(55, 12)) + 0.4
    np.save(dataset_dir / "X1.npy", x0)
    np.save(dataset_dir / "X2.npy", x1)
    np.save(dataset_dir / "X1_pca50.npy", x0)
    np.save(dataset_dir / "X2_pca50.npy", x1)

    def build(task_kind: str, **overrides):
        args = Namespace(
            task_kind=task_kind,
            dataset="nCoV",
            data_root=tmp_path / "data",
            output_dir=tmp_path / "out",
            task_output=tmp_path / "out" / f"{task_kind.replace('-', '_')}.npz",
            seed=0,
            embedding_dim=2,
            n_epochs=12,
            pca_components=50,
            perplexity=10.0,
            backend="tsne",
            n_neighbors=None,
            n_jobs=1,
            threads=2,
            require_cpu=True,
            tol=1e-9,
            max_iter=200,
            rel_reg=1.0,
            effective_k=10,
            init="random",
            rc_beta=1e-4,
            multimap_native=False,
            reuse_from=None,
            max_cells_per_batch=40,
            max_cells_batch0=None,
            max_cells_batch1=None,
            force=True,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    return build


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def _finite(payload: dict[str, np.ndarray], name: str) -> float:
    value = float(payload[name])
    assert math.isfinite(value), f"{name} is not finite"
    assert value >= 0.0, f"{name} is negative"
    return value


@pytest.mark.parametrize(
    "task_kind, schema",
    [("ot", "ot"), ("eot-t0", "ot"), ("cca-50", "two_stage"), ("joint-pca", "fused")],
)
def test_a_real_task_writes_its_schema_fields(task_args, task_kind, schema):
    args = task_args(task_kind)
    run_task(args)
    payload = _load(Path(args.task_output))

    assert str(payload["timing_schema"]) == schema
    total = _finite(payload, "time_total")
    assert _finite(payload, "time_wall_outer") == pytest.approx(total)

    if schema == "fused":
        for name in (*timing.STAGE_FIELDS, *timing.OT_FIELDS):
            assert math.isnan(float(payload[name])), name
        return

    pre = _finite(payload, "time_pre_embedding")
    assert pre <= total + TOLERANCE_SECONDS

    if schema == "ot":
        ot_sum = sum(_finite(payload, name) for name in timing.OT_FIELDS)
        assert ot_sum <= pre + TOLERANCE_SECONDS


def test_eot_t0_has_ot_timings_but_no_embedding_stage(task_args):
    """It runs no backend at all, so an embedding stage would be an invention.

    Its OT sub-timings are kept anyway -- that decomposition is the scientific point, and it is the
    one place the direct-2D "no second half, so total only" rule is deliberately not applied.
    """
    args = task_args("eot-t0")
    run_task(args)
    payload = _load(Path(args.task_output))

    assert math.isnan(float(payload["time_embedding"]))
    for name in timing.EMBEDDING_DIAGNOSTIC_FIELDS:
        assert math.isnan(float(payload[name])), name
    for name in timing.OT_FIELDS:
        _finite(payload, name)


@pytest.mark.parametrize(
    "task_kind",
    ["ot", "ot-knndiag", "ot-selfotmedian", "ot-gramdiag", "ot-zerodiag", "ot-zerodiag-split"],
)
def test_the_ot_graph_is_never_charged_twice(task_args, task_kind):
    """The double-counting guard.

    These four hand the backend a finished graph: ``run_tsne_from_graph`` assigns ``affinities.P``
    directly, so ``prepare_initial`` takes its precomputed branch and never builds a
    ``MultiscaleMixture``. The graph's cost therefore belongs to ``time_ot_postprocess`` and to
    nothing else. If a refactor ever routes them through openTSNE's own affinity path, this fails
    rather than quietly reporting the same work under two names.
    """
    args = task_args(task_kind)
    run_task(args)
    payload = _load(Path(args.task_output))

    assert math.isnan(float(payload["time_embedding_graph"]))
    _finite(payload, "time_ot_postprocess")
    # The init is still measured -- these pass an ndarray initialization, so it is timed inside
    # run_tsne_from_graph rather than by the openTSNE wrapper.
    _finite(payload, "time_embedding_initialization")
    _finite(payload, "time_embedding_optimization")


def test_eot_50_does_get_a_backend_graph(task_args):
    """The other side of the same rule: this one hands over coordinates, not a graph."""
    args = task_args("eot-50")
    run_task(args)
    payload = _load(Path(args.task_output))
    _finite(payload, "time_embedding_graph")


def test_nested_stages_stay_within_their_parent(task_args):
    args = task_args("ot")
    run_task(args)
    payload = _load(Path(args.task_output))

    pre = float(payload["time_pre_embedding"])
    embed = float(payload["time_embedding"])
    assert sum(float(payload[n]) for n in timing.OT_FIELDS) <= pre + TOLERANCE_SECONDS
    for name in timing.EMBEDDING_DIAGNOSTIC_FIELDS:
        value = float(payload[name])
        if math.isfinite(value):
            assert value <= embed + TOLERANCE_SECONDS, name
    assert pre + embed <= float(payload["time_total"]) + TOLERANCE_SECONDS


def test_resource_metadata_is_recorded_and_resolvable(task_args):
    args = task_args("joint-pca")
    run_task(args)
    payload = _load(Path(args.task_output))

    assert int(payload["threads_requested"]) == 2
    assert int(payload["n_jobs"]) == 1
    assert int(payload["n_batches"]) == 2
    assert int(payload["n_obs"]) == 80
    assert int(payload["n_features_input"]) > 0
    assert str(payload["device"]) == "cpu"
    assert str(payload["blas_name"])

    fingerprint = str(payload["env_fingerprint"])
    sidecar = Path(args.task_output).parent / f"environment_{fingerprint}.json"
    assert sidecar.exists(), "the compact fingerprint must resolve to a full provenance record"


def test_legacy_total_seconds_is_still_written(task_args):
    """It is what every cached file and the existing figures read; its meaning must not move."""
    args = task_args("joint-pca")
    run_task(args)
    payload = _load(Path(args.task_output))
    assert float(payload["total_seconds"]) > 0.0


def test_a_reuse_run_records_no_total(task_args, tmp_path):
    """End to end: the flag forfeits ``time_total`` in the artifact, not just in the analysis."""
    full = task_args("cca-50", output_dir=tmp_path / "embeddings")
    full.task_output = tmp_path / "embeddings" / "nCoV" / "cca_50_seed_0.npz"
    run_task(full)

    reuse = task_args("cca-50", reuse_from="embeddings", backend="umap", n_neighbors=10)
    reuse.task_output = tmp_path / "umap_out" / "cca_50_umap_seed_0.npz"
    # `load_cached_representation` resolves through the results tree, which this temporary layout
    # does not mirror, so drive `finalise` on the same inputs instead of the full path. The
    # end-to-end reuse plumbing is covered by test_benchmark_sweep.
    timer = timing.StageTimer()
    timer.record("time_embedding", 0.05)
    fields = timing.finalise(
        timer,
        schema="two_stage",
        wall_outer=0.06,
        task_kind="cca-50",
        representation_reused=True,
    )
    assert math.isnan(fields["time_total"])
    assert fields["time_embedding"] == 0.05


def _run_dispatch(args, timer):
    """Drive ``_compute_task`` on synthetic counts, bypassing the dataset loaders.

    Constructing the two batches here keeps these tests on the real dispatch and subprocess
    plumbing without needing a single-cell raw-count fixture on disk.
    """
    from ne_aligned_utils.benchmark.data import BenchmarkData
    from ne_aligned_utils.benchmark.sweep import _compute_task

    rng = np.random.RandomState(0)
    data = BenchmarkData(
        dataset="synthetic",
        x0=rng.poisson(4.0, size=(20, 8)).astype(float),
        x1=rng.poisson(4.0, size=(18, 8)).astype(float),
        labels0=None,
        labels1=None,
        label_names=(),
        dataset_names=("b0", "b1"),
        sample_indices0=np.arange(20),
        sample_indices1=np.arange(18),
        input_source="raw_counts",
    )
    with timing.instrument_embedders(timer):
        return _compute_task(data, args, timer=timer)


def test_seurat_takes_its_total_from_r(task_args, monkeypatch):
    """The two-stage subprocess case, through the mock Rscript rather than a real R install.

    The mock reports an implausibly large algorithmic section, which is the point: a parent
    measurement could not possibly exceed it, so the assertions below can only pass if the child's
    number is the one that became the total.
    """
    monkeypatch.setenv(RSCRIPT_ENV_VAR, f"{sys.executable} {MOCK_RSCRIPT}")
    monkeypatch.setenv("NE_ALIGNED_MOCK_R_ALGO_SECONDS", "1234.5")

    args = task_args("seurat-50")
    timer = timing.StageTimer()
    _run_dispatch(args, timer)
    fields = timing.finalise(timer, schema="two_stage", wall_outer=0.75, task_kind="seurat-50")

    assert timer.get("time_pre_embedding") == pytest.approx(1234.5)
    assert fields["time_total"] > 1234.0, "the child's duration must be the canonical total"
    assert fields["time_wall_outer"] == 0.75
    assert fields["time_wall_outer"] < fields["time_total"], "the parent number is diagnostic here"
    assert math.isfinite(fields["time_embedding"])


def test_r_without_a_reported_duration_still_gets_a_total(task_args, monkeypatch):
    """A mock interpreter, or an R build predating the child timer.

    The total falls back to the parent measurement rather than being lost, and the NaN excluded
    overhead is what marks the row as un-separated.
    """
    monkeypatch.setenv(RSCRIPT_ENV_VAR, f"{sys.executable} {MOCK_RSCRIPT}")
    monkeypatch.delenv("NE_ALIGNED_MOCK_R_ALGO_SECONDS", raising=False)

    args = task_args("seurat-2d")
    timer = timing.StageTimer()
    _run_dispatch(args, timer)
    fields = timing.finalise(timer, schema="fused", wall_outer=0.5, task_kind="seurat-2d")

    assert fields["time_total"] == 0.5
    assert math.isnan(fields["time_excluded_overhead"])


def test_multimap_takes_its_total_from_the_child(task_args, monkeypatch):
    """Same ownership rule across the other subprocess boundary."""
    from ne_aligned_utils.methods.external.multimap import MULTIMAP_PYTHON_ENV_VAR

    mock = Path(__file__).resolve().parent / "mock_multimap_python.py"
    monkeypatch.setenv(MULTIMAP_PYTHON_ENV_VAR, f"{sys.executable} {mock}")
    monkeypatch.setenv("NE_ALIGNED_MOCK_MULTIMAP_ALGO_SECONDS", "999.5")

    # MultiMAP reads n_neighbors explicitly rather than accepting its own 15 * n_datasets default,
    # so the controlled profile needs a number here.
    args = task_args("multimap", n_neighbors=10)
    timer = timing.StageTimer()
    _run_dispatch(args, timer)
    fields = timing.finalise(timer, schema="fused", wall_outer=1.25, task_kind="multimap")

    assert fields["time_total"] == pytest.approx(999.5)
    assert fields["time_wall_outer"] == 1.25


# Thread allocation


def test_thread_env_covers_numba_as_well_as_blas():
    """NUMBA_NUM_THREADS is the one that was missing, and MultiMAP is why it matters."""
    from ne_aligned_utils.config import THREAD_ENV_VARS, thread_env

    assert "NUMBA_NUM_THREADS" in THREAD_ENV_VARS
    assert thread_env(4) == {name: "4" for name in THREAD_ENV_VARS}
    assert thread_env(None) == {}
    with pytest.raises(ValueError, match="at least 1"):
        thread_env(0)


# The analysis reads the canonical field


def test_the_analysis_refuses_a_frame_with_only_the_legacy_total():
    import pandas as pd

    from ne_aligned_utils.paper.summaries import task_timing as R

    legacy = pd.DataFrame({"seconds": [1.0, 2.0], "method_key": ["ot", "joint"]})
    with pytest.raises(R.RuntimeComparabilityError, match="no 'time_total'"):
        R.require_canonical_totals(legacy)

    all_nan = legacy.assign(time_total=[float("nan")] * 2)
    with pytest.raises(R.RuntimeComparabilityError, match="predate the timing schema"):
        R.require_canonical_totals(all_nan)


def test_save_npz_round_trips_nan_timings(tmp_path):
    """NaN has to survive the artifact, or "not applicable" becomes unreadable."""
    path = tmp_path / "task.npz"
    save_npz(path, embedding=np.zeros((3, 2)), time_total=1.5, time_embedding=float("nan"))
    payload = _load(path)
    assert float(payload["time_total"]) == 1.5
    assert math.isnan(float(payload["time_embedding"]))
