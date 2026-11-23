"""Receiving counts differ from outgoing coverage, including unequal-batch and cache cases."""

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from test_parallel_experiment import construct, setup as parallel_setup  # noqa: F401
from test_synthetic_experiment import experiment as orthogonal_setup, run_stage  # noqa: F401

from ne_aligned_utils.evaluation.hubness import (
    cross_batch_occurrences,
    occurrence_statistics,
    restore_saved_construction,
    summarize_coverage,
)
from ne_aligned_utils.paper.figures.hubness import occurrence_histograms


def test_counts_use_receiving_side_with_unequal_batches():
    forward, backward = np.array([[0], [0], [1]]), np.array([[1], [2], [0], [0]])
    rows, columns = np.zeros((3, 4), bool), np.zeros((3, 4), bool)
    rows[np.arange(3), np.arange(3)] = True
    columns[[0, 1, 2, 2], np.arange(4)] = True
    points = cross_batch_occurrences(forward, backward, rows, columns)

    def counts(graph, batch):
        return points.loc[(points.graph == graph) & (points.receiving_batch == batch)].k_occurrence

    np.testing.assert_array_equal(counts("knn_0_to_1", 1), [2, 1, 0, 0])
    np.testing.assert_array_equal(counts("knn_1_to_0", 0), [2, 1, 1])
    np.testing.assert_array_equal(counts("mutual_knn", 1), [1, 1, 0, 0])
    np.testing.assert_array_equal(counts("mutual_knn", 0), [0, 1, 1])
    np.testing.assert_array_equal(counts("shared_knn", 1), [2, 1, 1, 1])
    assert points.groupby(["graph", "receiving_batch"]).normalized_k_occurrence.mean().eq(1).all()
    stats = occurrence_statistics(points)
    assert stats.query("graph == 'knn_0_to_1'").anti_hub_fraction.item() == 0.5
    assert stats.query("graph == 'knn_0_to_1'").receiving_coverage.item() == 0.5
    assert stats.query("graph == 'knn_0_to_1'").skewness.item() == pytest.approx(
        0.28125 / 0.6875**1.5
    )
    assert stats.query("graph in ['sparse_ot', 'shared_knn']").receiving_coverage.eq(1).all()
    assert np.isnan(stats.query("graph == 'sparse_ot' and receiving_batch == 1").skewness.item())
    fig = occurrence_histograms(points)
    assert len(fig.axes) == 5
    assert len({ax.get_xlim() for ax in fig.axes}) == 1
    plt.close(fig)


def test_empty_mutual_support_is_not_normalized_to_fake_ones():
    forward, backward = np.array([[0], [1]]), np.array([[1], [0]])
    points = cross_batch_occurrences(
        forward, backward, np.eye(2, dtype=bool), np.eye(2, dtype=bool)
    )
    mutual = points.loc[points.graph == "mutual_knn"]
    assert mutual.k_occurrence.eq(0).all()
    assert mutual.normalized_k_occurrence.isna().all()
    assert occurrence_statistics(mutual).receiving_coverage.eq(0).all()
    raw = pd.DataFrame(
        dict(
            mode=["original"] * 2,
            magnitude=[10.0] * 2,
            graph=["knn_0_to_1"] * 2,
            receiving_batch=[1] * 2,
            corruption_seed=[0, 1],
            receiving_coverage=[0.5, 1.0],
        )
    )
    summary = summarize_coverage(raw)
    assert summary.receiving_coverage_mean.item() == 0.75
    assert summary.receiving_coverage_sd.item() == pytest.approx(np.sqrt(0.125))
    assert summary.n_seeds.item() == 2


def subsection(namespace, notebook, cell_id):
    cells = json.loads(notebook.read_text())["cells"]
    code = "".join(next(c for c in cells if c.get("id") == cell_id)["source"])
    exec(compile(code, cell_id, "exec"), namespace)


def check_subsection(e, namespace, filename, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    notebook = Path(__file__).parents[1] / "notebooks" / filename
    expected = {mode: e.make_batches(mode, 10.0, 0) for mode in e.config.data.modes}
    restored = type(e)(e.config, e.paths)
    restore_saved_construction(restored)
    for mode, batches in expected.items():
        for original, saved in zip(batches, restored.make_batches(mode, 10.0, 0), strict=True):
            np.testing.assert_allclose(original, saved, atol=1e-14, rtol=1e-14)
    for cell in json.loads(notebook.read_text())["cells"]:
        code = "".join(cell["source"])
        if "def prepare_cached_plan_inputs(" in code or "def sweep_only(" in code:
            exec(compile(code, "cached-plan-inputs", "exec"), namespace)
    namespace.update(
        config=e.config, experiment=e, plot_config=SimpleNamespace(rel_reg=1.0), plt=plt
    )
    for cell in ("receiving-hubness-setup", "receiving-hubness-compute", "receiving-hubness-plots"):
        subsection(namespace, notebook, cell)
    raw, points = namespace["hubness_stats"], namespace["hubness_points"]
    assert set(raw["mode"]) == set(e.config.data.modes)
    assert raw.k.eq(15).all()
    assert set(raw.magnitude) == set(e.config.corruption.magnitudes)
    assert raw.query("graph in ['shared_knn','sparse_ot']").receiving_coverage.eq(1).all()
    assert len(raw) == 8 * len(e.config.data.modes) * len(
        e.config.corruption.corruption_seeds
    ) * len(set(raw.magnitude)) * len(e.config.ot.epsilon_policies)
    assert not points.empty if 10.0 in e.config.corruption.magnitudes else points.empty

    def forbidden(*args, **kwargs):
        raise AssertionError("Loading cached receiving diagnostics attempted OT")

    from ne_aligned_utils import methods

    monkeypatch.setattr(methods, "compute_ot_plan", forbidden)
    namespace.update(RUN_EXPERIMENT=False, experiment=restored)
    for cell in ("receiving-hubness-setup", "receiving-hubness-compute"):
        subsection(namespace, notebook, cell)
    pd.testing.assert_frame_equal(namespace["hubness_stats"], raw, check_dtype=False)
    plt.close("all")


def test_parallel_subsection_and_saved_restore(parallel_setup, monkeypatch):  # noqa: F811
    e = construct(parallel_setup)
    check_subsection(e, parallel_setup, "mnist_parallel_batch_effect.ipynb", monkeypatch)


def test_orthogonal_subsection_and_saved_restore(orthogonal_setup, monkeypatch):  # noqa: F811
    e = orthogonal_setup
    for stage in ("batches", "subspaces", "corruptions"):
        run_stage(e, stage)
    e.initialize_run()
    check_subsection(e, e._notebook_namespace, "mnist_orthogonal_batch_effect.ipynb", monkeypatch)
