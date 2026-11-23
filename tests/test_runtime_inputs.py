"""Canonical timing selection from scanned task files; no method is run."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ne_aligned_utils.paper.inputs import runtime as inputs
from ne_aligned_utils.paper.rendering.runtime import RUNTIME_FAMILIES, runtime_render_context
from ne_aligned_utils.paper.summaries import runtime as summaries


def _tasks():
    return pd.DataFrame(
        [
            dict(
                dataset="nCoV",
                task_kind=key,
                seed=seed,
                timing_schema="two_stage_v1",
                task_path=f"{key}/{seed}.npz",
                rel_reg=1.0,
                device="cpu",
                time_total=float(index + seed + 1),
                time_pre_embedding=float(index + seed + 1) * 0.6,
                time_embedding=float(index + seed + 1) * 0.4,
                n_obs=100,
            )
            for index, key in enumerate(k for members in RUNTIME_FAMILIES.values() for k in members)
            for seed in range(3)
        ]
    )


def _patch(monkeypatch, scan, *, native):
    monkeypatch.setattr(inputs, "scan_task_tree", scan)
    monkeypatch.setattr(
        inputs,
        "read_scalar",
        lambda path, name: "or" if name == "graph_symmetrization" else native,
    )


@pytest.mark.parametrize("native", (False, True))
def test_cpu_selection_keeps_canonical_rows_and_names_the_multimap_arm(
    tmp_path, monkeypatch, native
):
    (tmp_path / "nCoV").mkdir()
    tasks = _tasks()
    rejected = tasks.iloc[:3].copy()
    rejected["rel_reg"] = 10.0
    scanned = pd.concat([tasks, rejected], ignore_index=True)
    before = scanned.copy(deep=True)
    _patch(monkeypatch, lambda path: scanned, native=native)
    result = inputs.load_runtime_frame(tmp_path, datasets=("nCoV",))
    assert_frame_equal(scanned, before)
    assert len(result) == 69
    assert result["time_total"].tolist() == tasks["time_total"].tolist()
    assert result["result_source"].eq("frozen_benchmark").all()
    multimap = result[result["task_kind"] == "multimap"]
    assert multimap["method_key"].eq("multimap" if native else "multimap-k15").all()
    annotated = summaries.annotate(result, context=runtime_render_context())
    assert annotated.loc[multimap.index, "included_in_plot"].eq(native).all()


def test_missing_tree_and_incomplete_pool_are_refused(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="missing benchmark dataset"):
        inputs.load_runtime_frame(tmp_path, datasets=("nCoV",))
    (tmp_path / "nCoV").mkdir()
    _patch(monkeypatch, lambda path: _tasks().iloc[:-1], native=True)
    with pytest.raises(ValueError, match="expected.*69.*found.*68"):
        inputs.load_runtime_frame(tmp_path, datasets=("nCoV",))


def test_supplementary_replacement_requires_all_three_seeds(tmp_path, monkeypatch):
    root = tmp_path / "frozen"
    (root / "nCoV").mkdir(parents=True)
    supplementary = tmp_path / "supplementary"
    replacements = {}
    for method, (arm, directory) in inputs.SUPPLEMENTARY_REPLACEMENTS.items():
        path = supplementary / arm / directory / "nCoV"
        path.mkdir(parents=True)
        tasks = _tasks()
        replacements[path] = tasks.loc[tasks.task_kind == method].copy()
        replacements[path]["time_total"] = 999.0
    _patch(
        monkeypatch,
        lambda path: _tasks() if path == root / "nCoV" else replacements[path],
        native=False,
    )
    kwargs = dict(datasets=("nCoV",), supplementary_root=supplementary)
    result = inputs.load_runtime_frame(root, **kwargs)
    corrected = result[result.result_source.str.startswith("supplementary:")]
    assert len(result) == 69
    assert len(corrected) == 9
    assert corrected.time_total.eq(999.0).all()
    # The native supplementary arm is the reported MultiMAP even without a task flag.
    assert corrected.loc[corrected.task_kind == "multimap", "method_key"].eq("multimap").all()
    first = next(iter(replacements))
    replacements[first] = replacements[first].iloc[:-1]
    with pytest.raises(ValueError, match="seeds 0, 1, 2"):
        inputs.load_runtime_frame(root, **kwargs)


def test_gpu_pool_is_separate_and_uses_phase_medians(tmp_path, monkeypatch):
    assert inputs.load_gpu_medians(tmp_path / "root", tmp_path).empty
    directory = tmp_path / inputs.GPU_SUBPATH / "nCoV"
    directory.mkdir(parents=True)
    tasks = _tasks().iloc[:3].copy()
    tasks["device"] = "cuda"
    cpu = tasks.iloc[:1].copy()
    cpu["device"] = "cpu"
    cpu["time_total"] = 10000.0
    _patch(monkeypatch, lambda path: pd.concat([tasks, cpu], ignore_index=True), native=False)
    result = inputs.load_gpu_medians(tmp_path / "root", tmp_path, datasets=("nCoV",))
    assert len(result) == 1
    assert result.iloc[0].n_runs == 3
    row = result.iloc[0]
    assert row.method_key == "ot"
    assert row.n_runs == 3
    assert row.time_total == tasks.time_total.median()
    assert row.time_pre_embedding == tasks.time_pre_embedding.median()


def test_gpu_tasks_in_the_results_root_are_kept_out_of_cpu_timings(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "nCoV").mkdir(parents=True)
    old = tmp_path / "supplementary"
    (old / inputs.GPU_SUBPATH / "nCoV").mkdir(parents=True)
    cpu = _tasks()
    gpu = cpu[cpu.task_kind == "ot"].copy()
    gpu["device"] = "cuda"
    gpu["time_total"] = 0.5
    stale = gpu.copy()
    stale["time_total"] = 99.0
    _patch(
        monkeypatch,
        lambda path: pd.concat([cpu, gpu], ignore_index=True) if path == root / "nCoV" else stale,
        native=True,
    )
    timing = inputs.load_runtime_frame(root, datasets=("nCoV",))
    assert len(timing) == 69 and timing["device"].eq("cpu").all()
    # The new run's GPU tasks win over the October supplementary run.
    medians = inputs.load_gpu_medians(root, old, datasets=("nCoV",))
    assert medians.time_total.tolist() == [0.5]
