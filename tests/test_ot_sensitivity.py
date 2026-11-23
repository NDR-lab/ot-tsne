"""OT-tSNE sensitivity sweep: grid lines, per-point files, row labels, and the OT cache."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import pytest

from ne_aligned_utils.benchmark import pipeline
from ne_aligned_utils.config import ROOT

SENSITIVITY = ROOT / "configs" / "ot_sensitivity.toml"
REL_REGS = [0.1, 0.5, 1.0, 5.0, 10.0]
KS = [5, 10, 15, 30, 60]


@pytest.fixture(autouse=True)
def isolated_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("NE_ALIGNED_DATA_ROOT", str(tmp_path / "prepared"))
    monkeypatch.setenv("NE_ALIGNED_RESULT_ROOT", str(tmp_path / "results"))


def _tasks(**filters):
    manifest, manifest_hash = pipeline.load_manifest(SENSITIVITY)
    return manifest, pipeline.expand_tasks(manifest, manifest_hash, allow_pending=True, **filters)


def test_grid_is_every_rel_reg_with_every_k_for_ot_only():
    manifest, tasks = _tasks()
    assert manifest["grid"]["rel_reg"] == REL_REGS
    assert manifest["grid"]["effective_k"] == KS
    assert {task.method for task in tasks} == {"ot"}
    assert len({task.dataset for task in tasks}) == 7
    assert len(tasks) == 25 * 3 * 7
    points = {(task.rel_reg, task.effective_k) for task in tasks}
    assert points == {(r, k) for r in REL_REGS for k in KS}


def test_grid_lines_are_tasks_of_the_full_grid():
    manifest, grid = _tasks()
    _, rel_reg_line = _tasks(effective_k_filter=15)
    _, k_line = _tasks(rel_reg_filter=1.0)
    assert len(rel_reg_line) == len(k_line) == 5 * 3 * 7
    assert {t.rel_reg for t in rel_reg_line} == set(REL_REGS)
    assert {t.effective_k for t in k_line} == set(KS)

    by_id = {task.run_id: task for task in grid}
    cross = {task.run_id for task in rel_reg_line} | {task.run_id for task in k_line}
    # 5 + 5 - 1 points, the default point shared by both lines.
    assert len(cross) == 9 * 3 * 7
    assert cross <= set(by_id)
    for task in rel_reg_line + k_line:
        assert pipeline.task_output(manifest, task) == pipeline.task_output(
            manifest, by_id[task.run_id]
        )


def test_every_task_and_point_has_its_own_files():
    manifest, tasks = _tasks()
    outputs = [pipeline.task_output(manifest, task) for task in tasks]
    assert len(set(outputs)) == len(outputs)
    assert len({task.run_id for task in tasks}) == len(tasks)
    # One arm per dataset; one aggregate and one metrics file per point.
    assert len({(task.dataset, task.arm_id) for task in tasks}) == 7
    groups = pipeline.arm_groups(tasks, manifest)
    assert len(groups) == 25 * 7
    assert all(len(group) == 3 for group in groups)
    aggregates = {pipeline.aggregate_path(manifest, group[0]) for group in groups}
    assert len(aggregates) == len(groups)
    example = next(
        t for t in tasks if t.rel_reg == 0.5 and t.effective_k == 15 and t.method_seed == 0
    )
    assert pipeline.aggregate_path(manifest, example).name == "aggregate--rel_reg_0p5__k_15.npz"
    assert pipeline.arm_metrics_path(manifest, example).name == "metrics--rel_reg_0p5__k_15.csv"
    assert pipeline.task_output(manifest, example).name == "ot_tsne_seed_0_rel_reg_0p5_k_15.npz"


def test_unknown_grid_value_is_an_error():
    with pytest.raises(ValueError, match="rel_reg=0.3"):
        _tasks(rel_reg_filter=0.3)
    with pytest.raises(ValueError, match="effective_k=20"):
        _tasks(effective_k_filter=20)


def test_main_benchmark_names_and_commands_are_unchanged():
    manifest, manifest_hash = pipeline.load_manifest(pipeline.DEFAULT_MANIFEST)
    tasks = pipeline.expand_tasks(manifest, manifest_hash, allow_pending=True)
    for task in tasks:
        assert pipeline.point_suffix(manifest, task) == ""
        assert pipeline.aggregate_path(manifest, task).name == "aggregate.npz"
        assert "--ot-cache-dir" not in pipeline.task_command(manifest, task)
    assert len(pipeline.arm_groups(tasks, manifest)) == len(pipeline.arm_groups(tasks))
    assert pipeline.selection_name(None, None, None) is None
    assert pipeline.selection_name("nCoV", None, "cuda") == "nCoV--all-experiments--cuda"
    assert (
        pipeline.selection_name(None, None, None, effective_k_filter=15)
        == "all-datasets--all-experiments--all-devices--k_15"
    )
    assert (
        pipeline.selection_name(None, None, None, rel_reg_filter=1.0)
        == "all-datasets--all-experiments--all-devices--rel_reg_1p0"
    )


def test_sensitivity_tasks_share_one_cache_folder_per_dataset():
    manifest, tasks = _tasks(dataset_filter="nCoV")
    folders = set()
    for task in tasks:
        command = pipeline.task_command(manifest, task)
        folders.add(command[command.index("--ot-cache-dir") + 1])
    assert len(folders) == 1
    assert Path(folders.pop()).parts[-2:] == ("ot_cache", "nCoV")


def test_scored_rows_are_labelled_by_their_own_point(monkeypatch):
    """An arm holding several points must not label every row with its first task's point."""
    manifest, manifest_hash = pipeline.load_manifest(SENSITIVITY)
    manifest = copy.deepcopy(manifest)
    manifest["experiments"][0]["aggregate_per_point"] = False
    tasks = [
        task
        for task in pipeline.expand_tasks(
            manifest, manifest_hash, dataset_filter="nCoV", allow_pending=True
        )
        if (task.rel_reg, task.effective_k) in {(0.1, 5), (1.0, 15)}
    ]

    def fake_score(command, **_kwargs):
        output = Path(command[command.index("--output-csv") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["dataset", "method", "seed", "rel_reg", "effective_k"])
            for rel_reg, k in ((0.1, 5), (1.0, 15)):
                for seed in range(3):
                    writer.writerow(["nCoV", "ot", seed, rel_reg, k])

    monkeypatch.setattr(pipeline.subprocess, "run", fake_score)
    pipeline.score_tasks(manifest, tasks)
    base = pipeline.result_base(manifest, manifest_hash)
    with (base / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    for row in rows:
        k = int(row["effective_k"])
        rel_reg = float(row["rel_reg"])
        assert row["configuration"] == f"ot rel_reg={rel_reg:g} k={k} sym=or"
        assert row["rel_reg_role"] == ("canonical" if rel_reg == 1.0 else "sensitivity")


# The OT cache must give exactly what an independent run gives.


def _pair(seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(120, 10))
    x1 = rng.normal(size=(140, 10)) + 0.5
    return x0, x1


def _embed(x0, x1, *, k, seed, cache_dir=None):
    from ne_aligned_utils.methods.ot_methods import compute_ot_tsne_embedding_from_arrays

    return compute_ot_tsne_embedding_from_arrays(
        x0,
        x1,
        rel_reg=1.0,
        effective_k=k,
        random_state=seed,
        n_epochs=30,
        early_exaggeration_iter=10,
        return_artifacts=True,
        ot_cache_dir=cache_dir,
        threads=1,
    )


def test_cached_plan_and_svd_reproduce_independent_runs_exactly(tmp_path):
    x0, x1 = _pair()
    cache_dir = tmp_path / "ot_cache"
    hits = []
    for k in (5, 10):
        for seed in (0, 1):
            fresh, fresh_artifacts = _embed(x0, x1, k=k, seed=seed)
            cached, cached_artifacts = _embed(x0, x1, k=k, seed=seed, cache_dir=cache_dir)
            hits.append(cached_artifacts["ot_cache_hit"])
            assert "ot_cache_hit" not in fresh_artifacts
            assert np.array_equal(fresh, cached)
            assert np.array_equal(fresh_artifacts["ot_plan"], cached_artifacts["ot_plan"])
            assert (fresh_artifacts["graph"] != cached_artifacts["graph"]).nnz == 0
            assert np.array_equal(
                fresh_artifacts["singular_values"], cached_artifacts["singular_values"]
            )
            for name in ("epsilon", "median_squared_distance", "sinkhorn_niter"):
                assert fresh_artifacts[name] == cached_artifacts[name]
    # One solve, then three reads.
    assert hits == [False, True, True, True]
    assert len(list(cache_dir.glob("plan_svd_*.npz"))) == 1
    assert not list(cache_dir.glob("*.tmp"))


def test_cache_key_separates_inputs_and_settings():
    from ne_aligned_utils.methods.ot_cache import cache_key

    x0, x1 = _pair()
    base = dict(rel_reg=1.0, tol=1e-12, max_iter=10000, device="cpu", threads=4)
    reference = cache_key(x0, x1, **base)
    assert cache_key(x0, x1, **base) == reference
    for change in (
        {"rel_reg": 0.5},
        {"tol": 1e-9},
        {"max_iter": 100},
        {"threads": 1},
        {"device": "cuda"},
    ):
        assert cache_key(x0, x1, **{**base, **change}) != reference
    other = x0.copy()
    other[0, 0] += 1e-12
    assert cache_key(other, x1, **base) != reference


def test_sinkhorn_log_does_not_change_the_plan():
    import ot

    from ne_aligned_utils.methods.ot_transport import compute_ot_plan

    x0, x1 = _pair()
    plan, artifacts = compute_ot_plan(x0, x1, rel_reg=1.0, return_artifacts=True)
    a = np.full(len(x0), 1 / len(x0))
    b = np.full(len(x1), 1 / len(x1))
    reference = ot.sinkhorn(
        a, b, ot.dist(x0, x1), reg=artifacts["epsilon"], numItermax=10000, stopThr=1e-12
    )
    assert np.array_equal(plan, reference)
    assert artifacts["sinkhorn_converged"] is True
    assert artifacts["sinkhorn_final_error"] < 1e-12


def test_sinkhorn_reports_non_convergence():
    from ne_aligned_utils.methods.ot_transport import compute_ot_plan

    x0, x1 = _pair()
    with pytest.warns(UserWarning, match="did not converge"):
        _, artifacts = compute_ot_plan(
            x0, x1, rel_reg=0.01, max_iter=20, tol=1e-12, return_artifacts=True
        )
    assert artifacts["sinkhorn_converged"] is False
    assert artifacts["sinkhorn_niter"] == 19


def test_submission_runs_the_sensitivity_manifest_one_line_at_a_time(tmp_path):
    import os
    import subprocess
    import sys

    capture = tmp_path / "submissions.txt"
    (tmp_path / "sbatch").write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$SBATCH_CAPTURE"\n'
        'if [[ " $* " == *" --parsable "* ]]; then printf "123\\n"; fi\n'
    )
    (tmp_path / "uv").write_text('#!/usr/bin/env bash\nshift 2\nexec "$TEST_PYTHON" "$@"\n')
    for name in ("sbatch", "uv"):
        (tmp_path / name).chmod(0o755)
    completed = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "submit_benchmark.sh"),
            *("--manifest", "configs/ot_sensitivity.toml", "--effective-k", "15"),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "SBATCH_CAPTURE": str(capture),
            "NE_ALIGNED_LAUNCHER_FILE": str(tmp_path / "missing-launchers.sh"),
            "TEST_PYTHON": sys.executable,
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    submissions = capture.read_text().splitlines()
    # One job per dataset, then one finalizer, all on the sensitivity manifest's k=15 line.
    assert len(submissions) == 8
    for line in submissions:
        assert "--manifest configs/ot_sensitivity.toml" in line
        assert "--effective-k 15" in line
        assert "--job-name=ot_sensitivity-" in line
    assert sum("--experiment ot_sensitivity_cpu" in line for line in submissions) == 7
