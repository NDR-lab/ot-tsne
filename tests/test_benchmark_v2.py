"""Contract tests for the canonical manifest-driven real-data benchmark."""

from __future__ import annotations

import copy
import importlib.util
import sys
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from ne_aligned_utils.benchmark import pipeline
from ne_aligned_utils.benchmark.scoring import cache_backend
from ne_aligned_utils.config import ROOT


@pytest.fixture(autouse=True)
def isolated_benchmark_roots(tmp_path, monkeypatch):
    """Task-planning tests must not load a caller's optional prepared data."""
    monkeypatch.setenv("NE_ALIGNED_DATA_ROOT", str(tmp_path / "prepared"))
    monkeypatch.setenv("NE_ALIGNED_RESULT_ROOT", str(tmp_path / "results"))


@pytest.fixture(scope="module")
def runner():
    path = ROOT / "scripts/run_benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.main is pipeline.main
    return pipeline


def test_canonical_script_reexports_the_package_implementation(runner):
    assert runner.main is pipeline.main


def test_parallel_cluster_stages_avoid_shared_writes(monkeypatch, runner):
    calls = []
    tasks = [object()]
    monkeypatch.setattr(runner, "load_manifest", lambda _path: ({}, "hash"))
    monkeypatch.setattr(
        runner,
        "expand_tasks",
        lambda _manifest, _hash, **kwargs: calls.append(("expand", kwargs)) or tasks,
    )
    monkeypatch.setattr(runner, "run_tasks", lambda *_args: calls.append(("run", None)))
    monkeypatch.setattr(runner, "aggregate_tasks", lambda *_args: calls.append(("aggregate", None)))
    monkeypatch.setattr(
        runner, "score_tasks", lambda *_args, **_kwargs: calls.append(("score", None))
    )
    monkeypatch.setattr(runner, "validate_tasks", lambda *_args: calls.append(("validate", None)))
    monkeypatch.setattr(
        runner, "write_task_metadata", lambda *_args, **_kwargs: calls.append(("metadata", None))
    )

    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "run-aggregate"])
    runner.main()
    assert [name for name, _ in calls] == ["expand", "run", "aggregate"]
    assert calls[0][1]["persist_sample"] is True

    calls.clear()
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "finalize"])
    runner.main()
    assert [name for name, _ in calls] == ["expand", "score", "validate", "metadata"]
    assert calls[0][1]["persist_sample"] is False


def test_ncov_checkpoint_matrix_is_fully_expanded_and_seed_safe(runner):
    manifest, manifest_hash = runner.load_manifest(runner.DEFAULT_MANIFEST)
    tasks = runner.expand_tasks(
        manifest,
        manifest_hash,
        dataset_filter="nCoV",
        allow_pending=True,
    )

    # Seven 50D methods, eleven direct-2D methods, and five OT variants, each at three seeds.
    assert len(tasks) == 69
    assert Counter(task.experiment for task in tasks) == {
        "main_50d_tsne_cpu": 21,
        "direct_2d_cpu": 33,
        "ot_diagonal_cpu": 15,
    }
    assert not {"cca-50", "cca-2d", "cca-2d-uncentered"}.intersection(task.method for task in tasks)
    assert {task.method_seed for task in tasks} == {0, 1, 2}
    assert all(task.visualization_seed == task.method_seed for task in tasks)
    assert {manifest["seeds"]["sample"]} == {0}
    assert {manifest["seeds"]["preprocessing"]} == {0}
    assert {task.pipeline for task in tasks} == {
        "integrated_50d_then_common_tsne",
        "method_produces_final_2d",
        "ot_graph_then_common_tsne",
    }

    by_method = {task.method: task for task in tasks if task.method_seed == 0}
    assert by_method["multimap"].method_settings["native_profile"] is True
    # The paper's suggested starting point for revised ct-SNE.
    assert by_method["rc-tsne"].method_settings["beta"] == pytest.approx(1e-4)
    assert by_method["ref"].method_settings["initialization_seed"] == "task_seed"

    assert "--multimap-native" in runner.task_command(manifest, by_method["multimap"])
    rc_command = runner.task_command(manifest, by_method["rc-tsne"])
    assert rc_command[rc_command.index("--rc-beta") + 1] == "0.0001"

    diagonal = [task for task in tasks if task.experiment == "ot_diagonal_cpu"]
    assert {task.graph_symmetrization for task in diagonal} == {"or"}
    knndiag = [task for task in diagonal if task.method == "ot-knndiag"]
    assert {task.knndiag_symmetrization for task in knndiag} == {"or"}


def test_run_identity_separates_scientific_arms(runner):
    manifest, manifest_hash = runner.load_manifest(runner.DEFAULT_MANIFEST)
    tasks = runner.expand_tasks(
        manifest,
        manifest_hash,
        dataset_filter="nCoV",
        experiment_filter="ot_diagonal_cpu",
        allow_pending=True,
    )
    base = next(
        task
        for task in tasks
        if task.method == "ot" and task.method_seed == 0 and task.rel_reg == 1.0
    )
    # Alternative scientific settings must hash differently even when they are not in
    # the submitted configuration's active grid.
    counterpart = replace(base, graph_symmetrization="and")
    assert base.run_id != counterpart.run_id
    assert base.arm_id != counterpart.arm_id
    assert set(base.identity) == {
        "dataset",
        "preprocessing_manifest_hash",
        "representation_family",
        "fixed_sample_hash",
        "method",
        "pipeline",
        "method_seed",
        "visualization_seed",
        "rel_reg",
        "effective_k",
        "diagonal_construction",
        "graph_symmetrization",
        "knndiag_symmetrization",
        "transport",
        "backend",
        "device",
        "method_settings",
    }


def test_manifest_has_only_real_data_and_uot_scope_is_restricted(runner):
    manifest, _ = runner.load_manifest(runner.DEFAULT_MANIFEST)
    assert not {"human_brain_log", "mnist", "ATAC"}.intersection(manifest["datasets"])
    uot = next(item for item in manifest["experiments"] if item["kind"] == "uot")
    assert set(uot["datasets"]) == {"hematopoietic"}
    assert uot["enabled"] is False
    assert uot["rel_reg_m"] == [0.1, 1.0, 10.0]

    expected = {
        "nCoV",
        "human_brain",
        "lupus",
        "PBMC",
        "mouse_atlas",
        "hematopoietic",
        "yao_10x_ent_l2l3",
    }
    assert set(manifest["datasets"]) == expected
    assert all(spec["enabled"] for spec in manifest["datasets"].values())
    for experiment in manifest["experiments"][:3]:
        assert set(experiment["datasets"]) == expected


def test_dataset_config_adapters_and_registry_agree(runner):
    from scripts.prepare.build_dataset import validate_dataset_declarations

    manifest, _ = runner.load_manifest(runner.DEFAULT_MANIFEST)
    validate_dataset_declarations(manifest)


def test_cosmetic_manifest_changes_do_not_change_task_or_arm_identity(
    runner, tmp_path, monkeypatch
):
    # Keep both manifests on the same data state. Otherwise this test becomes dependent on
    # whether the optional release bundle has been extracted under the repository's data root.
    monkeypatch.setenv("NE_ALIGNED_DATA_ROOT", str(tmp_path / "prepared"))
    manifest, manifest_hash = runner.load_manifest(runner.DEFAULT_MANIFEST)
    historical_shape = copy.deepcopy(manifest)
    historical_shape["name"] = "historical-directory-name"
    historical_shape["data_root"] = "/an/external/data/location"
    historical_shape["result_root"] = "/an/external/result/location"
    historical_shape["scheduler_metadata"] = {"queue": "removed-from-release"}
    historical_hash = runner.canonical_hash(historical_shape)
    assert historical_hash != manifest_hash

    current = runner.expand_tasks(
        manifest, manifest_hash, dataset_filter="nCoV", profile_name="smoke", allow_pending=True
    )
    historical = runner.expand_tasks(
        historical_shape,
        historical_hash,
        dataset_filter="nCoV",
        profile_name="smoke",
        allow_pending=True,
    )
    assert [task.run_id for task in current] == [task.run_id for task in historical]
    assert [task.arm_id for task in current] == [task.arm_id for task in historical]


def test_all_real_datasets_are_explicitly_uncapped(runner):
    manifest, _ = runner.load_manifest(runner.DEFAULT_MANIFEST)
    assert all(spec["sample_policy"] == "all_cells" for spec in manifest["datasets"].values())
    assert all("max_cells_batch0" not in spec for spec in manifest["datasets"].values())
    assert all("max_cells_batch1" not in spec for spec in manifest["datasets"].values())


def test_smoke_profile_is_one_seed_and_canonical_only(runner):
    manifest, manifest_hash = runner.load_manifest(runner.DEFAULT_MANIFEST)
    tasks = runner.expand_tasks(
        manifest,
        manifest_hash,
        dataset_filter="nCoV",
        profile_name="smoke",
        allow_pending=True,
    )
    assert len(tasks) == 23
    assert {task.method_seed for task in tasks} == {0}
    assert {task.rel_reg for task in tasks if task.rel_reg is not None} == {1.0}
    diagonal = [task for task in tasks if task.experiment == "ot_diagonal_cpu"]
    assert {task.graph_symmetrization for task in diagonal} == {"or"}


def test_method_filter_selects_one_yao_method_without_changing_its_grid(runner):
    manifest, manifest_hash = runner.load_manifest(runner.DEFAULT_MANIFEST)
    tasks = runner.expand_tasks(
        manifest,
        manifest_hash,
        dataset_filter="yao_10x_ent_l2l3",
        experiment_filter="ot_diagonal_cpu",
        method_filter="ot-selfotmedian",
        allow_pending=True,
    )

    assert len(tasks) == 3
    assert {task.method for task in tasks} == {"ot-selfotmedian"}
    assert {task.graph_symmetrization for task in tasks} == {"or"}
    assert {task.rel_reg for task in tasks} == {1.0}
    assert {task.method_seed for task in tasks} == {0, 1, 2}


def test_uot_is_fixed_entropy_and_does_not_expand_diagonal_ablations(runner):
    manifest, manifest_hash = runner.load_manifest(runner.DEFAULT_MANIFEST)
    tasks = runner.expand_tasks(
        manifest,
        manifest_hash,
        experiment_filter="uot_hematopoietic_cpu",
        include_disabled=True,
        allow_pending=True,
    )
    assert len(tasks) == 18
    assert {task.dataset for task in tasks} == {"hematopoietic"}
    assert {task.method for task in tasks} == {"ot-uot-topk", "ot-uot-mass"}
    assert {task.rel_reg for task in tasks} == {1.0}
    assert {task.rel_reg_m for task in tasks} == {0.1, 1.0, 10.0}
    assert {task.graph_symmetrization for task in tasks} == {"or"}


def test_yao_device_comparison_is_canonical_and_device_separated(runner):
    manifest, manifest_hash = runner.load_manifest(runner.DEFAULT_MANIFEST)
    tasks = runner.expand_tasks(
        manifest,
        manifest_hash,
        experiment_filter="yao_ot_device_agreement",
        include_disabled=True,
        allow_pending=True,
    )
    assert len(tasks) == 6
    assert {task.device for task in tasks} == {"cpu", "cuda"}
    assert {task.rel_reg for task in tasks} == {1.0}
    assert {task.method_seed for task in tasks} == {0, 1, 2}
    assert all(task.method_settings["ot_dtype"] == "float64" for task in tasks)
    assert len({task.run_id for task in tasks}) == 6


def test_declared_backend_covers_storage_name_overrides():
    cache = {
        "backend": np.asarray("tsne"),
        "liger2_50_tsne_embeddings": np.empty((1, 1, 2)),
    }
    assert cache_backend(cache) == "tsne"


@pytest.mark.parametrize("ot_init", ("eot", "random"))
def test_every_method_runs_under_any_ot_init(runner, monkeypatch, ot_init):
    """The init setting reaches OT-tSNE only. Every other task parses, and keeps the run id it
    has under random init, so its results stay valid when the OT-tSNE init changes."""
    from ne_aligned_utils.benchmark.sweep import core
    from ne_aligned_utils.benchmark.sweep.registry import EPS_K_METHODS

    manifest, manifest_hash = runner.load_manifest(runner.DEFAULT_MANIFEST)
    chosen = copy.deepcopy(manifest)
    chosen["settings"]["ot_init"] = ot_init
    random = copy.deepcopy(manifest)
    random["settings"]["ot_init"] = "random"

    def plan(m):
        return runner.expand_tasks(m, manifest_hash, dataset_filter="nCoV", allow_pending=True)

    reference = {(t.method, t.method_seed): t.run_id for t in plan(random)}
    for task in plan(chosen):
        command = runner.task_command(chosen, task)
        monkeypatch.setattr(sys, "argv", ["sweep", *command[3:]])
        args = core.parse_args()
        if task.method in EPS_K_METHODS:
            assert args.init == ot_init
        else:
            assert "--init" not in command and args.init == "random"
            assert task.run_id == reference[(task.method, task.method_seed)]
