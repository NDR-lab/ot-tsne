"""The task-timing scanner preserves saved fields and never substitutes wall-clock seconds."""


import numpy as np
import pytest

from ne_aligned_utils.paper.summaries import task_timing as timing


def test_saved_timing_and_resource_fields_survive_scanning(tmp_path):
    stages = (
        "time_total",
        "time_pre_embedding",
        "time_embedding",
        "time_ot_cost",
        "time_ot_solve",
        "time_ot_postprocess",
        "time_ot_solve_self",
        "time_embedding_graph",
        "time_embedding_initialization",
        "time_embedding_optimization",
        "time_embedding_reference",
        "time_embedding_query",
        "time_wall_outer",
        "time_excluded_overhead",
        "time_warmup",
    )
    resources = (
        "timing_schema",
        "threads_requested",
        "n_jobs",
        "blas_name",
        "device",
        "slurm_cpus_per_task",
        "env_fingerprint",
        "n_obs",
        "n_batches",
        "n_features_input",
    )
    assert timing.RUNTIME_STAGE_COLUMNS == stages
    assert timing.RUNTIME_RESOURCE_COLUMNS == resources
    values = {name: float(index + 1) for index, name in enumerate(stages)}
    values.update({name: f"stored-{name}" for name in resources})
    np.savez(tmp_path / "task.npz", **values, task_kind="ot", batch_sizes=[2, 3])
    np.savez(tmp_path / "ignored_benchmark_embeddings.npz", embedding=np.zeros((5, 2)))
    actual = timing.scan_task_tree(tmp_path)
    assert len(actual) == 1
    for name, expected in values.items():
        assert actual.iloc[0][name] == expected
    assert actual.iloc[0]["method_key"] == "ot"
    assert actual.iloc[0]["n_cells"] == 5
    assert not actual.iloc[0]["representation_reused"]


def test_missing_optional_fields_remain_nan(tmp_path):
    np.savez(tmp_path / "task.npz", time_total=2.0, task_kind="joint")
    actual = timing.scan_task_tree(tmp_path)
    assert np.isnan(actual.iloc[0]["time_ot_solve"])
    assert np.isnan(actual.iloc[0]["n_cells"])


def test_empty_tree_is_not_silently_accepted(tmp_path):
    with pytest.raises(FileNotFoundError, match="no task"):
        timing.scan_task_tree(tmp_path)
