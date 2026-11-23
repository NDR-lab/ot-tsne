"""Fast end-to-end contract for the public benchmark wiring."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ne_aligned_utils.benchmark import pipeline
from ne_aligned_utils.data.preprocessing import prepare, write_preprocessing_manifest


def test_prepare_run_aggregate_score_validate_contract(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    counts = rng.poisson(3.0, size=(48, 24)).astype(float) + 1.0
    obs = pd.DataFrame(
        {
            "method": pd.Categorical(["data1"] * 24 + ["data2"] * 24),
            "cell_type": pd.Categorical((["A", "B", "C"] * 8) * 2),
        },
        index=[f"cell-{index}" for index in range(48)],
    )
    adata = ad.AnnData(counts.copy(), obs=obs)
    adata.layers["counts"] = counts.copy()
    data_root = tmp_path / "data"
    dataset_dir = data_root / "nCoV"
    out = prepare(
        adata,
        dataset_dir,
        "nCoV",
        n_top_genes=20,
        n_pca=10,
        batch_names=("batch 0", "batch 1"),
    )
    source = tmp_path / "tiny-source.txt"
    source.write_text("deterministic tiny fixture\n")
    write_preprocessing_manifest(
        out,
        source_paths=[source],
        loader="tiny_fixture",
        preprocessing_seed=0,
        n_top_genes=20,
        n_pca=10,
    )

    base, _ = pipeline.load_manifest(pipeline.DEFAULT_MANIFEST)
    manifest = copy.deepcopy(base)
    manifest["name"] = "smoke-contract"
    manifest["data_root"] = str(data_root)
    manifest["result_root"] = str(tmp_path / "results")
    monkeypatch.setenv("NE_ALIGNED_DATA_ROOT", manifest["data_root"])
    monkeypatch.setenv("NE_ALIGNED_RESULT_ROOT", manifest["result_root"])
    manifest["datasets"] = {"nCoV": manifest["datasets"]["nCoV"]}
    manifest["seeds"]["downstream"] = [0]
    manifest["experiments"] = [
        {
            "name": "direct_2d_cpu",
            "enabled": True,
            "datasets": ["nCoV"],
            "kind": "methods",
            "pipeline": "method_produces_final_2d",
            "device": "cpu",
            "methods": ["joint-pca"],
        }
    ]
    manifest_hash = pipeline.canonical_hash(manifest)
    tasks = pipeline.expand_tasks(manifest, manifest_hash, persist_sample=True)
    assert len(tasks) == 1

    pipeline.run_tasks(manifest, tasks)
    first = np.load(pipeline.task_output(manifest, tasks[0]), allow_pickle=False)["embedding"]
    pipeline.run_tasks(manifest, tasks)
    second = np.load(pipeline.task_output(manifest, tasks[0]), allow_pickle=False)["embedding"]
    np.testing.assert_allclose(first, second, rtol=0, atol=0)
    assert first.shape == (48, 2)
    assert np.isfinite(first).all()

    pipeline.write_task_metadata(manifest, tasks)
    pipeline.aggregate_tasks(manifest, tasks)
    pipeline.score_tasks(manifest, tasks)
    pipeline.validate_tasks(manifest, tasks)

    root = pipeline.result_base(manifest, manifest_hash)
    readable = pd.read_csv(root / "task_metadata.csv")
    metrics = pd.read_csv(root / "metrics.csv")
    dataset_metrics = pd.read_csv(root / "metrics" / "nCoV--all-experiments--all-devices.csv")
    assert readable["run_id"].tolist() == [tasks[0].run_id]
    assert not Path(readable["output_embedding_path"].iloc[0]).is_absolute()
    # A filtered selection writes its own table and leaves the full one alone.
    before = (root / "task_metadata.csv").read_bytes()
    filtered = pipeline.write_task_metadata(
        manifest, tasks, dataset_filter="nCoV", device_filter="cpu"
    )
    assert filtered == root / "task_metadata" / "nCoV--all-experiments--cpu.csv"
    assert (root / "task_metadata.csv").read_bytes() == before
    assert len(metrics) == 1
    # Each row records its run: every setting, and a readable configuration name.
    assert json.loads(metrics["method_settings"].iloc[0]) == tasks[0].method_settings
    assert metrics["configuration"].iloc[0].startswith(tasks[0].method)
    pd.testing.assert_frame_equal(metrics, dataset_metrics)
    assert not metrics.duplicated(["dataset", "method", "seed"]).any()
    assert np.isfinite(metrics["neighborhood_purity_score"]).all()
    assert np.isfinite(metrics["silhouette_score"]).all()

    # A method may add its own columns (OT-tSNE adds init_used). Scored after one that does not,
    # the combined table must still hold them.
    extra_columns = iter([{}, {"init_used": "eot"}])

    def fake_score(command, **kwargs):
        output = Path(command[command.index("--output-csv") + 1])
        row = {"dataset": "nCoV", "method": "joint-pca", "seed": "0", **next(extra_columns)}
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    monkeypatch.setattr(pipeline, "arm_groups", lambda tasks, manifest: [tasks, tasks])
    monkeypatch.setattr(pipeline.subprocess, "run", fake_score)
    pipeline.score_tasks(manifest, tasks)
    mixed = pd.read_csv(root / "metrics.csv")
    assert mixed["init_used"].isna().tolist() == [True, False]
