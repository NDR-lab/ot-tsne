"""Check supplementary result replacements and retained benchmark identities."""

from __future__ import annotations

import json

import pandas as pd

from ne_aligned_utils.paper.inputs.current import (
    SUPPLEMENTARY_REPLACEMENTS,
    _replace_with_supplementary,
)


def test_supplementary_rows_replace_frozen_slots_without_renaming_methods(tmp_path):
    dataset = "example"
    preprocessing_hash = "prep"
    sample_hash = "sample"
    benchmark_hash = "abcdef123456" + "0" * 52
    benchmark_root = tmp_path / "real-data-rerun-v2--abcdef123456"
    benchmark_root.mkdir()
    supplementary_root = tmp_path / "supplementary"

    rows = []
    for method in SUPPLEMENTARY_REPLACEMENTS:
        for seed in range(3):
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "source_cache": f"frozen/{method}.npz",
                    "preprocessing_hash": preprocessing_hash,
                    "sample_hash": sample_hash,
                    "experiment": "direct_2d_cpu",
                    "device": "cpu",
                    "transport": "balanced",
                    "ilisi_score": -1.0,
                }
            )
    frozen = pd.DataFrame(rows)

    for offset, (method, (arm, directory)) in enumerate(SUPPLEMENTARY_REPLACEMENTS.items()):
        arm_root = supplementary_root / arm
        arm_root.mkdir(parents=True)
        (arm_root / "supplementary_manifest.json").write_text(
            json.dumps(
                {
                    "benchmark_manifest_hash": benchmark_hash,
                    "preprocessing_hashes": {dataset: preprocessing_hash},
                    "sample_hashes": {dataset: sample_hash},
                    "task_seeds": [0, 1, 2],
                }
            )
        )
        method_root = arm_root / directory / dataset
        method_root.mkdir(parents=True)
        (method_root / "aggregate.npz").touch()
        pd.DataFrame(
            {
                "dataset": [dataset] * 3,
                "method": [method] * 3,
                "seed": [0, 1, 2],
                "source_cache": ["remote/aggregate.npz"] * 3,
                "ilisi_score": [offset + 0.1, offset + 0.2, offset + 0.3],
            }
        ).to_csv(method_root / "metrics.csv", index=False)

    replaced = _replace_with_supplementary(
        frozen,
        benchmark_root=benchmark_root,
        dataset=dataset,
        supplementary_root=supplementary_root,
    )

    assert set(replaced["method"]) == {"multimap", "rc-tsne", "ref"}
    assert set(replaced["experiment"]) == {"direct_2d_cpu"}
    assert set(replaced["device"]) == {"cpu"}
    assert set(replaced["transport"]) == {"balanced"}
    assert not (replaced["ilisi_score"] == -1.0).any()
    assert replaced["result_source"].str.startswith("supplementary:").all()
    assert replaced["source_cache"].map(lambda value: value.endswith("aggregate.npz")).all()
