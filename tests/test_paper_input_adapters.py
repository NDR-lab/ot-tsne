"""Current input contracts and import order without requiring installed results."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ne_aligned_utils.paper.inputs import current

ROOT = Path(__file__).resolve().parents[1]
KEYS = json.loads((ROOT / "tests" / "fixtures" / "figures" / "FIGURE_BASELINE.json").read_text())[
    "method_order"
]


@pytest.mark.parametrize(
    "first",
    [
        "ne_aligned_utils.paper.inputs.current",
        "ne_aligned_utils.paper.figures.runtime",
        "ne_aligned_utils.paper.summaries.ranks",
        "ne_aligned_utils.paper.registry",
        "ne_aligned_utils.paper.rendering.presentation",
        "ne_aligned_utils.paper.build",
    ],
)
def test_import_order_in_fresh_process(first):
    code = f"""
import importlib
importlib.import_module({first!r})
from ne_aligned_utils.paper.inputs import current
from ne_aligned_utils.paper.inputs.current import load_canonical_frame, CANONICAL_DATASETS
assert load_canonical_frame is current.load_canonical_frame
assert CANONICAL_DATASETS is current.CANONICAL_DATASETS
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def _current_table():
    return pd.DataFrame(
        [
            dict(
                dataset="nCoV",
                method="multimap" if key == "multimap" else key,
                seed=seed,
                arm_id=key,
                backend="tsne",
                experiment="embeddings_cpu",
                device="cpu",
                transport="balanced",
                rel_reg=1.0,
                effective_k=15.0,
                graph_symmetrization="or",
                init=np.nan,
                source_cache=f"remote/nCoV/{key}/aggregate.npz",
                preprocessing_hash="prep",
                sample_hash="sample",
                extra_provenance="keep",
                **{metric: 0.1 + index * 0.02 + seed * 0.001 for metric in current.METRIC_RENAMES},
            )
            for index, key in enumerate(KEYS)
            for seed in range(3)
        ]
    )


def _write_current(root, frame):
    frame.to_csv(root / "metrics.csv", index=False)
    pd.DataFrame(
        [
            dict(
                dataset="nCoV",
                method="multimap",
                arm_id="multimap",
                method_settings='{"native_profile": true}',
            )
        ]
    ).to_csv(root / "task_metadata.csv", index=False)


def test_current_retains_provenance_and_current_identities(tmp_path):
    frame = _current_table()
    frame.loc[0, "backend"] = "umap"  # summary backend filtering remains a separate stage
    _write_current(tmp_path, frame)
    actual = current.load_canonical_frame(tmp_path, datasets=("nCoV",), require_source_caches=False)
    assert len(actual) == 69
    assert actual.loc[0, "backend"] == "umap"
    assert (
        actual["method_key"].tolist()
        == frame["method"].replace({"multimap": "multimap"}).tolist()
    )
    assert {"ref", "seurat-50", "seurat-2d"} <= set(actual["method_key"])
    assert set(actual["result_source"]) == {"frozen_benchmark"}
    assert set(actual["supplementary_manifest"]) == {""}
    for column in ("source_cache", "preprocessing_hash", "sample_hash", "extra_provenance"):
        pd.testing.assert_series_equal(actual[column], frame[column], check_exact=True)
    for raw, normalized in current.METRIC_RENAMES.items():
        np.testing.assert_allclose(actual[normalized], frame[raw], rtol=1e-14)
    assert set(actual["init"]) == {"random"}
    with pytest.raises(FileNotFoundError, match="missing local aggregate"):
        current.load_canonical_frame(tmp_path, datasets=("nCoV",))


def test_current_row_count_and_metric_bounds_still_raise(tmp_path):
    frame = _current_table()
    _write_current(tmp_path, frame.iloc[:-1])
    with pytest.raises(ValueError, match="expected 69"):
        current.load_canonical_frame(tmp_path, datasets=("nCoV",), require_source_caches=False)
    frame.loc[0, "ilisi_score"] = 1.1
    _write_current(tmp_path, frame)
    with pytest.raises(ValueError, match="outside"):
        current.load_canonical_frame(tmp_path, datasets=("nCoV",), require_source_caches=False)
