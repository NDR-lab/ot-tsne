"""Characterize current selection, including deliberately absent stricter checks."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from ne_aligned_utils.paper.inputs import current as data
from ne_aligned_utils.paper.selection.benchmark import (
    CURRENT_BENCHMARK,
    canonical_mask,
    is_canonical_row,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, True),
        ({"backend": "umap"}, True),
        ({"backend": None}, True),
        ({"device": "cuda"}, False),
        ({"device": None}, False),
        ({"transport": "unbalanced"}, False),
        ({"transport": None}, False),
        ({"rel_reg": 0.1}, False),
        ({"rel_reg": "1"}, True),
        ({"rel_reg": np.nan}, True),
        ({"rel_reg": None}, True),
        ({"rel_reg": pd.NA}, True),
        ({"effective_k": "15"}, True),
        ({"effective_k": np.nan}, False),
        ({"effective_k": 30}, False),
        ({"graph_symmetrization": "and"}, False),
        ({"graph_symmetrization": None}, False),
        ({"method": "liger1-2d"}, False),
        ({"method": "seurat3-2d"}, False),
        ({"method": "ot-zerodiag-split"}, False),
        ({"method": "eot-50", "rel_reg": 0.1}, False),
        ({"method": "multimap", "effective_k": 30, "graph_symmetrization": None}, True),
        ({"method": "joint", "effective_k": None, "graph_symmetrization": None}, True),
        # Unknown identities are rejected later by annotation, not by this predicate.
        ({"method": "unknown"}, True),
    ],
)
def test_submitted_predicate(changes, expected):
    row = pd.Series(
        dict(
            method="ot",
            device="cpu",
            transport="balanced",
            rel_reg=1.0,
            effective_k=15,
            graph_symmetrization="or",
            backend="tsne",
        )
        | changes
    )
    assert bool(is_canonical_row(row)) is expected
    assert bool(is_canonical_row(row)) is expected


def test_policy_is_immutable_and_mask_preserves_input():
    with pytest.raises(FrozenInstanceError):
        CURRENT_BENCHMARK.rel_reg = 0.1
    frame = pd.DataFrame(
        [
            dict(method="joint", device="cpu", transport="balanced", rel_reg=np.nan),
            dict(method="joint", device="cuda", transport="balanced", rel_reg=np.nan),
        ],
        index=[7, 2],
    )
    before = frame.copy(deep=True)
    assert_series_equal(canonical_mask(frame), pd.Series([True, False], index=[7, 2]))
    assert_frame_equal(frame, before, check_exact=True)
    with pytest.raises(KeyError):
        canonical_mask(frame.drop(columns="device"))
    with pytest.raises(ValueError):
        is_canonical_row(frame.iloc[0].to_dict() | {"rel_reg": "bad"})


def _table(dataset="d1"):
    baseline = json.loads(
        (ROOT / "tests" / "fixtures" / "figures" / "FIGURE_BASELINE.json").read_text()
    )
    rows = []
    # Intentionally reverse registry order: the loader must not sort method/seed rows.
    for key in reversed(baseline["method_order"]):
        method = "multimap" if key == "multimap" else key
        for seed in (2, 0, 1):
            rows.append(
                dict(
                    dataset=dataset,
                    method=method,
                    seed=seed,
                    arm_id=method,
                    source_cache=f"{dataset}/missing/aggregate.npz",
                    init=np.nan,
                    backend="na",
                    experiment="fixture",
                    device="cpu",
                    transport="balanced",
                    rel_reg=1.0 if method.startswith(("ot", "eot")) else np.nan,
                    effective_k=15,
                    graph_symmetrization="or",
                    neighborhood_purity_score=0.2,
                    silhouette_score=0.3,
                    ilisi_score=0.4,
                    kbet_score=0.5,
                )
            )
    return pd.DataFrame(rows)


def _write(root, table):
    table.to_csv(root / "metrics.csv", index=False)
    datasets = table.dataset.unique()
    pd.DataFrame(
        [
            dict(
                dataset=d,
                method="multimap",
                arm_id="multimap",
                method_settings='{"native_profile": true}',
            )
            for d in datasets
        ]
    ).to_csv(root / "task_metadata.csv", index=False)


def test_loader_subset_order_dtypes_and_missing_data(tmp_path):
    table = pd.concat([_table("d1"), _table("d2")], ignore_index=True)
    _write(tmp_path, table)
    result = data.load_canonical_frame(tmp_path, datasets=("d2", "d1"), require_source_caches=False)
    assert result.dataset.tolist() == ["d2"] * 69 + ["d1"] * 69
    assert result.method.tolist() == table.method.tolist()
    assert result.seed.tolist() == table.seed.tolist()
    assert result.seed.dtype == np.dtype("int64")
    assert result.rel_reg.dtype == np.dtype("float64")
    assert result.rel_reg.isna().sum() == table.rel_reg.isna().sum()
    assert result.init.eq("random").all()
    assert result.loc[result.method == "multimap", "method_key"].eq("multimap").all()
    subset = data.load_canonical_frame(tmp_path, datasets=("d1",), require_source_caches=False)
    assert len(subset) == 69
    with pytest.raises(ValueError, match="no rows"):
        data.load_canonical_frame(tmp_path, datasets=("absent",), require_source_caches=False)
    with pytest.raises(FileNotFoundError):
        data.load_canonical_frame(
            tmp_path / "absent", datasets=("d1",), require_source_caches=False
        )
    with pytest.raises(FileNotFoundError):
        data.load_canonical_frame(tmp_path, datasets=("d1",))


@pytest.mark.parametrize(
    "source,settings,cache,expected",
    [
        ("frozen_benchmark", '{"native_profile": true}', "", "multimap"),
        ("frozen_benchmark", '{"native_profile": false, "n_neighbors": 15}', "", "multimap-k15"),
        # A row's own settings win over every fallback.
        (
            "frozen_benchmark",
            '{"native_profile": false, "n_neighbors": 15}',
            "multimap_native/a.npz",
            "multimap-k15",
        ),
        # Rows without settings (older supplementary swaps) fall back to where they came from.
        ("frozen_benchmark", None, "", "multimap-k15"),
        ("supplementary:multimap_native", None, "", "multimap"),
        ("frozen_benchmark", None, "multimap_native/a.npz", "multimap"),
    ],
)
def test_native_multimap_identity_remains_an_adapter_concern(source, settings, cache, expected):
    assert data._method_key("multimap", source, cache, method_settings=settings) == expected


def test_native_multimap_task_provenance(tmp_path):
    tasks = tmp_path / "tasks" / "0"
    tasks.mkdir(parents=True)
    np.savez(tasks / "task.npz", multimap_native=True)
    assert (
        data._method_key("multimap", "frozen_benchmark", str(tmp_path / "aggregate.npz"))
        == "multimap"
    )


def test_release_count_and_unknown_identity_checks_are_unchanged(tmp_path):
    table = _table()
    _write(tmp_path, table.iloc[:-1])
    with pytest.raises(ValueError, match="expected 69"):
        data.load_canonical_frame(tmp_path, datasets=("d1",), require_source_caches=False)
    table.loc[0, "method"] = "unknown"
    _write(tmp_path, table)
    with pytest.raises(KeyError, match="missing from"):
        data.load_canonical_frame(tmp_path, datasets=("d1",), require_source_caches=False)
    # Count-only release validation does not reject repeated seed identities today.
    table = _table()
    table["seed"] = 0
    _write(tmp_path, table)
    assert (
        len(data.load_canonical_frame(tmp_path, datasets=("d1",), require_source_caches=False))
        == 69
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("benchmark_manifest_hash", "wrong"),
        ("preprocessing_hashes", {"d1": "wrong"}),
        ("sample_hashes", {"d1": "wrong"}),
        ("task_seeds", [0, 1, 9]),
    ],
)
def test_supplementary_provenance_rejection_is_unchanged(tmp_path, field, value):
    arm = tmp_path / "corrected"
    arm.mkdir()
    manifest = dict(
        benchmark_manifest_hash="abcdef000",
        preprocessing_hashes={"d1": "prep"},
        sample_hashes={"d1": "sample"},
        task_seeds=[0, 1, 2],
    )
    manifest[field] = value
    (arm / "supplementary_manifest.json").write_text(json.dumps(manifest))
    frozen = pd.DataFrame(
        dict(seed=[0, 1, 2], preprocessing_hash=["prep"] * 3, sample_hash=["sample"] * 3)
    )
    with pytest.raises(ValueError):
        data._validate_supplementary_manifest(
            tmp_path,
            benchmark_root=tmp_path / "benchmark--abcdef",
            arm="corrected",
            dataset="d1",
            base_rows=frozen,
        )


def test_loader_operation_order_remains_selection_before_annotation(tmp_path, monkeypatch):
    _write(tmp_path, _table())
    events = []
    original_read, original_mask, original_key = (
        data._read_metric_table,
        data.canonical_mask,
        data._method_key,
    )

    def read(*args):
        events.append("load")
        return original_read(*args)

    def replace(table, **kwargs):
        events.append("replace")
        return table.assign(result_source="frozen_benchmark", supplementary_manifest="")

    def mask(table, policy):
        events.append("select")
        return original_mask(table, policy)

    def key(*args, **kwargs):
        events.append("annotate")
        return original_key(*args, **kwargs)

    monkeypatch.setattr(data, "_read_metric_table", read)
    monkeypatch.setattr(data, "_replace_with_supplementary", replace)
    monkeypatch.setattr(data, "canonical_mask", mask)
    monkeypatch.setattr(data, "_method_key", key)
    data.load_canonical_frame(
        tmp_path, datasets=("d1",), supplementary_root=tmp_path, require_source_caches=False
    )
    assert events == ["load", "replace", "select", "select"] + ["annotate"] * 69
