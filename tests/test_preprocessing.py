"""Check shared preprocessing, raw-count preservation, and dataset identity."""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from dataclasses import replace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from ne_aligned_utils.benchmark.data import _read_feature_list
from ne_aligned_utils.config import ROOT
from ne_aligned_utils.data import preprocessing


@pytest.fixture(scope="module")
def unified_builder():
    path = ROOT / "scripts/prepare/build_dataset.py"
    spec = importlib.util.spec_from_file_location("unified_real_data_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_unified_builder_registers_all_real_datasets(unified_builder):
    assert set(unified_builder.BUILD_SPECS) == {
        "nCoV",
        "human_brain",
        "lupus",
        "PBMC",
        "mouse_atlas",
        "hematopoietic",
        "hematopoietic_shared",
        "yao_10x_ent_l2l3",
    }
    shared = unified_builder.BUILD_SPECS["hematopoietic_shared"]
    assert shared.raw_dataset == "hematopoietic"


def test_preparation_accepts_configured_subset_and_rejects_unknown_datasets(unified_builder):
    with (ROOT / "configs/benchmark.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    manifest["datasets"] = {"PBMC": manifest["datasets"]["PBMC"]}
    unified_builder.validate_dataset_declarations(manifest)
    manifest["datasets"]["unknown"] = manifest["datasets"]["PBMC"]
    with pytest.raises(ValueError, match="Dataset declarations disagree"):
        unified_builder.validate_dataset_declarations(manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("raw_adapter", "load_unknown", "raw_adapter="),
        ("source_files", ["data/PBMC/raw/wrong.txt"], "configured source files"),
    ],
)
def test_preparation_rejects_inconsistent_declarations(unified_builder, field, value, message):
    with (ROOT / "configs/benchmark.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    manifest["datasets"] = {"PBMC": manifest["datasets"]["PBMC"]}
    manifest["datasets"]["PBMC"][field] = value
    with pytest.raises(ValueError, match=message):
        unified_builder.validate_dataset_declarations(manifest)


def test_preparation_rejects_adapter_missing_from_registry(monkeypatch, unified_builder):
    with (ROOT / "configs/benchmark.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    manifest["datasets"] = {"PBMC": manifest["datasets"]["PBMC"]}
    monkeypatch.setattr(unified_builder, "BENCHMARK_DATASETS", ())
    with pytest.raises(ValueError, match="Dataset declarations disagree"):
        unified_builder.validate_dataset_declarations(manifest)


def test_preparation_keeps_generated_data_separate_from_external_raw_root(
    tmp_path, monkeypatch, unified_builder
):
    raw_root = tmp_path / "external-raw"
    prepared_root = tmp_path / "local-prepared"
    raw_dir = raw_root / "PBMC" / "raw"
    raw_dir.mkdir(parents=True)
    spec = unified_builder.BUILD_SPECS["PBMC"]
    for name in spec.source_files:
        (raw_dir / name).write_bytes(b"raw source")
    monkeypatch.setenv("NE_ALIGNED_RAW_ROOT", str(raw_root))
    monkeypatch.setenv("NE_ALIGNED_DATA_ROOT", str(prepared_root))
    loaded_from = []

    def load_pbmc(path):
        loaded_from.append(path)

    monkeypatch.setitem(
        unified_builder.BUILD_SPECS,
        "PBMC",
        replace(spec, loader=load_pbmc),
    )
    prepared_at = []
    monkeypatch.setattr(
        unified_builder, "prepare", lambda _adata, path, *_args, **_kwargs: prepared_at.append(path)
    )
    monkeypatch.setattr(unified_builder, "check_saved", lambda _out: None)
    monkeypatch.setattr(
        unified_builder,
        "write_preprocessing_manifest",
        lambda *_args, **_kwargs: {"preprocessing_hash": "test"},
    )
    unified_builder.main(["PBMC"])
    assert loaded_from == [raw_dir]
    assert prepared_at == [prepared_root / "PBMC"]
    assert not (raw_root / "benchmark").exists()
    assert all((raw_dir / name).read_bytes() == b"raw source" for name in spec.source_files)


def test_human_brain_adapter_selects_replicates_before_common_qc(tmp_path, unified_builder):
    raw = tmp_path / "raw"
    raw.mkdir()
    obs = pd.DataFrame(
        {"replicate_id": ["4", "2", "1"]},
        index=["cell4", "cell2", "cell1"],
    )
    ad.AnnData(
        X=np.asarray([[1, 0], [2, 0], [0, 3]], dtype=float),
        obs=obs,
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(raw / "replicates.h5ad")
    pd.DataFrame(
        {
            "replicate_id": [4, 2, 1],
            "cell_types": ["A", "discarded", "B"],
        },
        index=obs.index,
    ).to_csv(raw / "cell_metadata.csv")

    loaded = unified_builder.load_human_brain(raw)
    assert list(loaded.obs_names) == ["cell4", "cell1"]
    assert list(loaded.obs["method"].astype(str)) == ["data1", "data2"]
    assert list(loaded.obs["cell_type"].astype(str)) == ["A", "B"]
    np.testing.assert_array_equal(loaded.layers["counts"], loaded.X)


def test_hematopoietic_shared_adapter_subsets_before_prepare(
    tmp_path, monkeypatch, unified_builder
):
    obs = pd.DataFrame(
        {
            "method": ["data1", "data1", "data2", "data2"],
            "cell_type": ["CMP", "other", "GMP", "MEP"],
        },
        index=["a", "b", "c", "d"],
    )
    source = ad.AnnData(np.ones((4, 2)), obs=obs)
    source.layers["counts"] = source.X.copy()
    monkeypatch.setattr(unified_builder, "_load_tran_two_batch", lambda _path: source)

    loaded = unified_builder.load_hematopoietic_shared(tmp_path)
    assert list(loaded.obs_names) == ["a", "c", "d"]
    assert set(loaded.obs["cell_type"].astype(str)) == {"CMP", "GMP", "MEP"}


def test_canonical_data_write_requires_explicit_permission(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(preprocessing, "DATA_DIR", data_dir)

    with pytest.raises(ValueError, match="Refusing to overwrite canonical data"):
        preprocessing.validate_output_path(data_dir / "PBMC")

    assert (
        preprocessing.validate_output_path(data_dir / "PBMC", allow_canonical_write=True)
        == data_dir / "PBMC"
    )
    staging = tmp_path / "results" / "preprocessing" / "PBMC"
    assert preprocessing.validate_output_path(staging) == staging


def test_joint_pca_is_reproducible():
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=(400, 60))
    x2 = rng.normal(size=(300, 60))

    first = preprocessing.joint_pca(x1, x2, 20)
    second = preprocessing.joint_pca(x1, x2, 20)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_standard_writer_uses_one_file_format(tmp_path):
    output = tmp_path / "prepared"
    x1 = np.arange(6).reshape(2, 3)
    x2 = np.arange(3).reshape(1, 3)
    metadata = pd.DataFrame(
        {"method": ["data1", "data1", "data2"], "cell_type": ["T", "B", "T"]},
        index=["a", "b", "c"],
    )

    preprocessing.write_dataset_files(
        output,
        X1=x1,
        X2=x2,
        batch_names=("first", "second"),
        features=("g1", "g2", "g3"),
        metadata=metadata,
    )

    assert _read_feature_list(output / "features.txt") == ("g1", "g2", "g3")
    assert tuple(np.load(output / "dataset_names.npy", allow_pickle=True)) == ("first", "second")
    assert tuple(np.load(output / "cell_type_names.npy", allow_pickle=True)) == ("B", "T")
    np.testing.assert_array_equal(np.load(output / "labels1.npy"), [1, 0])
    np.testing.assert_array_equal(np.load(output / "labels2.npy"), [1])
    assert list(pd.read_csv(output / "cells_metadata.csv", index_col=0).index) == ["a", "b", "c"]


def test_preprocessing_manifest_uses_portable_provenance_paths(tmp_path):
    source = tmp_path / "raw" / "source.h5ad"
    source.parent.mkdir()
    source.write_bytes(b"source")
    output = tmp_path / "prepared" / "example"
    output.mkdir(parents=True)
    out = {
        "dataset": "example",
        "res_path": output,
        "batch_names": ("a", "b"),
        "features": np.asarray(["g1"]),
        "X1": np.asarray([[1.0]]),
        "X2": np.asarray([[2.0]]),
        "X1_counts": np.asarray([[1.0]]),
        "X2_counts": np.asarray([[2.0]]),
        "X1_pca": np.asarray([[0.0]]),
        "X2_pca": np.asarray([[1.0]]),
        "adata_hv": ad.AnnData(
            np.ones((2, 1)),
            obs=pd.DataFrame(index=["c1", "c2"]),
        ),
    }

    manifest = preprocessing.write_preprocessing_manifest(
        out,
        source_paths=[source],
        loader="load_example",
        preprocessing_seed=0,
        n_top_genes=1,
        n_pca=1,
    )

    assert manifest["provenance"]["source_files"] == ["source.h5ad"]
    assert manifest["provenance"]["output_directory"] == "example"
    assert str(tmp_path) not in str(manifest)


def test_feature_reader_accepts_old_headerless_files(tmp_path):
    path = tmp_path / "features.txt"
    path.write_text("g1\ng2\n")
    assert _read_feature_list(path) == ("g1", "g2")


def test_standard_writer_rejects_mismatched_rows(tmp_path):
    with pytest.raises(ValueError, match="metadata rows"):
        preprocessing.write_dataset_files(
            tmp_path / "prepared",
            X1=np.zeros((2, 3)),
            X2=np.zeros((1, 3)),
            batch_names=("first", "second"),
            features=("g1", "g2", "g3"),
            metadata=pd.DataFrame({"cell_type": ["T", "B"]}),
        )


def test_prepare_orders_interleaved_metadata_like_split_arrays(monkeypatch, tmp_path):
    counts = np.array(
        [
            [4, 0, 1],
            [0, 5, 1],
            [3, 1, 0],
            [1, 4, 0],
        ],
        dtype=float,
    )
    obs = pd.DataFrame(
        {
            "method": ["data2", "data1", "data2", "data1"],
            "cell_type": ["B", "T", "T", "B"],
        },
        index=["m0", "f0", "m1", "f1"],
    )
    adata = ad.AnnData(counts.copy(), obs=obs)
    adata.layers["counts"] = counts.copy()

    monkeypatch.setattr(preprocessing.sc.pp, "filter_cells", lambda *args, **kwargs: None)
    monkeypatch.setattr(preprocessing.sc.pp, "filter_genes", lambda *args, **kwargs: None)

    def mark_all_genes(adata, **kwargs):
        adata.var["highly_variable"] = True

    monkeypatch.setattr(preprocessing.sc.pp, "highly_variable_genes", mark_all_genes)
    monkeypatch.setattr(preprocessing.sc.pp, "normalize_total", lambda *args, **kwargs: None)
    monkeypatch.setattr(preprocessing.sc.pp, "log1p", lambda *args, **kwargs: None)

    out = preprocessing.prepare(
        adata,
        tmp_path / "unused",
        "toy",
        n_pca=None,
        save=False,
    )

    assert list(out["adata_hv"].obs_names) == ["f0", "f1", "m0", "m1"]
    np.testing.assert_array_equal(out["X1"], counts[[1, 3]])
    np.testing.assert_array_equal(out["X2"], counts[[0, 2]])
    np.testing.assert_array_equal(out["y1"], [1, 0])
    np.testing.assert_array_equal(out["y2"], [0, 1])
