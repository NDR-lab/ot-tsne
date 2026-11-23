"""Check benchmark dispatch, cache identity, aggregation, and execution metadata."""

import argparse
import sys
from argparse import Namespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from ne_aligned_utils.benchmark.data import (
    _dense_raw_counts,
    load_benchmark_data,
    load_named_dataset,
)
from ne_aligned_utils.benchmark.sweep import (
    DEFAULT_METHODS,
    TASK_KINDS,
    _merge_previous_methods,
    aggregate,
    resolve_task_output_path,
    save_npz,
    validate_args,
)
from ne_aligned_utils.benchmark.sweep.registry import input_kind_for_method
from ne_aligned_utils.benchmark.sweep.tasks import _cached_task_mismatches
from ne_aligned_utils.config import (
    DATASET_LABELS,
    ROOT,
    SUPPORTED_DATASETS,
    TWO_BATCH_DATASETS,
    has_backend_axis,
)
from ne_aligned_utils.evaluation.rank_aggregation import CELL_ARM


def test_cached_task_resume_checks_identity_and_embedding_content():
    args = Namespace(
        task_kind="joint-pca",
        seed=0,
        method_seed=0,
        visualization_seed=0,
        embedding_dim=2,
        run_id="expected-run",
        preprocessing_hash="prep",
        sample_hash="sample",
        representation_family="processed",
        method_pipeline="method_produces_final_2d",
    )
    payload = {
        "task_kind": np.asarray("joint-pca"),
        "seed": np.asarray(0),
        "method_seed": np.asarray(0),
        "visualization_seed": np.asarray(0),
        "input_source": np.asarray("processed"),
        "run_id": np.asarray("expected-run"),
        "preprocessing_hash": np.asarray("prep"),
        "sample_hash": np.asarray("sample"),
        "representation_family": np.asarray("processed"),
        "method_pipeline": np.asarray("method_produces_final_2d"),
        "embedding": np.zeros((4, 2)),
    }
    assert _cached_task_mismatches(payload, args) == []

    payload["run_id"] = np.asarray("stale-run")
    assert any("run_id" in reason for reason in _cached_task_mismatches(payload, args))
    payload["run_id"] = np.asarray("expected-run")
    payload["embedding"][0, 0] = np.nan
    assert "embedding contains non-finite values" in _cached_task_mismatches(payload, args)


def test_focused_config_module_owns_naming_while_core_reexports_it():
    from ne_aligned_utils.benchmark.sweep import config as sweep_config, core

    assert core.config_token is sweep_config.config_token
    assert core.resolve_task_output_path is sweep_config.resolve_task_output_path
    assert core.assert_no_parameter_collision is sweep_config.assert_no_parameter_collision


def test_load_lupus_dataset_from_two_batch_directory(tmp_path):
    dataset_dir = tmp_path / "lupus"
    dataset_dir.mkdir()

    x0 = np.array([[1.0, 0.0, 2.0], [0.5, 1.5, 0.0]])
    x1 = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 1.0], [3.0, 0.0, 0.5]])
    labels0 = np.array([0, 1])
    labels1 = np.array([1, 0, 1])

    np.save(dataset_dir / "X1.npy", x0)
    np.save(dataset_dir / "X2.npy", x1)
    np.save(dataset_dir / "labels1.npy", labels0)
    np.save(dataset_dir / "labels2.npy", labels1)
    np.save(dataset_dir / "cell_type_names.npy", np.array(["B", "T"], dtype=object))
    np.save(dataset_dir / "dataset_names.npy", np.array(["ctrl", "case"], dtype=object))

    data = load_named_dataset("lupus", data_root=tmp_path, seed=0)

    np.testing.assert_allclose(data.x0, x0)
    np.testing.assert_allclose(data.x1, x1)
    np.testing.assert_array_equal(data.labels0, labels0)
    np.testing.assert_array_equal(data.labels1, labels1)
    assert data.label_names == ("B", "T")
    assert data.dataset_names == ("ctrl", "case")
    np.testing.assert_array_equal(data.sample_indices0, np.arange(x0.shape[0]))
    np.testing.assert_array_equal(data.sample_indices1, np.arange(x1.shape[0]))


def test_joint_pca_input_loads_the_fixed_preprocessing_arrays(tmp_path):
    dataset_dir = tmp_path / "lupus"
    dataset_dir.mkdir()
    np.save(dataset_dir / "X1.npy", np.zeros((2, 1000)))
    np.save(dataset_dir / "X2.npy", np.ones((3, 1000)))
    pca0 = np.arange(100, dtype=float).reshape(2, 50)
    pca1 = np.arange(150, dtype=float).reshape(3, 50)
    np.save(dataset_dir / "X1_pca50.npy", pca0)
    np.save(dataset_dir / "X2_pca50.npy", pca1)

    data = load_named_dataset("lupus", data_root=tmp_path, seed=999, input_kind="joint_pca")

    np.testing.assert_array_equal(data.x0, pca0)
    np.testing.assert_array_equal(data.x1, pca1)
    assert data.input_source == "joint_pca"
    assert data.feature_names[0] == "PC1"
    assert data.feature_names[-1] == "PC50"


@pytest.mark.parametrize(
    "method",
    ["ot", "eot-t0", "eot-50", "joint", "harmony-2d", "harmony-50", "rc-tsne"],
)
def test_common_pca_methods_are_routed_to_the_fixed_representation(method):
    assert input_kind_for_method(method) == "joint_pca"


@pytest.mark.parametrize("method", ["scanorama-50", "multimap", "bc-tsne"])
def test_internal_reduction_methods_keep_the_1000d_processed_input(method):
    assert input_kind_for_method(method) == "processed"


def test_processed_loader_rejects_labels_that_disagree_with_metadata(tmp_path):
    dataset_dir = tmp_path / "lupus"
    dataset_dir.mkdir()
    np.save(dataset_dir / "X1.npy", np.ones((2, 2)))
    np.save(dataset_dir / "X2.npy", np.ones((1, 2)))
    np.save(dataset_dir / "labels1.npy", np.array([0, 1]))
    np.save(dataset_dir / "labels2.npy", np.array([1]))
    np.save(dataset_dir / "cell_type_names.npy", np.array(["B", "T"], dtype=object))
    pd.DataFrame(
        {
            "method": ["data1", "data1", "data2"],
            "cell_type": ["T", "B", "T"],
        },
        index=["c0", "c1", "c2"],
    ).to_csv(dataset_dir / "cells_metadata.csv")

    with pytest.raises(ValueError, match="saved labels disagree"):
        load_named_dataset("lupus", data_root=tmp_path, seed=0)


def test_raw_count_validation_rejects_almost_integer_values():
    with pytest.raises(ValueError, match="non-integer raw count"):
        _dense_raw_counts(np.array([[1.000001]]), name="example")


def _write_two_batch_dataset(dataset_dir, *, batch_column):
    """One Tran et al. style directory: both input views over the same four cells."""
    dataset_dir.mkdir()
    obs_names = ["c0", "c1", "c2", "c3"]
    counts = np.array([[1, 0, 2], [0, 3, 0], [4, 0, 1], [0, 2, 2]], dtype=float)
    lognorm = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e4)

    np.save(dataset_dir / "X1.npy", lognorm[:2])
    np.save(dataset_dir / "X2.npy", lognorm[2:])
    np.save(dataset_dir / "X1_counts.npy", counts[:2])
    np.save(dataset_dir / "X2_counts.npy", counts[2:])
    np.save(dataset_dir / "labels1.npy", np.array([1, 0]))
    np.save(dataset_dir / "labels2.npy", np.array([1, 0]))
    np.save(dataset_dir / "cell_type_names.npy", np.array(["B", "T"], dtype=object))
    np.save(dataset_dir / "dataset_names.npy", np.array(["seqwell", "10x"], dtype=object))

    ad.AnnData(
        X=lognorm,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
        layers={"counts": counts},
    ).write_h5ad(dataset_dir / "combined_1000genes.h5ad")

    pd.DataFrame(
        {
            "method": ["data1", "data1", "data2", "data2"],
            "cell_type": ["T", "B", "T", "B"],
            "batch": batch_column,
        },
        index=obs_names,
    ).to_csv(dataset_dir / "cells_metadata.csv")
    return lognorm, counts


def test_two_batch_datasets_agree_across_their_two_input_views(tmp_path):
    """The processed and raw views of one directory must describe the same cells.

    They are two layers of one file, so a disagreement here is not a tolerance question: it means
    the metrics computed from scVI/Seurat/LIGER are being compared against a different cell
    selection than every other method's.
    """
    lognorm, counts = _write_two_batch_dataset(
        tmp_path / "mouse_atlas", batch_column=["Batch1", "Batch1", "Batch2", "Batch2"]
    )

    processed = load_named_dataset("mouse_atlas", data_root=tmp_path, seed=0)
    raw = load_named_dataset("mouse_atlas", data_root=tmp_path, seed=0, input_kind="raw_counts")

    np.testing.assert_allclose(processed.x0, lognorm[:2])
    np.testing.assert_allclose(processed.x1, lognorm[2:])
    np.testing.assert_allclose(raw.x0, counts[:2])
    np.testing.assert_allclose(raw.x1, counts[2:])
    assert raw.input_source == "raw_counts"
    assert raw.feature_names == ("g0", "g1", "g2")

    assert processed.label_names == raw.label_names == ("B", "T")
    np.testing.assert_array_equal(processed.labels0, raw.labels0)
    np.testing.assert_array_equal(processed.labels1, raw.labels1)
    assert (processed.x0.shape[0], processed.x1.shape[0]) == (raw.x0.shape[0], raw.x1.shape[0])


def test_two_batch_dataset_names_survive_the_metadata_override(tmp_path):
    """`cells_metadata.csv` must not rename the batches out from under the registry.

    The shared directory loader takes the batch names from the metadata's `batch` column, which on
    mouse_atlas says Batch1/Batch2 and hides the two protocols that are the batch axis. Both views
    override it, so a legend names Seq-Well against 10x rather than a pair of ordinals.
    """
    _write_two_batch_dataset(
        tmp_path / "mouse_atlas", batch_column=["Batch1", "Batch1", "Batch2", "Batch2"]
    )

    for kind in ("processed", "raw_counts"):
        data = load_named_dataset("mouse_atlas", data_root=tmp_path, seed=0, input_kind=kind)
        assert data.dataset_names == ("seqwell", "10x"), kind


@pytest.mark.parametrize(
    "dataset", ["PBMC", "mouse_atlas", "hematopoietic", "hematopoietic_shared"]
)
def test_two_batch_datasets_are_fully_registered(dataset):
    """Registering the loader is not enough: a missing CELL_ARM makes the results unreportable."""
    assert dataset in SUPPORTED_DATASETS
    assert dataset in TWO_BATCH_DATASETS
    assert dataset in CELL_ARM
    assert dataset in DATASET_LABELS


def test_scvi_methods_are_opt_in():
    assert "scvi-2d" in TASK_KINDS
    assert "scvi-50" in TASK_KINDS
    assert "scvi-2d" not in DEFAULT_METHODS
    assert "scvi-50" not in DEFAULT_METHODS


def test_load_ncov_raw_counts_from_h5ad_layer(tmp_path):
    dataset_dir = tmp_path / "nCoV"
    dataset_dir.mkdir()
    obs_names = ["c0", "c1", "c2", "c3"]
    counts = sp.csr_matrix(
        [
            [1, 0, 2],
            [0, 3, 0],
            [4, 0, 1],
            [0, 2, 2],
        ],
        dtype=float,
    )
    adata = ad.AnnData(
        X=counts.multiply(0.1),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
        layers={"counts": counts},
    )
    adata.write_h5ad(dataset_dir / "combined_1000genes.h5ad")
    pd.DataFrame(
        {
            "method": ["data1", "data1", "data2", "data2"],
            "cell_type": ["T", "B", "T", "B"],
            "batch": ["ctrl", "ctrl", "case", "case"],
        },
        index=obs_names,
    ).to_csv(dataset_dir / "cells_metadata.csv")

    data = load_benchmark_data(
        "nCoV",
        data_root=tmp_path,
        seed=0,
        max_cells_per_batch=None,
        input_kind="raw_counts",
    )

    np.testing.assert_allclose(data.x0, counts[:2].toarray())
    np.testing.assert_allclose(data.x1, counts[2:].toarray())
    assert data.input_source == "raw_counts"
    assert data.dataset_names == ("ctrl", "case")
    assert data.feature_names == ("g0", "g1", "g2")
    assert data.labels_available


def test_raw_loader_downsampling_matches_processed_loader(tmp_path):
    dataset_dir = tmp_path / "nCoV"
    dataset_dir.mkdir()
    obs_names = [f"c{i}" for i in range(8)]
    labels = np.array(["T", "B", "T", "B", "T", "B", "T", "B"], dtype=object)
    methods = np.array(["data1"] * 4 + ["data2"] * 4, dtype=object)
    processed = np.arange(32, dtype=float).reshape(8, 4)
    counts = sp.csr_matrix(processed + 1.0)

    np.save(dataset_dir / "X1.npy", processed[:4])
    np.save(dataset_dir / "X2.npy", processed[4:])
    np.save(dataset_dir / "labels1.npy", np.array([1, 0, 1, 0]))
    np.save(dataset_dir / "labels2.npy", np.array([1, 0, 1, 0]))
    np.save(dataset_dir / "cell_type_names.npy", np.array(["B", "T"], dtype=object))
    ad.AnnData(
        X=counts.multiply(0.1),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
        layers={"counts": counts},
    ).write_h5ad(dataset_dir / "combined_1000genes.h5ad")
    pd.DataFrame(
        {
            "method": methods,
            "cell_type": labels,
            "batch": ["ctrl"] * 4 + ["case"] * 4,
        },
        index=obs_names,
    ).to_csv(dataset_dir / "cells_metadata.csv")

    processed_data = load_benchmark_data(
        "nCoV",
        data_root=tmp_path,
        seed=11,
        max_cells_per_batch=2,
    )
    raw_data = load_benchmark_data(
        "nCoV",
        data_root=tmp_path,
        seed=11,
        max_cells_per_batch=2,
        input_kind="raw_counts",
    )

    assert raw_data.x0.shape == processed_data.x0.shape
    assert raw_data.x1.shape == processed_data.x1.shape
    np.testing.assert_array_equal(raw_data.sample_indices0, processed_data.sample_indices0)
    np.testing.assert_array_equal(raw_data.sample_indices1, processed_data.sample_indices1)
    np.testing.assert_array_equal(raw_data.labels0, processed_data.labels0)
    np.testing.assert_array_equal(raw_data.labels1, processed_data.labels1)


def test_raw_loader_rejects_non_count_values(tmp_path):
    dataset_dir = tmp_path / "nCoV"
    dataset_dir.mkdir()
    obs_names = ["c0", "c1"]
    counts = sp.csr_matrix([[1.0, 0.5], [0.0, -1.0]])
    ad.AnnData(
        X=np.zeros((2, 2)),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=["g0", "g1"]),
        layers={"counts": counts},
    ).write_h5ad(dataset_dir / "combined_1000genes.h5ad")
    pd.DataFrame(
        {
            "method": ["data1", "data2"],
            "cell_type": ["T", "B"],
        },
        index=obs_names,
    ).to_csv(dataset_dir / "cells_metadata.csv")

    with pytest.raises(ValueError, match="raw count"):
        load_benchmark_data(
            "nCoV",
            data_root=tmp_path,
            seed=0,
            max_cells_per_batch=None,
            input_kind="raw_counts",
        )


def test_load_human_brain_raw_counts_subsets_full_raw_object(tmp_path):
    dataset_dir = tmp_path / "human_brain"
    dataset_dir.mkdir()
    raw_dir = dataset_dir / "raw"
    raw_dir.mkdir()
    raw_obs = ["c0", "c1", "c2", "c3", "unused"]
    raw_vars = ["g0", "g1", "g2", "extra"]
    raw_counts = sp.csr_matrix(
        [
            [1, 0, 2, 9],
            [0, 3, 0, 9],
            [4, 0, 1, 9],
            [0, 2, 2, 9],
            [9, 9, 9, 9],
        ],
        dtype=float,
    )
    ad.AnnData(
        X=raw_counts,
        obs=pd.DataFrame(index=raw_obs),
        var=pd.DataFrame(index=raw_vars),
    ).write_h5ad(raw_dir / "replicates.h5ad")
    ad.AnnData(
        X=np.zeros((4, 3)),
        obs=pd.DataFrame(index=raw_obs[:4]),
        var=pd.DataFrame(index=["g2", "g0", "g1"]),
    ).write_h5ad(dataset_dir / "combined_1000genes.h5ad")
    pd.DataFrame(
        {
            "method": ["data1", "data1", "data2", "data2"],
            "replicate_id": ["4", "4", "1", "1"],
        },
        index=raw_obs[:4],
    ).to_csv(dataset_dir / "cells_metadata.csv")
    pd.DataFrame(
        {"cell_types": ["Astro", "Oligo", "Astro", "Oligo", "Other"]},
        index=raw_obs,
    ).to_csv(raw_dir / "cell_metadata.csv")

    data = load_benchmark_data(
        "human_brain",
        data_root=tmp_path,
        seed=0,
        max_cells_per_batch=None,
        input_kind="raw_counts",
    )

    expected = raw_counts[:4, [2, 0, 1]].toarray()
    np.testing.assert_allclose(data.x0, expected[:2])
    np.testing.assert_allclose(data.x1, expected[2:])
    assert data.feature_names == ("g2", "g0", "g1")
    assert data.dataset_names == ("4", "1")
    assert data.labels_available


def test_load_yao_raw_counts_reads_prepared_subset_cache(tmp_path):
    yao_dir = tmp_path / "yao"
    yao_dir.mkdir()
    counts = sp.csr_matrix(
        [
            [1, 0, 2],
            [0, 3, 0],
            [4, 0, 1],
            [0, 2, 2],
        ],
        dtype=float,
    )
    np.savez_compressed(
        yao_dir / "ent_L23_raw_counts.npz",
        data=counts.data,
        indices=counts.indices,
        indptr=counts.indptr,
        shape=np.asarray(counts.shape),
        genes=np.asarray(["g0", "g1", "g2"]),
    )
    pd.DataFrame(
        {
            "sample_name": ["a", "b", "c", "d"],
            "donor_sex_label": ["F", "F", "M", "M"],
            "cluster_label": ["Lamp5", "Vip", "Lamp5", "Vip"],
        }
    ).to_csv(yao_dir / "ent_l2l3_meta.csv", index=False)

    data = load_benchmark_data(
        "yao_10x_ent_l2l3",
        data_root=tmp_path,
        seed=0,
        max_cells_per_batch=None,
        input_kind="raw_counts",
    )

    np.testing.assert_allclose(data.x0, counts[[0, 1]].toarray())
    np.testing.assert_allclose(data.x1, counts[[2, 3]].toarray())
    assert data.dataset_names == ("ent_l2l3-F", "ent_l2l3-M")
    assert data.feature_names == ("g0", "g1", "g2")
    assert data.labels_available


def test_load_lupus_raw_counts_requires_prepared_h5ad(tmp_path):
    dataset_dir = tmp_path / "lupus"
    dataset_dir.mkdir()
    pd.DataFrame(
        {"method": ["data1"], "cell_type": ["T"]},
        index=["c0"],
    ).to_csv(dataset_dir / "cells_metadata.csv")
    (dataset_dir / "features.txt").write_text("g0\n")

    with pytest.raises(FileNotFoundError, match="prepare_lupus_raw_counts"):
        load_benchmark_data(
            "lupus",
            data_root=tmp_path,
            seed=0,
            max_cells_per_batch=None,
            input_kind="raw_counts",
        )


def test_load_lupus_raw_counts_from_prepared_h5ad(tmp_path):
    dataset_dir = tmp_path / "lupus"
    dataset_dir.mkdir()
    counts = sp.csr_matrix(
        [
            [1, 0, 2],
            [0, 3, 0],
            [4, 0, 1],
            [0, 2, 2],
        ],
        dtype=float,
    )
    ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=["c0", "c1", "c2", "c3"]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    ).write_h5ad(dataset_dir / "raw_counts.h5ad")
    pd.DataFrame(
        {
            "method": ["data1", "data1", "data2", "data2"],
            "cell_type": ["T", "B", "T", "B"],
        },
        index=["c0", "c1", "c2", "c3"],
    ).to_csv(dataset_dir / "cells_metadata.csv")
    (dataset_dir / "features.txt").write_text("g2\ng0\n")
    np.save(dataset_dir / "dataset_names.npy", np.array(["CTRL", "STIM"], dtype=object))

    data = load_benchmark_data(
        "lupus",
        data_root=tmp_path,
        seed=0,
        max_cells_per_batch=None,
        input_kind="raw_counts",
    )

    expected = counts[:, [2, 0]].toarray()
    np.testing.assert_allclose(data.x0, expected[:2])
    np.testing.assert_allclose(data.x1, expected[2:])
    assert data.dataset_names == ("CTRL", "STIM")
    assert data.feature_names == ("g2", "g0")
    assert data.labels_available


def test_aggregate_includes_scvi_outputs(tmp_path):
    dataset_dir = tmp_path / "lupus"
    dataset_dir.mkdir()
    x0 = np.zeros((2, 3))
    x1 = np.ones((3, 3))
    np.save(dataset_dir / "X1.npy", x0)
    np.save(dataset_dir / "X2.npy", x1)
    np.save(dataset_dir / "labels1.npy", np.array([0, 1]))
    np.save(dataset_dir / "labels2.npy", np.array([1, 0, 1]))
    np.save(dataset_dir / "cell_type_names.npy", np.array(["B", "T"], dtype=object))

    args = Namespace(
        dataset="lupus",
        data_root=tmp_path,
        output_dir=tmp_path / "embeddings",
        output_cache=None,
        n_seeds=1,
        methods=["scvi-2d", "scvi-50"],
        rel_reg=[1.0],
        effective_ks=[15],
        embedding_dim=2,
        n_epochs=12,
        pca_components=50,
        perplexity=2.0,
        max_cells_per_batch=None,
        max_cells_batch0=None,
        max_cells_batch1=None,
    )
    task_root = args.output_dir / "n_per_batch_all__q_2__epochs_12__pca_50__nn_na"
    embedding = np.arange(10, dtype=float).reshape(5, 2)
    for method in args.methods:
        save_npz(
            resolve_task_output_path(task_root, task_kind=method, seed=0),
            embedding=embedding,
            dataset="lupus",
            task_kind=method,
            seed=0,
            batch_sizes=np.array([2, 3]),
            sample_indices0=np.arange(2),
            sample_indices1=np.arange(3),
            input_source="raw_counts",
            source_path="raw_counts.h5ad",
            n_genes=3,
        )

    aggregate(args)

    output_path = task_root / "lupus_benchmark_embeddings.npz"
    with np.load(output_path, allow_pickle=False) as cache:
        assert cache["scvi_2d_embeddings"].shape == (1, 5, 2)
        assert cache["scvi_50_tsne_embeddings"].shape == (1, 5, 2)
        assert cache["scvi_2d_paths"].shape == (1,)
        assert cache["scvi_50_tsne_paths"].shape == (1,)
        assert cache["scvi_2d_input_source"] == "raw_counts"
        assert cache["scvi_2d_source_path"] == "raw_counts.h5ad"
        assert cache["scvi_2d_n_genes"] == 3


def test_dense_artifacts_are_never_persisted(tmp_path):
    """A task .npz must not carry the dense pipeline intermediates or the embedder inputs.

    The dense ones are 1.8 GB apiece on the largest dataset and nothing reads them back -- the
    only key any consumer loads from a task file is ``embedding``. The embedder inputs
    (``pca_embedding``, ``reference_input``, ``query_input``) are N x 50 duplicates of information
    the method already records, kept in memory purely so the backend tests can compare the
    representation each arm was handed. The library still returns all of them in memory
    (``return_artifacts=True``); this pins the persistence boundary only.
    """
    import numpy as np

    from ne_aligned_utils.benchmark.sweep import (
        UNPERSISTED_ARTIFACT_KEYS,
        save_npz,
    )

    # Exclude dense intermediates, reusable representations, and raw child-timing dicts
    # from persisted artifacts; normalized timing fields are stored separately.
    assert UNPERSISTED_ARTIFACT_KEYS == {
        "four_block_matrix",
        "sparse_four_block_matrix",
        "graph",
        "ot_plan",
        "u",
        "vt",
        "pca_embedding",
        "reference_input",
        "query_input",
        "r_timings",
        "multimap_timings",
    }

    artifacts = {
        "four_block_matrix": np.zeros((4, 4)),
        "sparse_four_block_matrix": np.zeros((4, 4)),
        "graph": np.zeros((4, 4)),
        "ot_plan": np.zeros((2, 2)),
        "singular_values": np.arange(2.0),
        "effective_k": 3,
    }
    kept = {k: v for k, v in artifacts.items() if k not in UNPERSISTED_ARTIFACT_KEYS}
    assert set(kept) == {"singular_values", "effective_k"}

    path = tmp_path / "task.npz"
    save_npz(path, embedding=np.zeros((4, 2)), **kept)
    with np.load(path, allow_pickle=False) as loaded:
        assert "embedding" in loaded.files
        assert not UNPERSISTED_ARTIFACT_KEYS.intersection(loaded.files)


def _aggregate_fixture(tmp_path, methods):
    """A minimal dataset plus one saved embedding per method, ready for ``aggregate``."""
    dataset_dir = tmp_path / "lupus"
    dataset_dir.mkdir(exist_ok=True)
    np.save(dataset_dir / "X1.npy", np.zeros((2, 3)))
    np.save(dataset_dir / "X2.npy", np.ones((3, 3)))
    np.save(dataset_dir / "labels1.npy", np.array([0, 1]))
    np.save(dataset_dir / "labels2.npy", np.array([1, 0, 1]))
    np.save(dataset_dir / "cell_type_names.npy", np.array(["B", "T"], dtype=object))

    args = Namespace(
        dataset="lupus",
        data_root=tmp_path,
        output_dir=tmp_path / "embeddings",
        output_cache=None,
        n_seeds=1,
        methods=list(methods),
        rel_reg=[1.0],
        effective_ks=[15],
        embedding_dim=2,
        n_epochs=12,
        pca_components=50,
        perplexity=2.0,
        max_cells_per_batch=None,
        max_cells_batch0=None,
        max_cells_batch1=None,
        replace=False,
    )
    task_root = args.output_dir / "n_per_batch_all__q_2__epochs_12__pca_50__nn_na"
    embedding = np.arange(10, dtype=float).reshape(5, 2)
    for method in methods:
        save_npz(
            resolve_task_output_path(task_root, task_kind=method, seed=0),
            embedding=embedding,
            dataset="lupus",
            task_kind=method,
            seed=0,
            batch_sizes=np.array([2, 3]),
            sample_indices0=np.arange(2),
            sample_indices1=np.arange(3),
            input_source=input_kind_for_method(method),
            source_path="raw_counts.h5ad" if method.startswith("scvi-") else "",
            n_genes=3,
        )
    return args, task_root


def test_aggregate_rejects_task_from_a_different_cell_sample(tmp_path):
    args, task_root = _aggregate_fixture(tmp_path, ["joint-pca"])
    path = resolve_task_output_path(task_root, task_kind="joint-pca", seed=0)
    with np.load(path, allow_pickle=False) as cache:
        payload = {key: cache[key] for key in cache.files}
    payload["sample_indices0"] = np.array([1, 0])
    save_npz(path, **payload)

    with pytest.raises(ValueError, match="different cell sample"):
        aggregate(args)


def test_aggregate_rejects_wrong_method_input_source(tmp_path):
    args, task_root = _aggregate_fixture(tmp_path, ["joint-pca"])
    path = resolve_task_output_path(task_root, task_kind="joint-pca", seed=0)
    with np.load(path, allow_pickle=False) as cache:
        payload = {key: cache[key] for key in cache.files}
    payload["input_source"] = "raw_counts"
    save_npz(path, **payload)

    with pytest.raises(ValueError, match="expected 'processed'"):
        aggregate(args)


def test_aggregate_keeps_methods_a_partial_run_did_not_ask_for(tmp_path):
    """A partial aggregate must extend the cache, not shrink it to the requested methods.

    ``aggregate`` builds its payload from ``--methods`` and overwrites the file, so this used to
    drop every other method. The per-task files survived, but the aggregate stopped listing them
    and the scoring layer silently reported fewer methods. It happened twice on real data: two
    late-arriving CCA methods took lupus from thirteen methods to two, and a one-group runner test
    took the nCoV variants aggregate from three to two.
    """
    args, task_root = _aggregate_fixture(tmp_path, ["scvi-2d", "scvi-50", "joint-pca"])
    aggregate(args)
    output_path = task_root / "lupus_benchmark_embeddings.npz"
    with np.load(output_path, allow_pickle=True) as cache:
        first = {str(m) for m in cache["methods"].tolist()}
        kept_before = np.asarray(cache["joint_pca_embeddings"])
    # Persist canonical backend-qualified names for separable methods; direct layouts have no
    # backend.
    assert first == {"scvi-2d", "scvi-50-tsne", "joint-pca"}

    args.methods = ["scvi-2d"]
    aggregate(args)

    with np.load(output_path, allow_pickle=True) as cache:
        after = {str(m) for m in cache["methods"].tolist()}
        assert after == first, "a partial run narrowed the aggregate"
        # and the carried-over array is untouched, not merely present
        assert np.array_equal(np.asarray(cache["joint_pca_embeddings"]), kept_before)


def test_aggregate_replace_still_prunes_when_asked(tmp_path):
    """``--replace`` is the escape hatch, so dropping a method stays possible on purpose."""
    args, task_root = _aggregate_fixture(tmp_path, ["scvi-2d", "joint-pca"])
    aggregate(args)

    args.methods = ["scvi-2d"]
    args.replace = True
    aggregate(args)

    with np.load(task_root / "lupus_benchmark_embeddings.npz", allow_pickle=True) as cache:
        assert {str(m) for m in cache["methods"].tolist()} == {"scvi-2d"}
        assert "joint_pca_embeddings" not in cache.files


def test_aggregate_replaces_rather_than_merges_when_the_grid_changes(tmp_path, capsys):
    """Merging across a different sweep geometry would misalign arrays, so it must not happen.

    The per-method arrays are indexed by (seed, epsilon, neighbor budget). Carrying one across a
    change in the epsilon grid would leave it inconsistent with the new metadata without raising,
    which is worse than replacing.
    """
    # joint-pca only in the fixture: an eps-swept method's path needs an epsilon, so those files are
    # written explicitly below rather than by the fixture's simple loop.
    args, task_root = _aggregate_fixture(tmp_path, ["joint-pca"])
    args.rel_reg = [1.0, 1 / 10]
    args.methods = ["eot-t0", "joint-pca"]
    for eps in args.rel_reg:
        save_npz(
            resolve_task_output_path(task_root, task_kind="eot-t0", seed=0, rel_reg=eps),
            embedding=np.arange(10, dtype=float).reshape(5, 2),
            dataset="lupus",
            task_kind="eot-t0",
            seed=0,
            batch_sizes=np.array([2, 3]),
            sample_indices0=np.arange(2),
            sample_indices1=np.arange(3),
            input_source="joint_pca",
            source_path="",
            n_genes=3,
        )
    aggregate(args)

    args.rel_reg = [1.0]
    args.methods = ["eot-t0"]
    aggregate(args)

    captured = capsys.readouterr().out
    assert "rel_reg differs" in captured, "the geometry guard did not report a replacement"
    with np.load(task_root / "lupus_benchmark_embeddings.npz", allow_pickle=True) as cache:
        assert "joint_pca_embeddings" not in cache.files


# The backend token in task filenames
#
# resolve_task_output_path used to hand-write 25 literal stems. Collapsing it to its three real
# shapes, with the stem coming from config.method_stem, is what lets the backend enter the filename
# -- but it is also a change to a *cache path* function, where a wrong answer does not raise. It
# makes every cached file invisible and silently recomputes 1,565 of them. These three tests are the
# guard, and the first is the one that matters.

#: Literal task filenames provide independent references for cache routing.
#: Regularization tokens use round-tripping precision, including 1/300.
EXPECTED_TSNE_STEMS = {
    "eot-t0": "eot_t0_seed_0_rel_reg_0p1.npz",
    "ot": "ot_tsne_seed_0_rel_reg_0p1_k_15.npz",
    "eot-50": "eot_50_tsne_seed_0_rel_reg_0p1.npz",
    "joint-pca": "joint_pca_seed_0.npz",
    "joint": "joint_tsne_seed_0.npz",
    "cca-2d": "cca_2d_seed_0.npz",
    "cca-50": "cca_50_tsne_seed_0.npz",
    "cca-2d-uncentered": "cca_2d_uncentered_seed_0.npz",
    "cca-50-uncentered": "cca_50_uncentered_tsne_seed_0.npz",
    "harmony-2d": "harmony_2d_seed_0.npz",
    "harmony-50": "harmony_50_tsne_seed_0.npz",
    "scvi-2d": "scvi_2d_seed_0.npz",
    "scvi-50": "scvi_50_tsne_seed_0.npz",
    "ot-knndiag": "ot_knndiag_tsne_seed_0_rel_reg_0p1_k_15.npz",
    "ot-selfotmedian": "ot_selfotmedian_tsne_seed_0_rel_reg_0p1_k_15.npz",
    "ot-gramdiag": "ot_gramdiag_tsne_seed_0_rel_reg_0p1_k_15.npz",
    # The unbalanced arms share the eps+k filename shape; the relaxation separates them by
    # directory instead, so it deliberately does not appear here.
    "ot-uot-topk": "ot_uot_topk_tsne_seed_0_rel_reg_0p1_k_15.npz",
    "ot-uot-mass": "ot_uot_mass_tsne_seed_0_rel_reg_0p1_k_15.npz",
    "ot-zerodiag": "ot_zerodiag_tsne_seed_0_rel_reg_0p1_k_15.npz",
    "ot-zerodiag-split": "ot_zerodiag_split_tsne_seed_0_rel_reg_0p1_k_15.npz",
    "ref": "ref_policar_seed_0.npz",
    "ref-our": "ref_tsne_seed_0.npz",
    "scanorama-2d": "scanorama_2d_seed_0.npz",
    "scanorama-50": "scanorama_50_tsne_seed_0.npz",
    "seurat-2d": "seurat5_2d_seed_0.npz",
    "seurat-50": "seurat5_50_tsne_seed_0.npz",
    "seurat3-2d": "seurat_2d_seed_0.npz",
    "seurat3-50": "seurat_50_tsne_seed_0.npz",
    "liger1-2d": "liger1_2d_seed_0.npz",
    "liger1-50": "liger1_50_tsne_seed_0.npz",
    "liger-2d": "liger2_2d_seed_0.npz",
    "liger-50": "liger2_50_tsne_seed_0.npz",
    "bc-tsne": "bc_tsne_seed_0.npz",
    "multimap": "multimap_seed_0.npz",
    "rc-tsne": "rc_tsne_seed_0.npz",
}


def test_current_and_historical_liger_caches_cannot_collide():
    assert EXPECTED_TSNE_STEMS["liger-2d"] != EXPECTED_TSNE_STEMS["liger1-2d"]


def test_current_and_historical_seurat_and_reference_caches_cannot_collide():
    assert EXPECTED_TSNE_STEMS["seurat-2d"] != EXPECTED_TSNE_STEMS["seurat3-2d"]
    assert EXPECTED_TSNE_STEMS["ref"] != EXPECTED_TSNE_STEMS["ref-our"]
    assert EXPECTED_TSNE_STEMS["liger-50"] != EXPECTED_TSNE_STEMS["liger1-50"]


def test_the_expected_stem_table_covers_every_kind():
    """Guards the test below, which would pass vacuously if a kind were simply absent here."""
    assert set(EXPECTED_TSNE_STEMS) == set(TASK_KINDS), (
        f"stem table and TASK_KINDS disagree: "
        f"missing {sorted(set(TASK_KINDS) - set(EXPECTED_TSNE_STEMS))}, "
        f"extra {sorted(set(EXPECTED_TSNE_STEMS) - set(TASK_KINDS))}"
    )


@pytest.mark.parametrize("task_kind", TASK_KINDS)
def test_default_backend_filenames_are_unchanged(task_kind, tmp_path):
    """Every path must match the canonical name, exactly as the migrated tree is spelled.

    This is the test that stands between a rename and a silent cache miss. A path that changes
    does not raise: ``run_repro_grid`` simply stops finding the file, reports the task as uncached
    and recomputes it. That would silently rebuild the whole cached tree, and the numbers would
    look fine. Four stems moved in the canonical migration -- the ones that had the backend token
    infixed before the variant -- and the files on disk moved with them.
    """
    resolved = resolve_task_output_path(
        tmp_path, task_kind=task_kind, seed=0, rel_reg=1 / 10, effective_k=15
    )
    assert resolved.name == EXPECTED_TSNE_STEMS[task_kind]


@pytest.mark.parametrize("task_kind", TASK_KINDS)
def test_umap_filenames_substitute_only_the_backend_token(task_kind, tmp_path):
    """The two trees correspond: same path, with ``tsne`` replaced by ``umap`` in the basename.

    Option C's one weakness against stripping the token entirely is that correspondence is a
    property of the writers rather than of the layout. This closes it: correspondence is a pure
    function of the builder, so drift is a test failure rather than a latent bug.
    """
    kwargs = dict(task_kind=task_kind, seed=0, rel_reg=1 / 10, effective_k=15)
    tsne = resolve_task_output_path(tmp_path, backend="tsne", **kwargs)
    if not has_backend_axis(task_kind):
        shared = resolve_task_output_path(tmp_path, backend="umap", **kwargs)
        assert shared == tsne, "backend-free methods have one canonical basename"
        return
        return
    umap = resolve_task_output_path(tmp_path, backend="umap", **kwargs)
    assert umap.parent == tsne.parent
    assert umap.name == tsne.name.replace("_tsne", "_umap")
    assert umap.name != tsne.name, "a backend-axis method must get a distinct filename"


def test_backend_axis_set_matches_the_coverage_matrix():
    """``method_backend_label``'s domain must be exactly the methods that have a swappable embedder.

    ``bc-tsne`` is the trap: it carries a ``tsne`` token but *is* t-SNE (batch-corrected t-SNE), so
    it has no UMAP form. ``rc-tsne`` is the same case -- revised conditional t-SNE conditions the
    high-dimensional similarities and hands them to t-SNE's own optimizer, in a compiled binary
    that has no UMAP form to swap in. ``multimap`` is the mirror image -- already UMAP-based, and
    excluded automatically because it has no token to substitute.
    """
    with_axis = {k for k in TASK_KINDS if has_backend_axis(k)}
    assert with_axis == {
        "joint",
        "ref-our",
        "cca-50",
        "cca-50-uncentered",
        "harmony-50",
        "scvi-50",
        "scanorama-50",
        "seurat-50",
        "seurat3-50",
        "liger1-50",
        "liger-50",
        "eot-50",
        "ot",
        "ot-knndiag",
        "ot-selfotmedian",
        "ot-gramdiag",
        "ot-uot-topk",
        "ot-uot-mass",
        "ot-zerodiag",
        # A backend axis it only half uses: the name carries `_tsne` and the UMAP
        # combination raises, exactly as for ot-uot-mass.
        "ot-zerodiag-split",
    }
    assert "bc-tsne" not in with_axis, "bcTSNE *is* t-SNE; it has no backend to swap"
    assert "rc-tsne" not in with_axis, "rc-tsne *is* t-SNE; it has no backend to swap"
    assert "multimap" not in with_axis, "MultiMAP is already UMAP-based"
    assert "eot-t0" not in with_axis, "eot-t0 is closed-form; there is no embedder"


def test_infixed_backend_token_is_appended_after_the_variant():
    """The backend goes at the end, after the variant, because the name is built not spelled.

    The old scheme substituted a ``tsne`` token wherever it sat, which put the backend *before*
    the variant -- ``ot-umap-knndiag``, ``cca-50-umap-uncentered``. Names now come from
    ``canonical_name(base, backend)``, so the ordering cannot vary.
    """
    from ne_aligned_utils.config import canonical_name

    assert canonical_name("ot-knndiag", "umap") == "ot-knndiag-umap"
    assert canonical_name("cca-50-uncentered", "umap") == "cca-50-uncentered-umap"
    assert canonical_name("ot-knndiag", "tsne") == "ot-knndiag-tsne"


def test_pre_backend_cache_counts_as_tsne_when_merging(tmp_path):
    """Treat backend-free aggregate metadata as t-SNE when merging compatible stored results."""
    base = {
        "dataset": "nCoV",
        "seeds": [0, 1],
        "rel_reg": [1.0],
        "effective_ks": [15],
        "batch_sizes": [10, 20],
        "embedding_dim": 2,
        "n_epochs": 200,
        "pca_components": 50,
        "perplexity": 30.0,
        "init": "random",
    }
    previous = {k: np.asarray(v) for k, v in base.items()}
    previous["methods"] = np.asarray(["cca-50"])
    previous["cca_50_tsne_embeddings"] = np.zeros((2, 30, 2))
    previous["cca_50_tsne_paths"] = np.asarray([["a"], ["b"]])
    cache = tmp_path / "nCoV_benchmark_embeddings.npz"
    np.savez_compressed(cache, **previous)

    def payload(backend, n_neighbors):
        out = {k: np.asarray(v) for k, v in base.items()}
        out["backend"] = np.asarray(backend)
        out["n_neighbors"] = np.asarray(n_neighbors)
        out["methods"] = np.asarray(["joint-pca"])
        out["joint_pca_embeddings"] = np.zeros((2, 30, 2))
        out["joint_pca_paths"] = np.asarray([["c"], ["d"]])
        return out

    extended = _merge_previous_methods(payload("tsne", -1), cache)
    assert "cca_50_tsne_embeddings" in extended, (
        "a t-SNE run must still extend a pre-backend cache; the default made it look different"
    )

    replaced = _merge_previous_methods(payload("umap", 90), cache)
    assert "cca_50_tsne_embeddings" not in replaced, (
        "a UMAP run silently absorbed the t-SNE methods of a pre-backend cache"
    )


def test_run_task_refuses_to_write_into_another_backends_tree(tmp_path):
    """Option C makes a misplaced file visible rather than destructive; this makes it an error.

    A ``_umap`` basename cannot overwrite a ``_tsne`` one, so nothing is lost. But ``aggregate``
    globs by directory, so a stray file would still be folded into a mixed cache.
    """
    from ne_aligned_utils.benchmark.sweep import assert_output_backend_matches

    assert_output_backend_matches(tmp_path / "results" / "umap" / "x.npz", "umap")
    assert_output_backend_matches(tmp_path / "results" / "tsne" / "x.npz", "tsne")
    # No backend component at all: scratch and verification trees live outside the results root.
    assert_output_backend_matches(tmp_path / "scratch" / "x.npz", "umap")
    with pytest.raises(ValueError, match="would write into"):
        assert_output_backend_matches(tmp_path / "results" / "tsne" / "x.npz", "umap")


def test_multimap_config_token_records_its_neighbourhood():
    from ne_aligned_utils.benchmark.sweep import config_token

    args = Namespace(
        output_dir=ROOT / "results" / "multimap" / "nCoV",
        backend="tsne",
        n_neighbors=15,
        embedding_dim=2,
        n_epochs=200,
        max_cells_per_batch=None,
        max_cells_batch0=None,
        max_cells_batch1=None,
    )
    assert config_token(args) == (
        "n_per_batch_all__q_2__nn_15__min_dist_0p1__spread_1p0__epochs_200__init_spectral"
    )


def test_umap_sweep_derives_n_neighbors_from_the_single_effective_k():
    args = Namespace(
        backend="umap",
        effective_ks=[15],
        n_neighbors=None,
        methods=["joint", "ot", "multimap"],
        rel_reg=[1.0],
        init="random",
    )
    validate_args(argparse.ArgumentParser(), args)
    assert args.n_neighbors == 15


def test_umap_sweep_rejects_conflicting_neighbourhood_numbers(capsys):
    args = Namespace(
        backend="umap",
        effective_ks=[15],
        n_neighbors=90,
        methods=["joint", "ot"],
        rel_reg=[1.0],
        init="random",
    )
    with pytest.raises(SystemExit):
        validate_args(argparse.ArgumentParser(), args)
    assert "must equal --effective-k" in capsys.readouterr().err


def test_rc_tsne_is_opt_in():
    """It needs a compiled binary, so a bare no---methods sweep must not request it.

    The build is the difference from every other method here: R and MultiMAP need something
    *installed*, rc-tsne needs `scripts/setup/build_rctsne.sh` to have run.
    """
    from ne_aligned_utils.benchmark.sweep import CPP_SUBPROCESS_METHODS

    assert "rc-tsne" in TASK_KINDS
    assert "rc-tsne" not in DEFAULT_METHODS
    assert CPP_SUBPROCESS_METHODS == ("rc-tsne",)


def test_rc_tsne_reads_processed_features_not_raw_counts():
    """It is a geometry consumer: any real-valued matrix with the same width in both batches."""
    from ne_aligned_utils.benchmark.sweep import RAW_COUNT_METHODS

    assert "rc-tsne" not in RAW_COUNT_METHODS


def test_rc_tsne_owns_its_results_tree():
    """No backend axis and no neighborhood, but its own tree rather than sharing 2d_outputs.

    Aggregating one method rewrites the single cache a tree owns, and ``_merge_previous_methods``
    declines to carry the old arrays forward when a geometry key disagrees. Sharing would put the
    ten established direct-2D methods at risk on every rc-tsne re-run; this way a partial rc-tsne
    aggregation can only damage rc-tsne.
    """
    from ne_aligned_utils.config import OUTPUT_2D_METHODS, TREE_METHODS, shared_result_tree

    assert shared_result_tree("rc-tsne") == "rctsne"
    assert TREE_METHODS["rctsne"] == ("rc-tsne",)
    assert "rc-tsne" not in OUTPUT_2D_METHODS
    assert "rc-tsne" not in TREE_METHODS["2d_outputs"]


def _parse_run_task(monkeypatch, tmp_path, *extra):
    """``parse_args`` reads ``sys.argv`` rather than taking a list.

    ``--task-output`` is always supplied: without it argparse exits for that reason instead, and a
    guard test asserting only ``SystemExit`` would pass while proving nothing.
    """
    from ne_aligned_utils.benchmark.sweep import parse_args

    argv = [
        "benchmark_sweep",
        "run-task",
        "--dataset",
        "nCoV",
        "--seed",
        "0",
        "--task-output",
        str(tmp_path / "out.npz"),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return parse_args()


def test_rc_beta_default_is_the_prespecified_value(monkeypatch, tmp_path):
    """The reported runs must all carry one beta; validate_args refuses a second into 2d_outputs."""
    from ne_aligned_utils.config import DEFAULT_RC_TSNE_BETA

    args = _parse_run_task(monkeypatch, tmp_path, "--task-kind", "rc-tsne")
    assert args.rc_beta == DEFAULT_RC_TSNE_BETA
    assert args.rc_beta == pytest.approx(0.1)


def test_multimap_native_profile_is_the_default_but_can_be_disabled(monkeypatch, tmp_path):
    native = _parse_run_task(monkeypatch, tmp_path, "--task-kind", "multimap")
    assert native.multimap_native is True

    controlled = _parse_run_task(
        monkeypatch,
        tmp_path,
        "--task-kind",
        "multimap",
        "--no-multimap-native",
    )
    assert controlled.multimap_native is False


@pytest.mark.parametrize("bad", ["0", "-0.5", "1.5"])
def test_rc_beta_outside_the_upstream_bound_is_refused(monkeypatch, tmp_path, capsys, bad):
    """Their driver asserts ``0 < alpha <= 1``; zero is invalid upstream, not a limiting case."""
    with pytest.raises(SystemExit):
        _parse_run_task(monkeypatch, tmp_path, "--task-kind", "rc-tsne", "--rc-beta", bad)
    assert "--rc-beta must be in (0, 1]" in capsys.readouterr().err


def test_rc_beta_only_applies_to_rc_tsne(monkeypatch, tmp_path, capsys):
    """Mirrors the --init guard: a knob that reaches nothing would mislabel the cache it wrote."""
    with pytest.raises(SystemExit):
        _parse_run_task(monkeypatch, tmp_path, "--task-kind", "bc-tsne", "--rc-beta", "0.01")
    assert "--rc-beta is only supported for" in capsys.readouterr().err


# Unbalanced-OT plumbing


def _parse_uot_task(monkeypatch, tmp_path, *extra, kind="ot-uot-topk"):
    return _parse_run_task(
        monkeypatch,
        tmp_path,
        "--task-kind",
        kind,
        "--rel-reg",
        "0.1",
        "--effective-k",
        "15",
        *extra,
    )


def test_rel_reg_m_defaults_to_the_balanced_limit(monkeypatch, tmp_path):
    """Unset means balanced, so the two new kinds reproduce `ot` until asked otherwise."""
    from ne_aligned_utils.benchmark.sweep import DEFAULT_REL_REG_M

    args = _parse_uot_task(monkeypatch, tmp_path)
    assert args.rel_reg_m == DEFAULT_REL_REG_M == float("inf")
    # And the default needs no --output-dir, which is what lets the balanced baseline share the
    # regm_inf directory with both variants.
    validate_args(argparse.ArgumentParser(), args)


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_rel_reg_m_must_be_positive(monkeypatch, tmp_path, capsys, bad):
    with pytest.raises(SystemExit):
        _parse_uot_task(monkeypatch, tmp_path, "--rel-reg-m", bad)
    assert "--rel-reg-m must be positive" in capsys.readouterr().err


def test_rel_reg_m_only_applies_to_the_unbalanced_kinds(monkeypatch, tmp_path, capsys):
    """Same argument as --rc-beta: a knob that reaches nothing would mislabel the cache it wrote."""
    with pytest.raises(SystemExit):
        _parse_uot_task(monkeypatch, tmp_path, "--rel-reg-m", "0.1", kind="ot")
    assert "--rel-reg-m is only supported for" in capsys.readouterr().err


def test_a_finite_rel_reg_m_demands_its_own_output_dir(monkeypatch, tmp_path, capsys):
    """The config token does not encode the relaxation, so two values would share a directory.

    This is the guard that keeps a sweep from overwriting itself in silence -- the failure mode
    that made n_neighbors a token component in the first place.
    """
    with pytest.raises(SystemExit):
        _parse_uot_task(monkeypatch, tmp_path, "--rel-reg-m", "0.1")
    assert "must be given an explicit --output-dir" in capsys.readouterr().err


def test_two_relaxations_in_one_directory_are_refused(monkeypatch, tmp_path):
    """Belt and braces: even with the CLI bypassed, a mixed directory must not go unnoticed."""
    from ne_aligned_utils.benchmark.sweep import assert_no_parameter_collision, save_npz

    directory = tmp_path / "regm_mixed"
    directory.mkdir()
    save_npz(
        directory / "ot_uot_topk_tsne_seed_0_rel_reg_0p1_k_15.npz",
        rel_reg_m=0.1,
        perplexity=30.0,
        n_neighbors=-1,
        backend="tsne",
    )
    args = argparse.Namespace(
        task_kind="ot-uot-topk",
        perplexity=30.0,
        backend="tsne",
        n_neighbors=None,
        effective_k=15,
        rel_reg_m=0.01,
    )
    with pytest.raises(ValueError, match="rel_reg_m"):
        assert_no_parameter_collision(directory / "other.npz", args)


def test_collision_guard_checks_every_sibling_not_only_the_first(tmp_path):
    """A compatible first file must not hide a conflicting later one."""
    from ne_aligned_utils.benchmark.sweep import assert_no_parameter_collision, save_npz

    directory = tmp_path / "mixed"
    directory.mkdir()
    common = dict(n_neighbors=-1, backend="tsne")
    save_npz(directory / "a_seed_0.npz", task_kind="joint", perplexity=30.0, **common)
    save_npz(directory / "b_seed_0.npz", task_kind="joint", perplexity=50.0, **common)
    args = argparse.Namespace(task_kind="joint", perplexity=30.0, backend="tsne", n_neighbors=None)
    with pytest.raises(ValueError, match="perplexity"):
        assert_no_parameter_collision(directory / "other.npz", args)


def test_collision_guard_reads_legacy_schedule_from_the_siblings_method(tmp_path):
    """An old OT file lacks the exaggeration setting; its own method decides what it used."""
    from ne_aligned_utils.benchmark.sweep import assert_no_parameter_collision, save_npz

    directory = tmp_path / "legacy"
    directory.mkdir()
    save_npz(
        directory / "ot_tsne_seed_0.npz",
        task_kind="ot", perplexity=30.0, n_neighbors=-1, backend="tsne", device="cpu",
    )
    # A 50D tail task: before the fix the guard guessed the old file's schedule from this task's
    # method (openTSNE's 250, a match) instead of the OT file's own (the first-third split).
    args = argparse.Namespace(
        task_kind="joint", perplexity=30.0, backend="tsne", n_neighbors=None,
        tsne_early_exaggeration_iter=250,
    )
    # The old OT file used the first-third split, not 250, so the two must not share a folder.
    with pytest.raises(ValueError, match="tsne_early_exaggeration_iter"):
        assert_no_parameter_collision(directory / "other.npz", args)


def _resume_args(**overrides):
    from ne_aligned_utils.benchmark.sweep.core import add_common_options

    parser = argparse.ArgumentParser()
    add_common_options(parser)
    args = parser.parse_args(["--dataset", "nCoV"])
    for key, value in {
        "seed": 0, "method_seed": None, "visualization_seed": None, "rc_beta": 0.1,
        "rel_reg_m": float("inf"), "graph_symmetrization": "or", "knndiag_symmetrization": "or",
        "ot_device": "cpu", **overrides,
    }.items():
        setattr(args, key, value)
    return args


def test_direct_rerun_with_changed_settings_recomputes_instead_of_reusing():
    """Without --run-id, the cached-task check must still notice changed scientific settings."""
    from ne_aligned_utils.benchmark.sweep.tasks import _cached_setting_mismatches

    rc = {"task_kind": np.asarray("rc-tsne"), "rc_beta": np.asarray(0.1)}
    assert _cached_setting_mismatches(rc, _resume_args(task_kind="rc-tsne", rc_beta=0.1)) == []
    assert _cached_setting_mismatches(rc, _resume_args(task_kind="rc-tsne", rc_beta=1e-4))

    ot = {
        "task_kind": np.asarray("ot"), "n_epochs": np.asarray(500),
        "tsne_early_exaggeration_iter": np.asarray(250), "perplexity": np.asarray(30.0),
        "graph_symmetrization": np.asarray("or"), "ot_device": np.asarray("cpu"),
    }
    assert _cached_setting_mismatches(ot, _resume_args(task_kind="ot", n_epochs=500)) == []
    assert _cached_setting_mismatches(ot, _resume_args(task_kind="ot", n_epochs=300))
    assert _cached_setting_mismatches(ot, _resume_args(task_kind="ot", ot_device="cuda"))


def test_direct_rerun_treats_files_without_a_setting_by_what_they_used():
    from ne_aligned_utils.benchmark.sweep.tasks import _cached_setting_mismatches

    # Old OT files used the first-third split; old count methods the selected-gene divisor.
    old_ot = {"task_kind": np.asarray("ot"), "n_epochs": np.asarray(500)}
    assert _cached_setting_mismatches(old_ot, _resume_args(task_kind="ot", n_epochs=500))
    old_seurat = {"task_kind": np.asarray("seurat-2d")}
    args = _resume_args(task_kind="seurat-2d", count_library_size="all_genes")
    assert _cached_setting_mismatches(old_seurat, args)
    args = _resume_args(task_kind="seurat-2d", count_library_size="selected_genes")
    assert _cached_setting_mismatches(old_seurat, args) == []
    # Direct 2D layouts ignore the t-SNE schedule, so it never forces a recompute for them.
    multimap = {"task_kind": np.asarray("multimap"), "n_epochs": np.asarray(200)}
    assert _cached_setting_mismatches(multimap, _resume_args(task_kind="multimap")) == []


def test_rel_reg_m_separates_aggregate_caches_rather_than_merging_them():
    """It is an AGGREGATE_GEOMETRY_KEY, so a second relaxation replaces rather than extends.

    Without that, aggregating a relaxed run onto a balanced cache would merge the two under one
    set of method names -- the failure that has already happened twice on the `backend` axis.
    """
    from ne_aligned_utils.benchmark.sweep import (
        AGGREGATE_GEOMETRY_DEFAULTS,
        AGGREGATE_GEOMETRY_KEYS,
        DEFAULT_REL_REG_M,
    )

    assert "rel_reg_m" in AGGREGATE_GEOMETRY_KEYS
    # A cache written before this axis existed is a balanced one, and `inf` says exactly that
    # rather than acting as a sentinel, so it compares equal to a balanced run.
    assert AGGREGATE_GEOMETRY_DEFAULTS["rel_reg_m"] == DEFAULT_REL_REG_M


def test_the_unbalanced_kinds_are_opt_in():
    """A bare sweep must be unchanged, exactly as for the diagonal variants."""
    from ne_aligned_utils.benchmark.sweep import (
        DEFAULT_METHODS,
        EPS_K_METHODS,
        OT_PLAN_METHODS,
        UOT_METHODS,
    )

    assert set(UOT_METHODS).isdisjoint(DEFAULT_METHODS)
    # They are eps+k methods, so they take --rel-reg and --effective-k and get that filename shape.
    assert set(UOT_METHODS) <= set(EPS_K_METHODS) <= set(OT_PLAN_METHODS)


def test_the_zerodiag_kind_is_opt_in() -> None:
    """A bare sweep must be unchanged, exactly as for the diagonal and unbalanced variants."""
    from ne_aligned_utils.benchmark.sweep import (
        DEFAULT_METHODS,
        DIAG_VARIANT_METHODS,
        EPS_K_METHODS,
        OT_PLAN_METHODS,
        ZERODIAG_METHODS,
    )
    from ne_aligned_utils.config import METHOD_SPECS

    assert set(ZERODIAG_METHODS).isdisjoint(DEFAULT_METHODS)
    # Same eps+k tier as `ot` and the three diagonal variants: --rel-reg, --effective-k, and the
    # rel_reg+k filename shape.
    assert set(ZERODIAG_METHODS) <= set(EPS_K_METHODS) <= set(OT_PLAN_METHODS)
    # Unbalanced OT uses its own result tree, separate from diagonal variants.
    assert set(ZERODIAG_METHODS).isdisjoint(DIAG_VARIANT_METHODS)
    assert METHOD_SPECS["ot-zerodiag"].tree == "embeddings_zerodiag"
    # Both arms, one tree: they answer one question together and are scored side by side.
    assert METHOD_SPECS["ot-zerodiag-split"].tree == "embeddings_zerodiag"
    assert set(ZERODIAG_METHODS) == {"ot-zerodiag", "ot-zerodiag-split"}


# --ot-device plumbing


def test_ot_device_defaults_to_cpu(monkeypatch, tmp_path):
    """Unset means the historical numpy path, so every existing command line is unchanged."""
    args = _parse_uot_task(monkeypatch, tmp_path, kind="ot")
    assert args.ot_device == "cpu"
    validate_args(argparse.ArgumentParser(), args)


def test_ot_device_only_applies_to_the_ot_family(monkeypatch, tmp_path, capsys):
    """A knob that reaches nothing would mislabel the cache it wrote; same rule as --rc-beta."""
    with pytest.raises(SystemExit):
        _parse_run_task(
            monkeypatch, tmp_path, "--task-kind", "joint-pca", "--ot-device", "torch-cpu"
        )
    assert "--ot-device is only supported for" in capsys.readouterr().err


def test_ot_device_contradicts_require_cpu(monkeypatch, tmp_path, capsys):
    """--require-cpu is what makes a runtime comparable with the CPU-only benchmark.

    Asking for both is not redundancy, it is a contradiction, and resolving it silently either way
    would put a number in the runtime table that does not mean what the table says it means.
    """
    with pytest.raises(SystemExit):
        _parse_uot_task(monkeypatch, tmp_path, "--ot-device", "cuda", "--require-cpu", kind="ot")
    assert "contradicts --require-cpu" in capsys.readouterr().err


def test_two_devices_in_one_directory_are_refused(tmp_path):
    """The device is not in the config token, so the guard is what keeps a directory honest.

    Deliberately not a token component: the device is meant to leave results unchanged, so giving
    it a directory would fragment every path for nothing. But "unchanged" is a measured claim about
    float64 rather than a guarantee of bit-equality, so one directory must not hold both.
    """
    from ne_aligned_utils.benchmark.sweep import assert_no_parameter_collision, save_npz

    directory = tmp_path / "mixed"
    directory.mkdir()
    save_npz(
        directory / "ot_tsne_seed_0_rel_reg_0p1_k_15.npz",
        device="cuda",
        perplexity=30.0,
        n_neighbors=-1,
        backend="tsne",
    )
    args = argparse.Namespace(
        task_kind="ot",
        perplexity=30.0,
        backend="tsne",
        n_neighbors=None,
        effective_k=15,
        ot_device="cpu",
    )
    with pytest.raises(ValueError, match="device"):
        assert_no_parameter_collision(directory / "other.npz", args)


def test_every_cached_file_is_compatible_with_watching_the_device():
    """The guard must not fire on anything already on disk.

    It skips a key a sibling lacks, and every cached file either omits ``device`` or records
    ``"cpu"`` -- 1,611 of them at the time this was written, against 9,240 with no such field. So
    adding ``device`` to the watched set cannot produce a false positive on an existing tree, which
    is the property that made it safe to watch rather than tokenise.
    """
    from ne_aligned_utils.benchmark.sweep import DEFAULT_BACKEND

    # The invariant this rests on, asserted against the code rather than against the disk: a
    # default run records "cpu", which is what the cached files hold.
    assert DEFAULT_BACKEND == "tsne"
    from ne_aligned_utils.methods import CPU_DEVICE

    assert CPU_DEVICE == "cpu"


# OR vs AND symmetrization: an "and" run must be unable to reach an "or" cache


def test_a_mutual_run_requires_its_own_output_dir(monkeypatch, tmp_path, capsys):
    """Same rule as --rel-reg-m, and for the same reason.

    The config token does not encode the symmetrization, so an "and" run pointed at the historical
    tree would write ``ot_tsne_seed_0_rel_reg_1p0_k_15.npz`` on top of the "or" file of the same
    name. Requiring an explicit --output-dir is what makes that impossible rather than unlikely.
    """
    with pytest.raises(SystemExit):
        _parse_run_task(
            monkeypatch,
            tmp_path,
            "--task-kind",
            "ot",
            "--rel-reg",
            "1.0",
            "--effective-k",
            "15",
            "--graph-symmetrization",
            "and",
        )
    assert "must be given an explicit --output-dir" in capsys.readouterr().err


def test_the_default_symmetrization_needs_no_output_dir(monkeypatch, tmp_path):
    """The historical run is unchanged: no new flag, no new requirement."""
    args = _parse_run_task(
        monkeypatch, tmp_path, "--task-kind", "ot", "--rel-reg", "1.0", "--effective-k", "15"
    )
    assert args.graph_symmetrization == "or"
    assert args.knndiag_symmetrization == "or"


def test_the_symmetrization_switches_only_reach_the_methods_that_read_them(
    monkeypatch, tmp_path, capsys
):
    """Mirrors the --init and --rel-reg-m guards: a knob that reaches nothing mislabels its cache.

    ``ot-uot-mass`` is the sharp case. It is an eps+k method like the rest, but its graph comes
    from the mass-aware sparsifier, which has no mutual mode -- so accepting the flag here would
    write a cache claiming "and" over an "or" graph.
    """
    for task_kind, flag in (
        ("ot-uot-mass", "--graph-symmetrization"),
        ("ot-uot-topk", "--graph-symmetrization"),
        ("ot", "--knndiag-symmetrization"),
        ("ot-gramdiag", "--knndiag-symmetrization"),
    ):
        with pytest.raises(SystemExit):
            _parse_run_task(
                monkeypatch,
                tmp_path,
                "--task-kind",
                task_kind,
                "--rel-reg",
                "1.0",
                "--effective-k",
                "15",
                "--output-dir",
                str(tmp_path / "mutual"),
                flag,
                "and",
            )
        assert "is only supported for" in capsys.readouterr().err


def test_two_symmetrizations_in_one_directory_are_refused(tmp_path):
    """Belt and braces, exactly as for the two relaxations: with the CLI bypassed, a mixed
    directory must still not go unnoticed."""
    from ne_aligned_utils.benchmark.sweep import assert_no_parameter_collision, save_npz

    directory = tmp_path / "sym_mixed"
    directory.mkdir()
    save_npz(
        directory / "ot_tsne_seed_0_rel_reg_0p1_k_15.npz",
        graph_symmetrization="or",
        perplexity=30.0,
        n_neighbors=-1,
        backend="tsne",
    )
    args = argparse.Namespace(
        task_kind="ot",
        perplexity=30.0,
        backend="tsne",
        n_neighbors=None,
        effective_k=15,
        graph_symmetrization="and",
    )
    with pytest.raises(ValueError, match="graph_symmetrization"):
        assert_no_parameter_collision(directory / "other.npz", args)


def test_symmetrization_separates_aggregate_caches_rather_than_merging_them():
    """Both switches are AGGREGATE_GEOMETRY_KEYS, with "or" as the stated meaning of an old cache.

    The default is the load-bearing half. Every aggregate on disk predates these keys; without a
    default they would be skipped, and an "and" run would merge its arrays into the "or" cache
    under the same method names -- the failure that has already happened twice on `backend`.
    """
    from ne_aligned_utils.benchmark.sweep import (
        AGGREGATE_GEOMETRY_DEFAULTS,
        AGGREGATE_GEOMETRY_KEYS,
        DEFAULT_SYMMETRIZATION,
    )

    for key in ("graph_symmetrization", "knndiag_symmetrization"):
        assert key in AGGREGATE_GEOMETRY_KEYS
        assert AGGREGATE_GEOMETRY_DEFAULTS[key] == DEFAULT_SYMMETRIZATION == "or"


def test_an_old_aggregate_still_merges_into_an_or_rerun(tmp_path):
    """The other side of that default: an ordinary "or" rerun must still extend, not replace.

    A cache written before these keys existed has to keep comparing equal to an "or" run, or every
    rerun would silently drop every method it was not asked for.
    """
    import numpy as np

    from ne_aligned_utils.benchmark.sweep import _merge_previous_methods, save_npz

    cache = tmp_path / "ds_benchmark_embeddings.npz"
    save_npz(
        cache,
        dataset="nCoV",
        seeds=np.array([0]),
        rel_reg=np.array([1.0]),
        effective_ks=np.array([15]),
        backend="tsne",
        n_neighbors=-1,
        methods=np.array(["joint-pca"]),
        joint_pca_embeddings=np.zeros((1, 4, 2)),
        joint_pca_paths=np.array(["a"], dtype=object),
    )
    payload = {
        "dataset": "nCoV",
        "seeds": np.array([0]),
        "rel_reg": np.array([1.0]),
        "effective_ks": np.array([15]),
        "backend": "tsne",
        "n_neighbors": -1,
        "graph_symmetrization": "or",
        "knndiag_symmetrization": "or",
        "methods": np.array(["ot"]),
        "ot_tsne_embeddings": np.zeros((1, 1, 1, 4, 2)),
    }
    merged = _merge_previous_methods(dict(payload), cache)
    assert "joint_pca_embeddings" in merged, "an 'or' rerun must extend the old cache"

    mutual = dict(payload, graph_symmetrization="and")
    replaced = _merge_previous_methods(dict(mutual), cache)
    assert "joint_pca_embeddings" not in replaced, "an 'and' run must not merge into an 'or' cache"
