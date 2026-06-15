from argparse import Namespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from ne_aligned_utils.data import load_lupus
from ne_aligned_utils.embeddings_for_benchmarks import (
    DEFAULT_METHODS,
    TASK_KINDS,
    aggregate,
    load_benchmark_data,
    load_named_dataset,
    resolve_task_output_path,
    save_npz,
    submit_slurm,
)


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

    loaded_x0, loaded_x1, cell_type_names, loaded_labels0, loaded_labels1, dataset_names = load_lupus(
        str(tmp_path), return_meta=True
    )
    np.testing.assert_allclose(loaded_x0, x0)
    np.testing.assert_allclose(loaded_x1, x1)
    np.testing.assert_array_equal(loaded_labels0, labels0)
    np.testing.assert_array_equal(loaded_labels1, labels1)
    assert cell_type_names.tolist() == ["B", "T"]
    assert dataset_names.tolist() == ["ctrl", "case"]


def test_scvi_methods_are_opt_in():
    assert "scvi-2d" in TASK_KINDS
    assert "scvi-50-tsne" in TASK_KINDS
    assert "scvi-2d" not in DEFAULT_METHODS
    assert "scvi-50-tsne" not in DEFAULT_METHODS


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


def test_load_boris_raw_counts_subsets_full_raw_object(tmp_path):
    dataset_dir = tmp_path / "human_brain"
    dataset_dir.mkdir()
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
    ).write_h5ad(dataset_dir / "replicates.h5ad")
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
    ).to_csv(dataset_dir / "cell_metadata.csv")

    data = load_benchmark_data(
        "Boris_sc",
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


def test_load_yao_raw_counts_maps_ent_subset_to_full_counts(tmp_path):
    yao_new = tmp_path / "yao_new"
    yao_old = tmp_path / "yao_10x"
    yao_new.mkdir()
    yao_old.mkdir()
    counts = sp.csr_matrix(
        [
            [9, 9, 9],
            [1, 0, 2],
            [0, 3, 0],
            [4, 0, 1],
            [0, 2, 2],
        ],
        dtype=float,
    )
    with (yao_new / "ent_L23.pkl").open("wb") as handle:
        import pickle

        pickle.dump(
            {
                "counts": counts,
                "genes": np.array(["g0", "g1", "g2"], dtype=object),
                "clusters": np.arange(5),
                "clusterColors": np.array(["#000000"], dtype=object),
                "clusterNames": np.array(["cluster"], dtype=object),
            },
            handle,
        )
    pd.DataFrame({"sample_name": ["unused", "a", "b", "c", "d"]}).to_csv(
        yao_old / "10x_metadata.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "sample_name": ["a", "b", "c", "d"],
            "donor_sex_label": ["F", "F", "M", "M"],
            "cluster_label": ["Lamp5", "Vip", "Lamp5", "Vip"],
        }
    ).to_csv(yao_new / "ent_l2l3_meta.csv", index=False)

    data = load_benchmark_data(
        "yao_10x_ent_l2l3",
        data_root=tmp_path,
        seed=0,
        max_cells_per_batch=None,
        input_kind="raw_counts",
    )

    np.testing.assert_allclose(data.x0, counts[[1, 2]].toarray())
    np.testing.assert_allclose(data.x1, counts[[3, 4]].toarray())
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
        methods=["scvi-2d", "scvi-50-tsne"],
        eps_times=[1.0],
        effective_ks=[15],
        embedding_dim=2,
        n_epochs=12,
        tsne_pca_components=50,
        perplexity=2.0,
        max_cells_per_batch=None,
        max_cells_batch0=None,
        max_cells_batch1=None,
    )
    task_root = args.output_dir / "n_per_batch_all__q_2__epochs_12__tsne_pca_50"
    embedding = np.arange(10, dtype=float).reshape(5, 2)
    for method in args.methods:
        save_npz(
            resolve_task_output_path(task_root, task_kind=method, seed=0),
            embedding=embedding,
        )

    aggregate(args)

    output_path = task_root / "lupus_benchmark_embeddings.npz"
    with np.load(output_path, allow_pickle=False) as cache:
        assert cache["scvi_2d_embeddings"].shape == (1, 5, 2)
        assert cache["scvi_50_tsne_embeddings"].shape == (1, 5, 2)
        assert cache["scvi_2d_paths"].shape == (1,)
        assert cache["scvi_50_tsne_paths"].shape == (1,)


def test_submit_slurm_dry_run_scvi_methods_have_no_ot_args(tmp_path, capsys):
    args = Namespace(
        dataset="lupus",
        data_root=tmp_path / "data",
        output_dir=tmp_path / "embeddings",
        max_cells_per_batch=None,
        max_cells_batch0=None,
        max_cells_batch1=None,
        embedding_dim=2,
        n_epochs=200,
        tsne_pca_components=50,
        perplexity=30.0,
        n_jobs=1,
        tol=1e-12,
        max_iter=int(1e4),
        eps_times=[1.0],
        effective_ks=[15],
        methods=["scvi-2d", "scvi-50-tsne"],
        n_seeds=2,
        python_executable="python",
        overwrite_manifest=True,
        force=False,
        dry_run=True,
        slurm_job_prefix="embed-bench",
        slurm_time=None,
        slurm_partition=None,
        slurm_account=None,
        slurm_cpus_per_task=None,
        slurm_mem=None,
        slurm_use_gpu=False,
        slurm_extra_arg=[],
    )
    submit_slurm(args)
    stdout = capsys.readouterr().out

    run_task_lines = [line for line in stdout.splitlines() if " run-task " in line]
    assert len(run_task_lines) == 4
    assert all("--eps-times" not in line for line in run_task_lines)
    assert all("--effective-k" not in line for line in run_task_lines)
    aggregate_lines = [line for line in stdout.splitlines() if " aggregate " in line]
    assert len(aggregate_lines) == 1
    assert "--eps-times" not in aggregate_lines[0]
    assert "--effective-k" not in aggregate_lines[0]
