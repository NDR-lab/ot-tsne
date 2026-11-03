"""Build a canonical count-based benchmark dataset."""

from __future__ import annotations

import argparse
import os
import pickle
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from ne_aligned_utils.config import BENCHMARK_DATASETS, DATA_DIR
from ne_aligned_utils.data.preprocessing import (
    PREPROCESSING_RANDOM_SEED,
    check_saved,
    prepare,
    write_preprocessing_manifest,
)

DEFAULT_BENCHMARK_MANIFEST = Path(__file__).parents[2] / "configs" / "benchmark.toml"


@dataclass(frozen=True)
class BuildSpec:
    loader: Callable[[Path], ad.AnnData]
    batch_names: tuple[str, str]
    source_files: tuple[str, ...]
    raw_dataset: str | None = None
    extra_label_cols: tuple[str, ...] = ()


def _categorize_contract(adata: ad.AnnData) -> ad.AnnData:
    """Finish the common two-batch/count contract after dataset-specific loading."""
    for column in ("method", "cell_type"):
        adata.obs[column] = pd.Categorical(adata.obs[column].astype(str))
    adata.layers["counts"] = adata.X.copy()
    return adata


def load_human_brain(raw_dir: Path) -> ad.AnnData:
    """Load and align human-brain replicates 4 and 1."""
    input_path = raw_dir / "replicates.h5ad"
    metadata_path = raw_dir / "cell_metadata.csv"
    adata = sc.read_h5ad(input_path)
    metadata = pd.read_csv(metadata_path, index_col=0)
    metadata.index = metadata.index.astype(str)
    cells = pd.Index(adata.obs_names.astype(str))
    if not metadata.index.is_unique:
        raise ValueError(f"{metadata_path} contains duplicate cell IDs.")
    missing, extra = cells.difference(metadata.index), metadata.index.difference(cells)
    if len(missing) or len(extra):
        raise ValueError(
            f"Human-brain cell IDs differ: {len(missing)} missing and {len(extra)} extra."
        )
    metadata = metadata.loc[cells]
    if "cell_types" not in metadata or "replicate_id" not in adata.obs:
        raise KeyError("Human-brain inputs require cell_types and replicate_id columns.")
    if "replicate_id" in metadata and not np.array_equal(
        metadata["replicate_id"].astype(str).to_numpy(),
        adata.obs["replicate_id"].astype(str).to_numpy(),
    ):
        raise ValueError("replicate_id differs between the human-brain input files.")

    selected = {"4": "data1", "1": "data2"}
    keep = adata.obs["replicate_id"].astype(str).isin(selected)
    adata = adata[keep.to_numpy()].copy()
    metadata = metadata.loc[adata.obs_names.astype(str)]
    adata.obs["method"] = adata.obs["replicate_id"].astype(str).map(selected)
    adata.obs["cell_type"] = metadata["cell_types"].astype(str).to_numpy()
    return _categorize_contract(adata)


def _cell_type_column(columns: pd.Index) -> str | None:
    normalized = {
        str(column).lower().replace("_", "").replace(".", "").replace(" ", ""): str(column)
        for column in columns
    }
    return next((column for key, column in normalized.items() if "celltype" in key), None)


def _read_tran_cell_types(raw_dir: Path, prefix: str, cells: pd.Index) -> pd.Series:
    path = raw_dir / f"{prefix}_celltype.txt.gz"
    table = pd.read_csv(path, sep="\t", index_col=0)
    column = _cell_type_column(table.columns)
    if column is None:
        table = pd.read_csv(path, sep="\t")
        column = _cell_type_column(table.columns)
    if column is None:
        raise KeyError(f"{path} has no cell-type column.")
    table.index = table.index.astype(str)
    if table.index.is_unique and table.index.isin(cells).all():
        table = table.reindex(cells)
    elif len(table) == len(cells):
        table = table.set_axis(cells, axis=0)
    else:
        raise ValueError(f"{path} has {len(table)} rows for {len(cells)} expression cells.")
    if table[column].isna().any():
        raise ValueError(f"{path} has cells without a cell type.")
    return table[column].astype(str)


def _read_tran_batch(raw_dir: Path, prefix: str, method: str) -> ad.AnnData:
    """Load and align one Tran et al. expression and label batch."""
    path = raw_dir / f"{prefix}_exprs.txt.gz"
    columns = pd.read_csv(path, sep="\t", index_col=0, nrows=0).columns
    expression = pd.read_csv(
        path,
        sep="\t",
        index_col=0,
        dtype={column: np.float32 for column in columns},
    )
    cells = pd.Index(expression.columns.astype(str), name="cell_id")
    genes = pd.Index(expression.index.astype(str), name="gene")
    if not cells.is_unique:
        raise ValueError(f"{path} contains duplicate cell IDs.")
    cell_types = _read_tran_cell_types(raw_dir, prefix, cells)
    obs = pd.DataFrame(
        {"batch": prefix, "method": method, "cell_type": cell_types.to_numpy()},
        index=cells,
    )
    result = ad.AnnData(
        X=sp.csr_matrix(expression.to_numpy(dtype=np.float32).T),
        obs=obs,
        var=pd.DataFrame(index=genes),
    )
    if not result.var_names.is_unique:
        result.var_names_make_unique()
    return _categorize_contract(result)


def _load_tran_two_batch(raw_dir: Path) -> ad.AnnData:
    batch1 = _read_tran_batch(raw_dir, "b1", "data1")
    batch2 = _read_tran_batch(raw_dir, "b2", "data2")
    overlap = batch1.obs_names.intersection(batch2.obs_names)
    combined = sc.concat(
        {"b1": batch1, "b2": batch2},
        axis=0,
        join="inner",
        label="batch_concat",
        index_unique=None if len(overlap) == 0 else "-",
    )
    return _categorize_contract(combined)


def load_pbmc(raw_dir: Path) -> ad.AnnData:
    return _load_tran_two_batch(raw_dir)


def load_hematopoietic(raw_dir: Path) -> ad.AnnData:
    return _load_tran_two_batch(raw_dir)


def load_hematopoietic_shared(raw_dir: Path) -> ad.AnnData:
    adata = _load_tran_two_batch(raw_dir)
    keep_types = {"CMP", "GMP", "MEP"}
    observed = set(adata.obs["cell_type"].astype(str))
    if not keep_types.issubset(observed):
        raise ValueError(f"Missing shared hematopoietic labels: {sorted(keep_types - observed)}")
    return _categorize_contract(adata[adata.obs["cell_type"].astype(str).isin(keep_types)].copy())


def load_mouse_atlas(raw_dir: Path) -> ad.AnnData:
    """Load Seq-Well and 10x batches with harmonized cell-type labels."""
    expression_path = raw_dir / "filtered_total_batch1_seqwell_batch2_10x.txt.gz"
    metadata_path = raw_dir / "filtered_total_sample_ext_organ_celltype_batch.txt.gz"
    metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)
    expression = pd.read_csv(expression_path, sep="\t", index_col=0)
    cells = pd.Index(expression.columns.astype(str), name="cell_id")
    obs = metadata.set_index(metadata.index.astype(str)).reindex(cells)
    if obs.isna().all(axis=1).any():
        raise ValueError("Mouse-atlas count matrix contains cells without metadata.")
    required = {"batchlb", "ct", "organ"}
    missing = required.difference(obs.columns)
    if missing:
        raise KeyError(f"{metadata_path} is missing columns: {sorted(missing)}")
    batch_mapping = {"Batch1": "data1", "Batch2": "data2"}
    if not set(obs["batchlb"].astype(str)).issubset(batch_mapping):
        raise ValueError("Mouse-atlas batchlb contains an unexpected protocol batch.")
    obs["method"] = obs["batchlb"].astype(str).map(batch_mapping)
    obs["cell_type"] = obs["ct"].astype(str)
    adata = ad.AnnData(
        X=sp.csr_matrix(expression.to_numpy(dtype=np.float32).T),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(expression.index.astype(str), name="gene")),
    )
    return _categorize_contract(adata)


def load_lupus(raw_dir: Path) -> ad.AnnData:
    input_path = raw_dir / "raw_counts.h5ad"
    adata = sc.read_h5ad(input_path)
    required = {"method", "cell_type"}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise KeyError(f"{input_path} is missing obs columns: {sorted(missing)}")

    # The source contains all GEO cells; 13,999 have the benchmark cell-type annotation.
    annotated = adata.obs["cell_type"].notna() & adata.obs["cell_type"].astype(str).str.len().gt(0)
    adata = adata[annotated.to_numpy()].copy()
    methods = set(adata.obs["method"].astype(str))
    if methods != {"data1", "data2"}:
        raise ValueError(f"Expected method values data1/data2, found {sorted(methods)}.")

    adata.obs["method"] = pd.Categorical(
        adata.obs["method"].astype(str), categories=["data1", "data2"]
    )
    adata.obs["cell_type"] = pd.Categorical(adata.obs["cell_type"].astype(str))
    adata.layers["counts"] = adata.X.copy()
    return adata


def load_yao(raw_dir: Path) -> ad.AnnData:
    pickle_path = raw_dir / "ent_L23.pkl"
    metadata_path = raw_dir / "ent_l2l3_meta.csv"
    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted, locally supplied research data

    required = {"counts", "genes", "samples_names"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"{pickle_path} is missing keys: {sorted(missing)}")

    counts = sp.csr_matrix(payload["counts"])
    genes = pd.Index(np.asarray(payload["genes"], dtype=str), name="gene")
    sample_names = pd.Index(np.asarray(payload["samples_names"], dtype=str), name="sample_name")
    if counts.shape != (len(sample_names), len(genes)):
        raise ValueError(
            f"counts shape {counts.shape} does not match {len(sample_names)} cells and "
            f"{len(genes)} genes"
        )
    if not genes.is_unique or not sample_names.is_unique:
        raise ValueError("Yao gene and sample names must be unique.")
    if counts.data.size and (
        counts.data.min() < 0 or not np.array_equal(counts.data, np.round(counts.data))
    ):
        raise ValueError("Yao counts must be non-negative integers.")

    metadata = pd.read_csv(metadata_path, index_col="sample_name")
    metadata.index = metadata.index.astype(str)
    missing_cells = sample_names.difference(metadata.index)
    extra_cells = metadata.index.difference(sample_names)
    if len(missing_cells) or len(extra_cells):
        raise ValueError(
            f"Pickle/metadata cell IDs differ: {len(missing_cells)} absent from metadata and "
            f"{len(extra_cells)} absent from the pickle."
        )
    metadata = metadata.loc[sample_names].copy()

    observed_batches = set(metadata["donor_sex_label"].astype(str))
    if observed_batches != {"F", "M"}:
        raise ValueError(f"Expected donor-sex batches F/M, found {sorted(observed_batches)}.")
    metadata["method"] = pd.Categorical(
        metadata["donor_sex_label"].astype(str).map({"F": "data1", "M": "data2"}),
        categories=["data1", "data2"],
    )
    metadata["cell_type"] = pd.Categorical(metadata["cluster_label"].astype(str))

    adata = ad.AnnData(X=counts, obs=metadata, var=pd.DataFrame(index=genes))
    adata.layers["counts"] = counts.copy()
    return adata


def load_ncov(raw_dir: Path) -> ad.AnnData:
    """Load the two nCoV patient-5 time points."""
    input_path = raw_dir / "Final_nCoV_0716_upload.h5ad"
    adata = sc.read_h5ad(input_path)
    required = {"batch", "cell_type"}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise KeyError(f"{input_path} is missing obs columns: {sorted(missing)}")

    selected = ("COV-5-D13", "COV-5-D1")
    available = set(adata.obs["batch"].astype(str))
    if not set(selected).issubset(available):
        raise ValueError(f"{input_path} does not contain both historical nCoV batches {selected}.")
    adata = adata[adata.obs["batch"].astype(str).isin(selected)].copy()
    adata.obs["method"] = pd.Categorical(
        adata.obs["batch"].astype(str).map({selected[0]: "data1", selected[1]: "data2"}),
        categories=["data1", "data2"],
    )
    adata.obs["cell_type"] = pd.Categorical(adata.obs["cell_type"].astype(str))
    # Use X rather than logcounts because feature selection requires integer counts.
    adata.layers["counts"] = adata.X.copy()
    return adata


BUILD_SPECS = {
    "human_brain": BuildSpec(
        load_human_brain,
        ("replicate 4", "replicate 1"),
        ("replicates.h5ad", "cell_metadata.csv"),
    ),
    "lupus": BuildSpec(load_lupus, ("CTRL", "STIM"), ("raw_counts.h5ad",)),
    "nCoV": BuildSpec(
        load_ncov,
        ("COV-5-D13", "COV-5-D1"),
        ("Final_nCoV_0716_upload.h5ad",),
    ),
    "PBMC": BuildSpec(
        load_pbmc,
        ("b1", "b2"),
        ("b1_exprs.txt.gz", "b1_celltype.txt.gz", "b2_exprs.txt.gz", "b2_celltype.txt.gz"),
    ),
    "mouse_atlas": BuildSpec(
        load_mouse_atlas,
        ("seqwell", "10x"),
        (
            "filtered_total_batch1_seqwell_batch2_10x.txt.gz",
            "filtered_total_sample_ext_organ_celltype_batch.txt.gz",
        ),
        extra_label_cols=("organ",),
    ),
    "hematopoietic": BuildSpec(
        load_hematopoietic,
        ("b1", "b2"),
        ("b1_exprs.txt.gz", "b1_celltype.txt.gz", "b2_exprs.txt.gz", "b2_celltype.txt.gz"),
    ),
    "hematopoietic_shared": BuildSpec(
        load_hematopoietic_shared,
        ("b1", "b2"),
        ("b1_exprs.txt.gz", "b1_celltype.txt.gz", "b2_exprs.txt.gz", "b2_celltype.txt.gz"),
        raw_dataset="hematopoietic",
    ),
    "yao_10x_ent_l2l3": BuildSpec(
        load_yao,
        ("ent_l2l3-F", "ent_l2l3-M"),
        ("ent_L23.pkl", "ent_l2l3_meta.csv"),
    ),
}


def validate_dataset_declarations(manifest: dict) -> None:
    """Validate configured datasets against the available adapters and registry."""
    declared = set(manifest["datasets"])
    adapters = set(BUILD_SPECS)
    registry = set(BENCHMARK_DATASETS)
    if not declared <= adapters or not declared <= registry:
        raise ValueError(
            "Dataset declarations disagree: "
            f"config_only={sorted(declared - adapters)}, "
            f"adapter_only={sorted(adapters - declared)}, "
            f"registry_difference={sorted(declared ^ registry)}"
        )
    for dataset in manifest["datasets"]:
        spec = BUILD_SPECS[dataset]
        configured = manifest["datasets"][dataset]
        if configured.get("raw_adapter") != spec.loader.__name__:
            raise ValueError(
                f"{dataset}: raw_adapter={configured.get('raw_adapter')!r}, "
                f"expected {spec.loader.__name__!r}"
            )
        configured_names = tuple(Path(value).name for value in configured.get("source_files", ()))
        if configured_names != spec.source_files:
            raise ValueError(
                f"{dataset}: configured source files {configured_names!r} do not match "
                f"adapter sources {spec.source_files!r}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=tuple(BUILD_SPECS))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("NE_ALIGNED_RAW_ROOT", DATA_DIR)),
        help="raw-data parent; defaults to NE_ALIGNED_RAW_ROOT, then repository data/",
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--benchmark-manifest", type=Path, default=DEFAULT_BENCHMARK_MANIFEST)
    parser.add_argument("--n-top-genes", type=int, default=None)
    parser.add_argument("--n-pca", type=int, default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate that this adapter's required raw source files exist, then exit",
    )
    parser.add_argument(
        "--allow-canonical-write",
        action="store_true",
        help="allow a non-default --output-dir below data/",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with args.benchmark_manifest.open("rb") as handle:
        benchmark_manifest = tomllib.load(handle)
    validate_dataset_declarations(benchmark_manifest)
    policy = benchmark_manifest["preprocessing"]
    supported = {
        "min_genes_per_cell": 10,
        "min_cells_per_gene": 3,
        "feature_selection": "scanpy.seurat_v3_batch_aware_raw_counts",
        "normalization_target": 10000.0,
        "transformation": "log1p",
    }
    mismatches = {
        key: (policy.get(key), value)
        for key, value in supported.items()
        if policy.get(key) != value
    }
    if mismatches:
        raise ValueError(f"The benchmark manifest requests unsupported preprocessing: {mismatches}")
    n_top_genes = int(policy["selected_genes"] if args.n_top_genes is None else args.n_top_genes)
    n_pca = int(policy["joint_pca_components"] if args.n_pca is None else args.n_pca)
    spec = BUILD_SPECS[args.dataset]
    data_root = args.data_root.expanduser().resolve()
    raw_dir = args.raw_dir or data_root / (spec.raw_dataset or args.dataset) / "raw"
    prepared_root = Path(os.environ.get("NE_ALIGNED_DATA_ROOT", data_root / "benchmark"))
    default_output_dir = prepared_root.expanduser().resolve() / args.dataset
    output_dir = args.output_dir or default_output_dir
    missing_sources = [
        raw_dir / name for name in spec.source_files if not (raw_dir / name).is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "Missing raw source file(s): " + ", ".join(str(path) for path in missing_sources)
        )
    if args.check_only:
        print(f"[{args.dataset}] raw sources available under {raw_dir}")
        return
    adata = spec.loader(raw_dir)

    out = prepare(
        adata,
        output_dir,
        args.dataset,
        n_top_genes=n_top_genes,
        n_pca=n_pca,
        random_seed=PREPROCESSING_RANDOM_SEED,
        batch_names=spec.batch_names,
        extra_label_cols=spec.extra_label_cols,
        allow_canonical_write=(
            args.allow_canonical_write or output_dir.resolve() == default_output_dir.resolve()
        ),
    )
    check_saved(out)
    manifest = write_preprocessing_manifest(
        out,
        source_paths=[raw_dir / name for name in spec.source_files],
        loader=spec.loader.__name__,
        preprocessing_seed=PREPROCESSING_RANDOM_SEED,
        n_top_genes=n_top_genes,
        n_pca=n_pca,
    )
    print(f"[{args.dataset}] preprocessing_hash={manifest['preprocessing_hash']}")


if __name__ == "__main__":
    main()
