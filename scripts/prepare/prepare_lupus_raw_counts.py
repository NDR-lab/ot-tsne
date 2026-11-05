"""Prepare aligned lupus raw counts for benchmark methods requiring count inputs."""

from __future__ import annotations

import argparse
import gzip
import tarfile
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

from ne_aligned_utils.config import (  # noqa: E402
    ROOT,
)

GEO_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl"
RAW_TAR_NAME = "GSE96583_RAW.tar"
BATCH2_GENES_NAME = "GSE96583_batch2.genes.tsv.gz"
BATCH2_CELL_METADATA_NAME = "GSE96583_batch2.total.tsne.df.tsv.gz"
CTRL_MATRIX_NAME = "GSM2560248_2.1.mtx.gz"
STIM_MATRIX_NAME = "GSM2560249_2.2.mtx.gz"
CTRL_BARCODES_NAME = "GSM2560248_barcodes.tsv.gz"
STIM_BARCODES_NAME = "GSM2560249_barcodes.tsv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare lupus/IFNB raw counts from GSE96583 batch2 files."
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "lupus" / "raw")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "lupus" / "raw" / "raw_counts.h5ad"
    )
    parser.add_argument(
        "--current-metadata",
        type=Path,
        default=ROOT / "data" / "lupus" / "cells_metadata.csv",
        help="Optional current benchmark metadata to join by cell name.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download_if_missing(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def ensure_inputs(raw_dir: Path) -> tuple[Path, Path, Path]:
    raw_tar = raw_dir / RAW_TAR_NAME
    genes_path = raw_dir / BATCH2_GENES_NAME
    cell_metadata_path = raw_dir / BATCH2_CELL_METADATA_NAME
    download_if_missing(raw_tar, f"{GEO_BASE_URL}/{RAW_TAR_NAME}")
    download_if_missing(genes_path, f"{GEO_BASE_URL}/{BATCH2_GENES_NAME}")
    download_if_missing(cell_metadata_path, f"{GEO_BASE_URL}/{BATCH2_CELL_METADATA_NAME}")
    return raw_tar, genes_path, cell_metadata_path


def read_tar_gzip_lines(tar: tarfile.TarFile, member_name: str) -> list[str]:
    member = tar.extractfile(member_name)
    if member is None:
        raise FileNotFoundError(member_name)
    with gzip.open(member, "rt") as handle:
        return [line.strip() for line in handle if line.strip()]


def read_tar_matrix(tar: tarfile.TarFile, member_name: str) -> sp.csc_matrix:
    member = tar.extractfile(member_name)
    if member is None:
        raise FileNotFoundError(member_name)
    with gzip.open(member, "rb") as handle:
        matrix = scipy.io.mmread(handle)
    return sp.csc_matrix(matrix)


def seurat_style_batch2_names(
    ctrl_barcodes: list[str], stim_barcodes: list[str]
) -> tuple[list[str], list[str]]:
    ctrl_names = [barcode.replace("-", ".") for barcode in ctrl_barcodes]
    ctrl_seen = set(ctrl_names)
    stim_names = []
    for barcode in stim_barcodes:
        name = barcode.replace("-", ".")
        if name in ctrl_seen:
            name = f"{name}1"
        stim_names.append(name)
    return ctrl_names, stim_names


def make_unique(values: pd.Series) -> list[str]:
    values = values.astype(str).fillna("").replace({"": np.nan})
    fallback = pd.Series([f"gene_{idx}" for idx in range(values.shape[0])], index=values.index)
    values = values.fillna(fallback)
    counts: dict[str, int] = {}
    unique_values = []
    for value in values.tolist():
        number = counts.get(value, 0)
        unique_values.append(value if number == 0 else f"{value}-{number + 1}")
        counts[value] = number + 1
    return unique_values


def sanitize_h5ad_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
            frame[column] = values.where(values.notna(), "").astype(str)
    return frame


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        print(f"{args.output} already exists; use --force to overwrite.")
        return

    raw_tar, genes_path, cell_metadata_path = ensure_inputs(args.raw_dir)
    genes = pd.read_csv(
        genes_path,
        sep="\t",
        header=None,
        names=["gene_id", "gene_symbol"],
        compression="gzip",
    )

    with tarfile.open(raw_tar) as tar:
        ctrl_counts = read_tar_matrix(tar, CTRL_MATRIX_NAME)
        stim_counts = read_tar_matrix(tar, STIM_MATRIX_NAME)
        ctrl_barcodes = read_tar_gzip_lines(tar, CTRL_BARCODES_NAME)
        stim_barcodes = read_tar_gzip_lines(tar, STIM_BARCODES_NAME)

    if ctrl_counts.shape[0] != stim_counts.shape[0]:
        raise ValueError("CTRL and STIM matrices have different gene counts.")
    if ctrl_counts.shape[0] != genes.shape[0]:
        raise ValueError("Gene metadata row count does not match matrix gene count.")
    if ctrl_counts.shape[1] != len(ctrl_barcodes) or stim_counts.shape[1] != len(stim_barcodes):
        raise ValueError("Barcode counts do not match matrix cell counts.")

    ctrl_names, stim_names = seurat_style_batch2_names(ctrl_barcodes, stim_barcodes)
    counts = sp.hstack([ctrl_counts, stim_counts], format="csc").T.tocsr()
    obs = pd.DataFrame(
        {
            "stim": ["CTRL"] * len(ctrl_names) + ["STIM"] * len(stim_names),
            "method": ["data1"] * len(ctrl_names) + ["data2"] * len(stim_names),
        },
        index=pd.Index([*ctrl_names, *stim_names], name="cell"),
    )
    geo_cell_metadata = pd.read_csv(cell_metadata_path, sep="\t", compression="gzip")
    if "cell" in geo_cell_metadata.columns:
        geo_cell_metadata = geo_cell_metadata.set_index("cell")
        obs = obs.join(geo_cell_metadata, how="left", rsuffix="_geo")

    if args.current_metadata.exists():
        current = pd.read_csv(args.current_metadata, index_col=0)
        obs = obs.join(current, how="left", rsuffix="_benchmark")

    var_names = make_unique(genes["gene_symbol"])
    obs = sanitize_h5ad_frame(obs)
    var = sanitize_h5ad_frame(genes.copy())
    var.index = pd.Index(var_names, name="gene")
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)
    print(f"Saved lupus raw-count AnnData to {args.output}")
    print(f"Shape: {adata.shape}")


if __name__ == "__main__":
    main()
