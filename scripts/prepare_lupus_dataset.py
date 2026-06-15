from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]


def obs_column(frame: pd.DataFrame, column: str, purpose: str) -> pd.Series:
    if column not in frame.columns:
        available = ", ".join(str(name) for name in frame.columns)
        raise KeyError(f"Missing {purpose} column {column!r}. Available columns: {available}")
    return frame[column]


def choose_batches(values: pd.Series, batch0: str | None, batch1: str | None) -> tuple[str, str]:
    observed = [str(value) for value in pd.unique(values.dropna().astype(str))]
    if batch0 is not None or batch1 is not None:
        if batch0 is None or batch1 is None:
            raise ValueError("Provide both --batch0 and --batch1, or neither.")
        missing = [value for value in (batch0, batch1) if value not in observed]
        if missing:
            raise ValueError(f"Batch value(s) not found: {missing}. Available values: {observed}")
        return str(batch0), str(batch1)

    if len(observed) < 2:
        raise ValueError(f"Need at least two batches in the batch column; found {observed}.")
    return observed[0], observed[1]


def matrix_to_dense_float32(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def keep_frequent_cell_types(adata: sc.AnnData, min_cells_per_type: int) -> sc.AnnData:
    if min_cells_per_type <= 0 or "cell_type" not in adata.obs:
        return adata

    counts = adata.obs.groupby(["method", "cell_type"], observed=True).size()
    keep_types = []
    for cell_type in pd.unique(adata.obs["cell_type"].astype(str)):
        if all(counts.get((method, cell_type), 0) > min_cells_per_type for method in ("data1", "data2")):
            keep_types.append(cell_type)

    if not keep_types:
        raise ValueError(
            "No cell types remain after --min-cells-per-type filtering. "
            "Lower the threshold or inspect the cell type labels."
        )
    return adata[adata.obs["cell_type"].astype(str).isin(keep_types)].copy()


def downsample_batch_indices(n_rows: int, max_cells: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_cells is None or n_rows <= max_cells:
        return np.arange(n_rows, dtype=int)
    return np.sort(rng.choice(n_rows, size=int(max_cells), replace=False)).astype(int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a two-batch lupus AnnData file into the repo benchmark dataset format."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to the lupus .h5ad file.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "lupus")
    parser.add_argument("--batch-col", required=True, help="obs column used to split the two batches.")
    parser.add_argument("--cell-type-col", default=None, help="obs column with cell type labels.")
    parser.add_argument("--batch0", default=None, help="Value in --batch-col to export as data1.")
    parser.add_argument("--batch1", default=None, help="Value in --batch-col to export as data2.")
    parser.add_argument("--layer", default="counts", help="Layer to use before normalization; falls back to .X.")
    parser.add_argument("--n-top-genes", type=int, default=1000)
    parser.add_argument("--min-genes", type=int, default=10)
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--min-cells-per-type", type=int, default=20)
    parser.add_argument("--max-cells-per-batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-normalize", action="store_true", help="Use input values directly for HVG/export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adata = sc.read_h5ad(args.input)

    batch_values = obs_column(adata.obs, args.batch_col, "batch").astype(str)
    batch0, batch1 = choose_batches(batch_values, args.batch0, args.batch1)

    keep = batch_values.isin([batch0, batch1]).to_numpy()
    adata = adata[keep].copy()
    adata.obs["method"] = batch_values[keep].map({batch0: "data1", batch1: "data2"}).to_numpy()

    if args.cell_type_col is not None:
        adata.obs["cell_type"] = obs_column(adata.obs, args.cell_type_col, "cell type").astype(str).to_numpy()

    if args.layer in adata.layers:
        adata.X = adata.layers[args.layer].copy()

    sc.pp.filter_cells(adata, min_genes=int(args.min_genes))
    sc.pp.filter_genes(adata, min_cells=int(args.min_cells))

    if not args.skip_normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    if args.n_top_genes > 0 and args.n_top_genes < adata.n_vars:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=int(args.n_top_genes),
            batch_key="method",
            flavor="seurat",
        )
        adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()

    nonzero_cells = np.asarray(adata.X.sum(axis=1)).reshape(-1) != 0
    adata = adata[nonzero_cells].copy()
    adata = keep_frequent_cell_types(adata, int(args.min_cells_per_type))

    rng = np.random.default_rng(int(args.seed))
    masks = {
        "data1": np.where(adata.obs["method"].to_numpy() == "data1")[0],
        "data2": np.where(adata.obs["method"].to_numpy() == "data2")[0],
    }
    selected = {}
    for method, indices in masks.items():
        batch_choice = downsample_batch_indices(len(indices), args.max_cells_per_batch, rng)
        selected[method] = indices[batch_choice]

    x0 = matrix_to_dense_float32(adata.X[selected["data1"]])
    x1 = matrix_to_dense_float32(adata.X[selected["data2"]])
    metadata = pd.concat(
        [
            adata.obs.iloc[selected["data1"]].copy(),
            adata.obs.iloc[selected["data2"]].copy(),
        ],
        axis=0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "X1.npy", x0)
    np.save(args.output_dir / "X2.npy", x1)
    np.save(args.output_dir / "dataset_names.npy", np.asarray([batch0, batch1], dtype=object))
    pd.Series(adata.var_names.astype(str)).to_csv(args.output_dir / "features.txt", index=False, header=False)
    metadata.to_csv(args.output_dir / "cells_metadata.csv")

    if "cell_type" in metadata.columns:
        names = tuple(sorted(pd.unique(metadata["cell_type"].astype(str))))
        label_to_index = {name: idx for idx, name in enumerate(names)}
        labels = metadata["cell_type"].astype(str).map(label_to_index).to_numpy(dtype=int)
        n0 = x0.shape[0]
        np.save(args.output_dir / "labels1.npy", labels[:n0])
        np.save(args.output_dir / "labels2.npy", labels[n0:])
        np.save(args.output_dir / "cell_type_names.npy", np.asarray(names, dtype=object))

    print(f"Saved lupus benchmark arrays to {args.output_dir}")
    print(f"X1 shape: {x0.shape}; X2 shape: {x1.shape}")


if __name__ == "__main__":
    main()
