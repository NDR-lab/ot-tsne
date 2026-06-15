#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from ne_aligned_utils.benchmark_reporting import generate_overview_artifacts


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHES = {
    "nCoV": ROOT
    / "benchmarking_new/nCov/n_per_batch_all__q_2__epochs_200__tsne_pca_50/nCoV_benchmark_embeddings.npz",
    "Boris": ROOT
    / "benchmarking_new/Boris_sc/n_per_batch_5000__q_2__epochs_200__tsne_pca_50/Boris_sc_benchmark_embeddings.npz",
    "Yao": ROOT
    / "benchmarking_new/yao_10x_ent_l2l3/n_per_batch_5000_10000__q_2__epochs_200__tsne_pca_50/yao_10x_ent_l2l3_benchmark_embeddings.npz",
    "lupus": ROOT
    / "data/benchmark_embeddings/lupus/n_per_batch_all__q_2__epochs_200__tsne_pca_50/lupus_benchmark_embeddings.npz",
}


def _cache_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("cache must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("cache must use NAME=PATH")
    return name, Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark overview CSVs, figures, and an executed notebook."
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=ROOT / "results/benchmark_overview/regularization_raw_metrics.csv",
    )
    parser.add_argument(
        "--cache",
        type=_cache_argument,
        action="append",
        help="Aggregate cache as NAME=PATH. Repeat to replace the defaults.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/benchmark_overview"
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=ROOT / "notebooks/benchmarking_new_embedding_overview.ipynb",
    )
    parser.add_argument(
        "--skip-notebook",
        action="store_true",
        help="Generate tables and figures without executing the notebook.",
    )
    return parser.parse_args()


def execute_notebook(path: Path, output_dir: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    previous = os.environ.get("NE_ALIGNED_OVERVIEW_DIR")
    os.environ["NE_ALIGNED_OVERVIEW_DIR"] = str(output_dir.resolve())
    try:
        executed = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
    finally:
        if previous is None:
            os.environ.pop("NE_ALIGNED_OVERVIEW_DIR", None)
        else:
            os.environ["NE_ALIGNED_OVERVIEW_DIR"] = previous
    nbformat.write(executed, path)


def main() -> None:
    args = parse_args()
    cache_paths = dict(args.cache) if args.cache else DEFAULT_CACHES
    paths = generate_overview_artifacts(
        args.metrics_csv, cache_paths, args.output_dir
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    if not args.skip_notebook:
        execute_notebook(args.notebook, args.output_dir)
        print(f"executed_notebook: {args.notebook}")


if __name__ == "__main__":
    main()
