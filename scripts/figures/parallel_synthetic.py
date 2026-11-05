"""Render both parallel-PC1 modes from one explicit saved run; never fit methods."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ne_aligned_utils.paper.figures.parallel_synthetic import (
    ParallelPlotConfig,
    build_parallel_figures,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--magnitude", type=float, default=10.0)
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--embedding-seed", type=int, default=0)
    parser.add_argument("--rel-reg", type=float, default=1.0)
    args = parser.parse_args(argv)
    selection = ParallelPlotConfig(
        args.magnitude, args.corruption_seed, args.embedding_seed, args.rel_reg
    )
    figures = build_parallel_figures(args.run_root, output_dir=args.output_dir, selection=selection)
    for fig in figures.values():
        plt.close(fig)
    print(f"Saved {len(figures)} comparison figures under {args.output_dir}")


if __name__ == "__main__":
    main()
