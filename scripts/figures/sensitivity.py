#!/usr/bin/env python3
"""Sensitivity trajectories from benchmark metric tables; never runs a method.

One figure per axis pair, one panel per dataset, one trajectory per run configuration along the
swept setting. Settings that are not fixed with --fix and differ between rows give separate
trajectories; they are never averaged together.

Examples, for the OT-tSNE sensitivity grid (results root of configs/ot_sensitivity.toml):

  # rel_reg trajectories at the benchmark k
  scripts/figures/sensitivity.py --metrics <root>/metrics.csv --sweep rel_reg \\
      --fix effective_k=15 --output-dir <dir>
  # k trajectories, one per rel_reg
  scripts/figures/sensitivity.py --metrics <root>/metrics.csv --sweep effective_k --output-dir <dir>
  # rel_reg trajectories of the OT-tSNE variants in a benchmark table that has a rel_reg grid
  scripts/figures/sensitivity.py --metrics <metrics.csv> --sweep rel_reg \\
      --fix backend=tsne device=cpu --output-dir <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ne_aligned_utils.paper import layout, registry
from ne_aligned_utils.paper.figures.sensitivity import PAIRS, trajectory_grid
from ne_aligned_utils.paper.inputs.current import CANONICAL_DATASETS
from ne_aligned_utils.paper.selection.benchmark import CURRENT_BENCHMARK
from ne_aligned_utils.paper.summaries.sensitivity import sweep_points

DEFAULT_PAIRS = ("purity-kbet", "silhouette-ilisi")


def _setting(text: str) -> tuple[str, str]:
    column, sep, value = text.partition("=")
    if not sep or not column or not value:
        raise argparse.ArgumentTypeError(f"expected column=value, got {text!r}")
    return column, value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        nargs="+",
        required=True,
        help="metrics CSV files, or results roots holding metrics.csv; rows are concatenated",
    )
    parser.add_argument(
        "--sweep",
        required=True,
        help="setting along each trajectory: a metric-table column (rel_reg, effective_k, "
        "rel_reg_m) or a method_settings key (for example beta)",
    )
    parser.add_argument(
        "--fix",
        type=_setting,
        nargs="*",
        default=[],
        metavar="COLUMN=VALUE",
        help="keep only rows with these settings, for example effective_k=15 backend=tsne",
    )
    parser.add_argument("--methods", nargs="+", help="benchmark method names (default: all)")
    parser.add_argument(
        "--datasets", nargs="+", help="panel order (default: every dataset in the tables)"
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        choices=sorted(PAIRS),
        default=list(DEFAULT_PAIRS),
        help="axis pairs, bio metric on y and mixing metric on x (default: %(default)s)",
    )
    parser.add_argument("--ncols", type=int, help="panels per row (default: up to four)")
    parser.add_argument("--output-dir", type=Path, required=True, help="new or empty")
    return parser.parse_args(argv)


def _read(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        table = path / "metrics.csv" if path.is_dir() else path
        if not table.is_file():
            raise FileNotFoundError(table)
        frames.append(pd.read_csv(table))
    return pd.concat(frames, ignore_index=True)


def _datasets(frame: pd.DataFrame, chosen) -> list[str]:
    if chosen:
        return list(chosen)
    present = set(frame["dataset"])
    known = [d for d in CANONICAL_DATASETS if d in present]
    return known + sorted(present - set(known))


def _label(method: str) -> str:
    return registry.resolve(method).label


def main(argv=None) -> None:
    args = parse_args(argv)
    target = args.output_dir
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(f"output must be new or empty: {target}")
    frame = _read(args.metrics)
    datasets = _datasets(frame, args.datasets)
    points = sweep_points(
        frame,
        sweep=args.sweep,
        datasets=datasets,
        fixed=dict(args.fix),
        methods=args.methods,
        method_labels=_label,
    )
    datasets = [d for d in datasets if d in set(points["dataset"])]
    reported = {
        "method": "ot",
        "rel_reg": CURRENT_BENCHMARK.rel_reg,
        "effective_k": CURRENT_BENCHMARK.effective_k,
        "graph_symmetrization": CURRENT_BENCHMARK.graph_symmetrization,
        "backend": CURRENT_BENCHMARK.backend,
    }
    target.mkdir(parents=True, exist_ok=True)
    points.to_csv(target / f"sensitivity-{args.sweep}-points.csv", index=False)
    lines = points.drop_duplicates("line")["line_label"]
    print(f"{len(lines)} trajectories over {len(datasets)} datasets:")
    for label in lines:
        print(f"  {label}")
    with layout.paper_style():
        for pair in args.pairs:
            path = trajectory_grid(
                points,
                f"sensitivity-{args.sweep}-{pair}",
                sweep=args.sweep,
                pair=pair,
                datasets=datasets,
                output_root=target,
                reported=reported,
                ncols=args.ncols,
            )
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
