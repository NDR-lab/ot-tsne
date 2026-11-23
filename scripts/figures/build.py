#!/usr/bin/env python3
"""Build paper figures from completed benchmark artifacts; never runs a method.

Default: every benchmark family (metrics, runtime, ranks, qualitative, composition). The methods
schematic is built on its own: --families methods --method-arrays <npz> --output-dir <dir>.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ne_aligned_utils.paper.build import DEFAULT_TOP_R, FAMILIES, build, preflight
from ne_aligned_utils.paper.figures.methods import build_methods, preflight_methods
from ne_aligned_utils.paper.inputs.current import CANONICAL_DATASETS
from ne_aligned_utils.paper.selection.benchmark import (
    CURRENT_BENCHMARK,
    BenchmarkSelectionPolicy,
)


def _policy(args) -> BenchmarkSelectionPolicy:
    """The configuration to show: the reported benchmark unless an option changes it."""
    chosen = dict(
        rel_reg=float(args.rel_reg),
        effective_k=float(args.effective_k),
        graph_symmetrization=str(args.graph_symmetrization),
    )
    if all(getattr(CURRENT_BENCHMARK, key) == value for key, value in chosen.items()):
        return CURRENT_BENCHMARK
    name = "rel_reg={rel_reg:g} k={effective_k:g} sym={graph_symmetrization}".format(**chosen)
    return replace(CURRENT_BENCHMARK, name=name, **chosen)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--supplementary-root", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, help="new or empty; default <root>/paper_figures"
    )
    parser.add_argument("--datasets", nargs="+", default=list(CANONICAL_DATASETS))
    parser.add_argument(
        "--families", nargs="+", choices=(*FAMILIES, "methods"), default=list(FAMILIES)
    )
    parser.add_argument(
        "--top-r",
        nargs="+",
        type=int,
        default=list(DEFAULT_TOP_R),
        help="write a top-r finishes figure for each r (default: 3 2)",
    )
    parser.add_argument(
        "--rel-reg",
        type=float,
        default=CURRENT_BENCHMARK.rel_reg,
        help=f"OT regularisation the figures show (default {CURRENT_BENCHMARK.rel_reg:g})",
    )
    parser.add_argument(
        "--effective-k",
        type=float,
        default=CURRENT_BENCHMARK.effective_k,
        help=f"OT-tSNE k the figures show (default {CURRENT_BENCHMARK.effective_k:g})",
    )
    parser.add_argument(
        "--graph-symmetrization",
        choices=("or", "and"),
        default=CURRENT_BENCHMARK.graph_symmetrization,
        help="OT-tSNE graph symmetrisation the figures show (default or)",
    )
    parser.add_argument("--method-arrays", type=Path, help="saved methods stages; never recomputed")
    parser.add_argument("--dry-run", action="store_true", help="preflight only; write nothing")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if "methods" in args.families:
        if args.families != ["methods"] or args.method_arrays is None or args.output_dir is None:
            raise ValueError(
                "methods requires its own --families methods build, --method-arrays "
                "and --output-dir"
            )
        prepared = preflight_methods(args.method_arrays, args.output_dir)
        print(f"validated methods arrays: {prepared.source}")
        print(f"output root: {prepared.output}")
        if not args.dry_run:
            print(f"wrote {build_methods(prepared)}")
        return
    if args.results_root is None:
        raise ValueError("benchmark families require --results-root")
    if args.method_arrays is not None:
        raise ValueError("--method-arrays applies only to --families methods")
    output = args.output_dir or args.results_root / "paper_figures"
    prepared = preflight(
        args.results_root,
        output,
        datasets=tuple(args.datasets),
        families=tuple(args.families),
        supplementary_root=args.supplementary_root,
        top_r=tuple(args.top_r),
        policy=_policy(args),
    )
    print(
        f"validated {len(prepared.frame)} canonical rows across {len(prepared.datasets)} datasets"
    )
    print(f"pooled over: {', '.join(prepared.pool)}")
    print(f"families: {', '.join(prepared.families)}")
    print(f"output root: {prepared.output}")
    if not args.dry_run:
        print(f"wrote {build(prepared)}")


if __name__ == "__main__":
    main()
