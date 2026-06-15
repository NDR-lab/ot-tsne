#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ne_aligned_utils.benchmark_metrics import write_metric_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute all applicable metrics from aggregate benchmark caches."
    )
    parser.add_argument(
        "--aggregate-cache",
        action="append",
        required=True,
        type=Path,
        help="Aggregate .npz cache. Repeat for multiple datasets.",
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional subset of methods present in each cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_csv = args.summary_csv
    if summary_csv is None:
        summary_csv = args.output_csv.with_name(
            f"{args.output_csv.stem}_summary{args.output_csv.suffix}"
        )
    rows, summary = write_metric_reports(
        args.aggregate_cache,
        args.output_csv,
        summary_csv,
        methods=args.methods,
    )
    print(f"Wrote {len(rows)} per-run rows to {args.output_csv}")
    print(f"Wrote {len(summary)} summary rows to {summary_csv}")


if __name__ == "__main__":
    main()
