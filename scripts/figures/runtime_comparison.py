#!/usr/bin/env python3
"""The all-methods runtime figure with one added row under a method, from another results root.

Never runs a method. Every row comes from --results-root, except that --anchor-root (when given)
supplies the anchor method's own rows; the added row is the anchor method's runs in --extra-root,
drawn directly under the anchor and labelled --extra-label.

Example: OT-tSNE with the full plan SVD, and below it with a randomized 1000-direction SVD
(both combined roots hold every benchmark method):

  scripts/figures/runtime_comparison.py \\
      --results-root <randsvd1000 combined>/benchmark--combined \\
      --anchor-root <full-SVD combined>/benchmark--combined \\
      --extra-root <randsvd1000 combined>/benchmark--combined \\
      --anchor-label "OT-tSNE (full SVD)" --extra-label "OT-tSNE (rSVD 1000)" \\
      --output-dir <figures>/runtime --name runtime_all_methods_ot_fullsvd_vs_randsvd1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ne_aligned_utils.paper.figures.runtime import write_runtime_comparison
from ne_aligned_utils.paper.inputs.current import CANONICAL_DATASETS
from ne_aligned_utils.paper.inputs.runtime import load_runtime_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--extra-root", type=Path, required=True)
    parser.add_argument("--extra-label", required=True)
    parser.add_argument("--anchor-root", type=Path, default=None)
    parser.add_argument("--anchor-label", default=None, help="new label for the anchor's own row")
    parser.add_argument("--anchor", default="ot", help="method key the added row goes under")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True, help="output file stem")
    parser.add_argument("--datasets", nargs="+", default=list(CANONICAL_DATASETS))
    args = parser.parse_args()

    datasets = tuple(args.datasets)
    frame = load_runtime_frame(args.results_root, datasets=datasets)
    if args.anchor_root is not None:
        anchor = load_runtime_frame(args.anchor_root, datasets=datasets)
        anchor = anchor.loc[anchor["method_key"] == args.anchor]
        if anchor.empty:
            raise ValueError(f"{args.anchor_root} has no {args.anchor!r} rows")
        frame = pd.concat(
            [frame.loc[frame["method_key"] != args.anchor], anchor], ignore_index=True
        )
    extra = load_runtime_frame(args.extra_root, datasets=datasets)
    written = write_runtime_comparison(
        frame,
        extra,
        output_dir=args.output_dir,
        datasets=datasets,
        name=args.name,
        extra_label=args.extra_label,
        anchor=args.anchor,
        anchor_label=args.anchor_label,
    )
    for path in written.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
