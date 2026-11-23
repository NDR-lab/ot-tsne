#!/usr/bin/env python3
"""Hyperparameter and initialisation figures from benchmark metric tables; never runs a method.

Every figure shows changes from a reference configuration: the default point (rel_reg 1, k 15)
for the sensitivity grid, and the random init for the initialization runs.

  # sensitivity grid (results root of configs/ot_sensitivity.toml); --main-root adds the figures
  # that are not relative to the default: OT-tSNE's placing among all methods, and its actual
  # scores against the other methods
  scripts/figures/hyperparameters.py grid --metrics <sweep root> --output-dir <dir> \\
      [--main-root <main benchmark root> [--supplementary-root <root>]]

  # initialisation: the random-init root (the reference), the main root (eot init) and the other
  # init roots; --galleries draws the seed galleries and --agreement the layout agreement, both
  # from the aggregate caches
  scripts/figures/hyperparameters.py init --metrics <random root> <main root> <pca root> \\
      <spectral root> --output-dir <dir> [--galleries] [--agreement]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ne_aligned_utils.paper import layout
from ne_aligned_utils.paper.figures import hyperparameters as draw
from ne_aligned_utils.paper.inputs.current import CANONICAL_DATASETS
from ne_aligned_utils.paper.summaries import hyperparameters as summary

COMPOSITES = ("overall", "bio", "mixing")
SINGLE = ("purity", "silhouette", "ilisi", "kbet")
#: The single-metric pooled figures add the overall score as a centred last row.
SINGLE_WITH_OVERALL = (*SINGLE, "overall")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("grid", "init"):
        p = sub.add_parser(name)
        p.add_argument(
            "--metrics",
            type=Path,
            nargs="+",
            required=True,
            help="results roots holding metrics.csv, or metric CSV files; rows are concatenated",
        )
        p.add_argument("--datasets", nargs="+", help="default: the canonical datasets present")
        p.add_argument(
            "--noise",
            choices=("se", "default-sd"),
            default="se",
            help="noise scale a change must exceed: the SE of the difference of seed means "
            "(default), or the reference run's seed SD",
        )
        p.add_argument("--noise-factor", type=float, default=summary.NOISE_FACTOR)
        p.add_argument(
            "--min-change", type=float, default=summary.MIN_CHANGE, help="default 0.01 (1 point)"
        )
        p.add_argument("--output-dir", type=Path, required=True, help="new or empty")
    grid = sub.choices["grid"]
    grid.add_argument("--main-root", type=Path, help="main benchmark root, for the rank grids")
    grid.add_argument("--supplementary-root", type=Path)
    grid.add_argument("--normalizations", nargs="+", default=["minmax", "raw"])
    init = sub.choices["init"]
    init.add_argument("--methods", nargs="+", default=list(summary.OT_VARIANTS))
    init.add_argument("--device", default="cpu", help="keep only rows run on this device")
    init.add_argument("--galleries", action="store_true", help="seed galleries per init")
    init.add_argument(
        "--gallery-methods", nargs="+", default=["ot"], help="variants to draw galleries for"
    )
    init.add_argument("--agreement", action="store_true", help="layout agreement between seeds")
    init.add_argument(
        "--main-root",
        type=Path,
        help="main benchmark root: init-dots then uses the benchmark metric figures' y ranges",
    )
    init.add_argument("--supplementary-root", type=Path)
    return parser.parse_args(argv)


def _output(path: Path) -> Path:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"output must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _datasets(frame, chosen) -> list[str]:
    if chosen:
        return list(chosen)
    present = set(frame["dataset"])
    return [d for d in CANONICAL_DATASETS if d in present] + sorted(
        present - set(CANONICAL_DATASETS)
    )


def _noise(args) -> dict:
    return {
        "noise": args.noise.replace("-", "_"),
        "noise_factor": args.noise_factor,
        "min_change": args.min_change,
    }


def run_grid(args) -> None:
    frame = summary.read_metric_tables(args.metrics)
    datasets = _datasets(frame, args.datasets)
    target = _output(args.output_dir)
    changes = summary.grid_changes(frame, datasets=datasets, **_noise(args))
    changes.to_csv(target / "sensitivity-changes.csv", index=False)
    n_rel, n_k = changes["rel_reg"].nunique(), changes["effective_k"].nunique()
    print(f"{n_rel} rel_reg × {n_k} k values over {len(datasets)} datasets")

    with layout.paper_style():
        lines = {}
        for sweep in summary.SWEPT:
            try:
                lines[sweep] = summary.grid_line(changes, sweep)
            except ValueError as exc:
                print(f"no {sweep} line: {exc}")
        if lines:
            path = draw.line_panels(lines, COMPOSITES, output_root=target, datasets=datasets)
            print(f"wrote {path}")
            path = draw.line_panels(
                lines,
                SINGLE,
                output_root=target,
                name="sensitivity-lines-metrics",
                datasets=datasets,
            )
            print(f"wrote {path}")
        if n_rel > 1 and n_k > 1:
            pooled = summary.pooled_grid(changes, datasets)
            pooled.to_csv(target / "sensitivity-pooled.csv", index=False)
            for metrics, name in (
                (COMPOSITES, "sensitivity-pooled"),
                (SINGLE_WITH_OVERALL, "sensitivity-pooled-metrics"),
            ):
                path = draw.pooled_grids(
                    pooled, metrics, name=name, output_root=target, centre_last=True
                )
                print(f"wrote {path}")
            for metric in (*COMPOSITES, *SINGLE):
                path = draw.dataset_grids(changes, metric, datasets=datasets, output_root=target)
                print(f"wrote {path}")
        if args.main_root is not None:
            from ne_aligned_utils.paper.inputs.current import load_canonical_frame

            paper = load_canonical_frame(
                args.main_root,
                datasets=datasets,
                supplementary_root=args.supplementary_root,
                require_source_caches=False,
            )
            for normalization in args.normalizations:
                ranks = summary.ot_rank_grid(
                    paper, frame, datasets=datasets, normalization=normalization
                )
                ranks.to_csv(target / f"sensitivity-rank-{normalization}.csv", index=False)
                print(f"wrote {draw.rank_grids(ranks, output_root=target)}")
                print(f"wrote {draw.placing_lines(ranks, output_root=target)}")
            # Not relative to the default: actual scores among the other methods.
            context = summary.context_scores(paper, datasets=datasets)
            context.to_csv(target / "sensitivity-context.csv", index=False)
            for metric in COMPOSITES:
                for plot in (draw.robustness_strip, draw.absolute_lines):
                    path = plot(changes, context, metric, datasets=datasets, output_root=target)
                    print(f"wrote {path}")


def run_init(args) -> None:
    frame = summary.read_metric_tables(args.metrics)
    frame = frame[
        frame["method"].isin(args.methods)
        & (frame["device"].astype(str) == args.device)
        & np.isclose(frame["rel_reg"].astype(float), summary.DEFAULT_POINT[0])
        & np.isclose(frame["effective_k"].astype(float), summary.DEFAULT_POINT[1])
    ]
    if "graph_symmetrization" in frame:
        frame = frame[frame["graph_symmetrization"].astype(str) == "or"]
    datasets = _datasets(frame, args.datasets)
    target = _output(args.output_dir)
    changes = summary.init_changes(frame, datasets=datasets, methods=args.methods, **_noise(args))
    pooled = summary.pooled_init(changes, datasets)
    changes.to_csv(target / "init-changes.csv", index=False)
    pooled.to_csv(target / "init-pooled.csv", index=False)
    inits = summary.init_order(changes["init"].unique())
    print(f"inits {inits}, variants {args.methods}, {len(datasets)} datasets")

    with layout.paper_style():
        for metrics, name in (
            (COMPOSITES, "init-pooled"),
            (SINGLE_WITH_OVERALL, "init-pooled-metrics"),
        ):
            path = draw.init_grids(
                pooled,
                metrics,
                methods=args.methods,
                name=name,
                output_root=target,
                pooled=True,
                centre_last=True,
            )
            print(f"wrote {path}")
        for metric in (*COMPOSITES, *SINGLE):
            path = draw.init_grids(
                changes,
                (metric,),
                methods=args.methods,
                name=f"init-datasets-{metric}",
                output_root=target,
                pooled=False,
                datasets=datasets,
            )
            print(f"wrote {path}")
        agreement = None
        if args.agreement:
            agreement = summary.layout_agreement(
                frame, datasets=datasets, methods=args.methods, inits=inits
            )
            agreement.to_csv(target / "init-agreement.csv", index=False)
        path = draw.seed_spread(
            pooled, ("bio", "mixing"), methods=args.methods, output_root=target, agreement=agreement
        )
        print(f"wrote {path}")
        benchmark = None
        if args.main_root is not None:
            benchmark = summary.benchmark_metric_summary(
                args.main_root, datasets=datasets, supplementary_root=args.supplementary_root
            )
        path = draw.init_dots(
            changes,
            SINGLE,
            methods=args.methods,
            datasets=datasets,
            output_root=target,
            benchmark=benchmark,
        )
        print(f"wrote {path}")
        if args.galleries:
            for method in args.gallery_methods:
                for dataset in datasets:
                    path = draw.init_gallery(
                        frame, dataset=dataset, method=method, inits=inits, output_root=target
                    )
                    print(f"wrote {path}")


def main(argv=None) -> None:
    args = parse_args(argv)
    {"grid": run_grid, "init": run_init}[args.command](args)


if __name__ == "__main__":
    main()
