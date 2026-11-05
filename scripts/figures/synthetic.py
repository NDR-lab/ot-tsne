"""Render saved synthetic results from an explicit source checkout; never run OT/t-SNE."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

from ne_aligned_utils.paper import layout
from ne_aligned_utils.paper.figures.synthetic import PANEL_TITLE_PT, build_variant
from ne_aligned_utils.paper.inputs import synthetic


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", action="append", choices=tuple(synthetic.DEFAULT_RUNS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Explicit structured independent-scale run; no latest-run discovery",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source, output = args.source_root.resolve(), args.output_dir.resolve()
    variants = args.variant or list(synthetic.DEFAULT_RUNS)
    if len(variants) != len(set(variants)):
        raise ValueError("variants must be unique")
    if args.run_root is not None and variants != ["two_sided_independent_scales"]:
        raise ValueError("--run-root requires --variant two_sided_independent_scales")
    prepared = {
        variant: synthetic.preflight(
            variant, source_root=source, output_dir=output, run_root=args.run_root
        )
        for variant in variants
    }
    print(f"validated saved inputs for {', '.join(variants)}")
    if args.dry_run:
        return
    inputs = set()
    for runs in prepared.values():
        inputs.update(
            path
            for directory in runs.run_dirs.values()
            for path in directory.rglob("*")
            if path.is_file()
        )
        if runs.configs["original"].get("schema_version") != 1:
            inputs.add(synthetic.cache_directory(source) / f"{runs.variant}_directed_ot_top15.csv")
            inputs.add(
                synthetic.cache_directory(source)
                / f"{runs.variant}_unintegrated_tsne_disjoint_mag10.npz"
            )
        else:
            inputs.add(args.run_root / "config.json")
    inputs.update(
        path
        for path in (synthetic.mnist_directory(source) / "MNIST/raw").iterdir()
        if path.is_file()
    )

    def hashes():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(inputs)}

    before = hashes()
    paths = []
    with (
        layout.paper_style(),
        plt.rc_context(
            {
                key: PANEL_TITLE_PT
                for key in ("xtick.labelsize", "ytick.labelsize", "legend.fontsize")
            }
        ),
    ):
        for variant, runs in prepared.items():
            paths.extend(
                build_variant(variant, source_root=source, output_dir=output, prepared=runs)
            )
    if hashes() != before:
        raise ValueError("scientific inputs changed during synthetic rendering")
    manifest = {
        "source_root": str(source),
        "variants": variants,
        "sampling": synthetic.SAMPLING,
        "inputs": before,
        "outputs": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "reconstruction": "MNIST split and PCA reconstructed from the recorded configuration",
        "methods_executed": [],
    }
    (output / "source_runs.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(paths)} figures and source_runs.json under {output}")


if __name__ == "__main__":
    main()
