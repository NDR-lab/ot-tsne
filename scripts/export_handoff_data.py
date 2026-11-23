"""Export a benchmark results root into the handoff/data folder read by
notebooks/reproduce_paper_figures.ipynb.

The notebook imports no project code; this script is the only bridge between a results root and
the notebook. It writes:

  style.mplstyle            the paper plot style
  scored_long_frame.csv     canonical scored rows (one per dataset, method, seed)
  method_registry.csv       method names, families and markers from paper.registry
  embeddings/<dataset>.npz  seed-0 coordinates of the panel methods, plus batch and cell-type labels

usage: PYTHONPATH=src python scripts/export_handoff_data.py RESULTS_ROOT [OUT_DIR]
       (OUT_DIR defaults to handoff/data at the repository root)
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ne_aligned_utils.paper import registry
from ne_aligned_utils.paper.inputs.current import CANONICAL_DATASETS, load_canonical_frame
from ne_aligned_utils.paper.inputs.embeddings import load_reported_embedding

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Methods drawn in the notebook's embedding panels (its QUALITATIVE_COLUMNS).
EMBEDDING_KEYS = (
    "joint",
    "ot",
    "multimap",
    "rc-tsne",
    "seurat-50",
    "liger-50",
    "scanorama-50",
    "scvi-50",
    "eot-50",
)
REL_REG = 1.0


def export(root: Path, out: Path) -> None:
    (out / "embeddings").mkdir(parents=True, exist_ok=True)

    shutil.copy(
        Path(registry.__file__).parent / "styles" / "style.mplstyle", out / "style.mplstyle"
    )

    frame = load_canonical_frame(root, datasets=CANONICAL_DATASETS)
    frame.to_csv(out / "scored_long_frame.csv", index=False)

    # `shade` breaks ties inside one (family, subfamily); the registry's row order supplies it.
    rows = [
        {
            "key": key,
            "label": registry.METHODS[key].label,
            "family": registry.METHODS[key].family,
            "subfamily": registry.METHODS[key].subfamily,
            "shade": shade,
            "marker": registry.METHODS[key].marker,
            "color": registry.METHODS[key].color,
        }
        for shade, key in enumerate(registry.method_order())
    ]
    pd.DataFrame(rows).to_csv(out / "method_registry.csv", index=False)

    for dataset in CANONICAL_DATASETS:
        arrays: dict[str, np.ndarray] = {}
        for key in EMBEDDING_KEYS:
            embedding, cache = load_reported_embedding(frame, dataset, key, rel_reg=REL_REG)
            arrays[f"embedding__{key}"] = embedding
            labels = {
                "batch_ids": np.asarray(cache["batch_ids"], dtype=int),
                "labels0": np.asarray(cache["labels0"])[0],
                "labels1": np.asarray(cache["labels1"])[0],
                "label_names": np.asarray(cache["label_names"]).astype(str),
            }
            if "batch_ids" in arrays:
                for name, value in labels.items():
                    if not np.array_equal(arrays[name], value):
                        raise ValueError(f"{dataset}/{key}: {name} differs from the other methods")
            arrays.update(labels)
        np.savez_compressed(out / "embeddings" / f"{dataset}.npz", **arrays)
        print(f"{dataset:<20} {len(arrays['batch_ids']):>6} cells, {len(EMBEDDING_KEYS)} methods")
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_root", type=Path)
    parser.add_argument("out_dir", type=Path, nargs="?", default=REPO_ROOT / "handoff" / "data")
    args = parser.parse_args()
    export(args.results_root, args.out_dir)


if __name__ == "__main__":
    main()
