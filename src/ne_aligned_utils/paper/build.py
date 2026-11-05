"""Preflight and build the paper figure families from completed benchmark artifacts.

``preflight`` reads and validates every input a requested family needs and writes nothing.
``build`` renders into a new, empty directory and writes ``figure-manifest.json`` last, recording
the input hashes, the tables behind every figure and the hash of every output.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ne_aligned_utils import config
from ne_aligned_utils.analysis import cell_type_palette
from ne_aligned_utils.paper import layout, registry
from ne_aligned_utils.paper.figures import benchmark as benchmark_figures, composition, runtime
from ne_aligned_utils.paper.figures.qualitative import GalleryBlock
from ne_aligned_utils.paper.inputs import runtime as runtime_inputs
from ne_aligned_utils.paper.inputs.current import CANONICAL_DATASETS, load_canonical_frame
from ne_aligned_utils.paper.inputs.embeddings import load_reported_embedding
from ne_aligned_utils.paper.rendering import presentation, styles
from ne_aligned_utils.paper.selection.benchmark import CURRENT_BENCHMARK, reported_metric_rows
from ne_aligned_utils.paper.summaries import ranks as scoring
from ne_aligned_utils.paper.summaries.metrics import pool_datasets, summarise_seeds

FAMILIES = ("metrics", "runtime", "ranks", "qualitative", "composition")
DEFAULT_TOP_R = (3, 2)

#: Metric-grid figures: (file stem, x metric, y metric, x label, y label).
METRIC_GRIDS = (
    ("f_benchmark_ilisi_silhouette", "ilisi", "silhouette", "iLISI", "Silhouette"),
    ("f_benchmark_kbet_purity", "kbet", "purity", "kBET", "Purity"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frame_record(frame: pd.DataFrame) -> dict:
    return dict(
        rows=len(frame),
        columns=list(frame.columns),
        dtypes={name: str(dtype) for name, dtype in frame.dtypes.items()},
        csv_sha256=hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
    )


@dataclass(frozen=True)
class PreparedFigures:
    """Validated, loaded inputs; constructing this value writes no output."""

    root: Path
    output: Path
    datasets: tuple[str, ...]
    #: The datasets every pooled summary (ranks, top-r, pooled figure) averages over.
    pool: tuple[str, ...]
    families: tuple[str, ...]
    supplementary_root: Path | None
    top_r: tuple[int, ...]
    frame: pd.DataFrame
    input_hashes: dict[str, str]
    summary: pd.DataFrame | None = None
    pooled: pd.DataFrame | None = None
    timing: pd.DataFrame | None = None
    gpu_medians: pd.DataFrame | None = None
    #: normalization -> aggregate ranks; (normalization, r) -> top-r counts.
    ranks: dict[str, pd.DataFrame] = field(default_factory=dict)
    top_counts: dict[tuple[str, int], pd.DataFrame] = field(default_factory=dict)
    galleries: dict[str, GalleryBlock] = field(default_factory=dict)
    compositions: list[composition.CompositionBlock] = field(default_factory=list)


def _check_paths(root: Path, output: Path, datasets, supplementary: Path | None) -> None:
    protected = [root / name for name in (*datasets, "metrics")]
    if supplementary is not None:
        protected.append(supplementary)
    if (
        output == root
        or root.is_relative_to(output)
        or any(
            output == path or output.is_relative_to(path) or path.is_relative_to(output)
            for path in protected
        )
    ):
        raise ValueError("output must not overlap scientific input directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be new or empty: {output}")


def _gallery(frame: pd.DataFrame, dataset: str) -> GalleryBlock:
    cached = {
        key: load_reported_embedding(frame, dataset, key, rel_reg=CURRENT_BENCHMARK.rel_reg)
        for key, _heading in styles.QUALITATIVE_COLUMNS
    }
    first = cached[styles.QUALITATIVE_COLUMNS[0][0]][1]
    labels = np.concatenate(
        [np.asarray(first["labels0"][0]), np.asarray(first["labels1"][0])]
    ).astype(int)
    return GalleryBlock(
        name=registry.DATASET_LABELS.get(dataset, dataset),
        embeddings={key: value[0] for key, value in cached.items()},
        caches={key: value[1] for key, value in cached.items()},
        palette=cell_type_palette(tuple(str(value) for value in first["label_names"])),
        labels=labels,
    )


def _joint_cache(frame: pd.DataFrame, dataset: str):
    """The joint t-SNE aggregate cache: labels and batches as every method received them."""
    return load_reported_embedding(frame, dataset, "joint", rel_reg=CURRENT_BENCHMARK.rel_reg)[1]


def preflight(
    root: Path,
    output: Path,
    *,
    datasets: tuple[str, ...],
    families: tuple[str, ...] = FAMILIES,
    supplementary_root: Path | None = None,
    top_r: tuple[int, ...] = DEFAULT_TOP_R,
) -> PreparedFigures:
    """Validate all requested inputs before creating output or rendering anything."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("datasets must be non-empty and unique")
    if any(Path(name).name != name or name in {".", "..", ""} for name in datasets):
        raise ValueError("dataset names must be single path components")
    if not families or len(set(families)) != len(families) or set(families) - set(FAMILIES):
        raise ValueError(f"families must be a non-empty unique subset of {FAMILIES}")
    if not top_r or any(int(r) < 1 for r in top_r) or len(set(top_r)) != len(top_r):
        raise ValueError("top_r must be unique positive integers")
    supplementary = Path(supplementary_root).resolve() if supplementary_root is not None else None
    _check_paths(root, output, datasets, supplementary)
    # Overlapping subsets (hematopoietic_shared) get their own grid row but never join a pool.
    pool = tuple(d for d in datasets if d in CANONICAL_DATASETS)
    if not pool and {"metrics", "ranks", "composition"} & set(families):
        raise ValueError(f"pooled families need at least one of {CANONICAL_DATASETS}")

    needs_caches = bool({"qualitative", "composition"} & set(families))
    frame = load_canonical_frame(
        root,
        datasets=datasets,
        supplementary_root=supplementary,
        require_source_caches=needs_caches,
    )
    prepared: dict = {}
    if "metrics" in families:
        selected = reported_metric_rows(
            frame,
            backend=CURRENT_BENCHMARK.backend,
            no_backend=config.NO_BACKEND,
            rel_reg=CURRENT_BENCHMARK.rel_reg,
            effective_k=CURRENT_BENCHMARK.effective_k,
        )
        prepared["summary"] = summarise_seeds(selected, metrics=list(registry.METRICS))
        prepared["pooled"] = pool_datasets(prepared["summary"], datasets=pool)
    if "runtime" in families:
        prepared["timing"] = runtime_inputs.load_runtime_frame(
            root, datasets=datasets, supplementary_root=supplementary
        )
        prepared["gpu_medians"] = runtime_inputs.load_gpu_medians(root, supplementary)
    if "ranks" in families:
        prepared["ranks"] = {
            normalization: scoring.aggregate_ranks(
                frame, datasets=pool, normalization=normalization, rel_reg=CURRENT_BENCHMARK.rel_reg
            )
            for normalization in scoring.NORMALIZATIONS
        }
        prepared["top_counts"] = {
            (normalization, int(r)): scoring.top_k(
                frame, datasets=pool, normalization=normalization, k=int(r)
            )
            for normalization in scoring.NORMALIZATIONS
            for r in top_r
        }
    if "qualitative" in families:
        prepared["galleries"] = {dataset: _gallery(frame, dataset) for dataset in datasets}
    if "composition" in families:
        prepared["compositions"] = [
            composition.composition_block(
                dataset,
                registry.DATASET_LABELS.get(dataset, dataset),
                _joint_cache(frame, dataset),
            )
            for dataset in pool
        ]

    sources = set()
    for dataset in datasets:
        export = root / "metrics" / f"{dataset}--all-experiments--all-devices.csv"
        sources.add(export if export.is_file() else root / "metrics.csv")
        if "runtime" in families:
            sources.update((root / dataset).rglob("*.npz"))
    if needs_caches:
        sources.update(Path(path).resolve() for path in frame["source_cache"])
    if "runtime" in families:
        sources.update(root.rglob("environment_*.json"))
    if (root / "task_metadata.csv").is_file():
        sources.add(root / "task_metadata.csv")
    if supplementary is not None:
        for pattern in ("*.json", "*.csv", "*.npz"):
            sources.update(supplementary.rglob(pattern))
    hashes = {str(path.resolve()): _sha256(path) for path in sorted(sources)}
    return PreparedFigures(
        root=root,
        output=output,
        datasets=datasets,
        pool=pool,
        families=families,
        supplementary_root=supplementary,
        top_r=tuple(int(r) for r in top_r),
        frame=frame,
        input_hashes=hashes,
        **prepared,
    )


def _metrics(prepared: PreparedFigures, target: Path, frames: dict) -> None:
    frames["metric-summary.csv"] = prepared.summary
    frames["metric-summary-pooled.csv"] = prepared.pooled
    context = presentation.benchmark_render_context(
        target, method_keys=prepared.frame.method_key.unique(), datasets=prepared.datasets
    )
    for stem, x, y, x_label, y_label in METRIC_GRIDS:
        benchmark_figures.dataset_metric_grid(
            prepared.summary,
            stem,
            x=f"{x}_mean",
            y=f"{y}_mean",
            x_label=x_label,
            y_label=y_label,
            render_context=context,
        )
    benchmark_figures.pooled_metric_grid(
        prepared.pooled, "f_pooled_benchmark", render_context=context
    )


def _ranks(prepared: PreparedFigures, target: Path, frames: dict) -> None:
    (target / "ranks").mkdir()
    ranks, counts = presentation.rank_composer(), presentation.top_k_composer()
    for normalization, aggregate in prepared.ranks.items():
        frames[f"ranks/aggregate-{normalization}.csv"] = aggregate
        fig = ranks.rank_aggregate(aggregate)
        layout.save_figure(fig, f"rank-{normalization}", subdir="ranks", root=target)
    for (normalization, r), table in prepared.top_counts.items():
        stem = f"top-{r}-{normalization}"
        frames[f"ranks/{stem}.csv"] = table
        layout.save_figure(counts.top_k_finishes(table), stem, subdir="ranks", root=target)


def _qualitative(prepared: PreparedFigures, target: Path) -> None:
    composer = presentation.qualitative_composer()
    for dataset, block in prepared.galleries.items():
        cached = {key: (block.embeddings[key], block.caches[key]) for key in block.embeddings}
        fig = composer.embedding_panels(
            cached,
            block.palette,
            columns=styles.QUALITATIVE_COLUMNS,
            groups=styles.QUALITATIVE_GROUPS,
        )
        layout.save_figure(
            fig, f"embeddings-{dataset}", kind="density", subdir="qualitative", root=target
        )
    stacked = [
        prepared.galleries[d] for d in styles.STACKED_GALLERY_DATASETS if d in prepared.galleries
    ]
    if not stacked:
        return
    for stem, braces in (("embeddings-all", "once"), ("embeddings-all-grouped", "each")):
        fig = composer.stacked_panels(
            stacked,
            columns=styles.QUALITATIVE_COLUMNS,
            groups=styles.QUALITATIVE_GROUPS,
            braces=braces,
        )
        layout.save_figure(fig, stem, kind="density", subdir="qualitative", root=target)


def _composition(prepared: PreparedFigures, target: Path, frames: dict) -> None:
    blocks = prepared.compositions
    frames["composition/cell-type-counts.csv"] = pd.DataFrame(
        [
            dict(dataset=block.key, cell_type=name, batch_0=int(c0), batch_1=int(c1))
            for block in blocks
            for name, c0, c1 in zip(block.names, *block.counts)
        ]
    )
    (target / "composition").mkdir()
    name_column = composition.name_column_width(blocks)
    for index, part in enumerate(composition.split_by_size(blocks), start=1):
        fig = composition.batch_composition(part, name_column_in=name_column)
        layout.save_figure(fig, f"batch-composition-{index}", subdir="composition", root=target)


def build(prepared: PreparedFigures) -> Path:
    """Render preflighted families and write a success manifest only after verification."""
    target = prepared.output
    # Recheck at the write boundary; preflight is also a public, read-only operation.
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(f"output must be new or empty: {target}")
    for filename, expected in prepared.input_hashes.items():
        if _sha256(Path(filename)) != expected:
            raise RuntimeError(f"input changed after preflight: {filename}")
    target.mkdir(parents=True, exist_ok=True)
    frames = {"selected-metric-rows.csv": prepared.frame}
    try:
        with layout.paper_style():
            if "metrics" in prepared.families:
                _metrics(prepared, target, frames)
            if "runtime" in prepared.families:
                runtime.write_runtime_figures(
                    prepared.timing,
                    root=prepared.root,
                    output_dir=target / "runtime",
                    datasets=prepared.datasets,
                    gpu_medians=prepared.gpu_medians,
                )
            if "ranks" in prepared.families:
                _ranks(prepared, target, frames)
            if "qualitative" in prepared.families:
                _qualitative(prepared, target)
            if "composition" in prepared.families:
                _composition(prepared, target, frames)
        for filename, frame in frames.items():
            frame.to_csv(target / filename, index=False)
        for filename, expected in prepared.input_hashes.items():
            if _sha256(Path(filename)) != expected:
                raise RuntimeError(f"scientific input changed during rendering: {filename}")
        outputs = {}
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            record = dict(sha256=_sha256(path), bytes=path.stat().st_size)
            if path.suffix == ".png":
                with Image.open(path) as image:
                    record["dimensions_px"] = list(image.size)
                    record["dpi"] = list(image.info.get("dpi", ()))
            outputs[str(path.relative_to(target))] = record
        policy = CURRENT_BENCHMARK
        manifest = dict(
            schema_version=2,
            datasets=list(prepared.datasets),
            pooled_datasets=list(prepared.pool),
            families=list(prepared.families),
            top_r=list(prepared.top_r),
            source_result_root=str(prepared.root),
            supplementary_root=(
                str(prepared.supplementary_root)
                if prepared.supplementary_root is not None
                else None
            ),
            configuration=dict(
                rel_reg=policy.rel_reg, effective_k=policy.effective_k, backend=policy.backend
            ),
            environment={
                "python": platform.python_version(),
                **{
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "matplotlib", "pillow")
                },
            },
            inputs=prepared.input_hashes,
            frames={name: _frame_record(f) for name, f in frames.items()},
            outputs=outputs,
        )
        manifest_path = target / "figure-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest_path
    finally:
        plt.close("all")
