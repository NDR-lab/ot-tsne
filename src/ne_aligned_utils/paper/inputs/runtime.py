"""Select canonical task timings from completed benchmark trees; never runs a method."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ne_aligned_utils.paper.inputs.current import (
    CANONICAL_DATASETS,
    EXPECTED_CANONICAL_ROWS,
    SUPPLEMENTARY_REPLACEMENTS,
    multimap_key,
)
from ne_aligned_utils.paper.selection.benchmark import CURRENT_BENCHMARK, BenchmarkSelectionPolicy
from ne_aligned_utils.paper.summaries.task_timing import scan_task_tree

#: The October CUDA timing run of OT-tSNE on yao, inside the supplementary root. Used only when
#: the results root itself has no GPU tasks.
GPU_SUBPATH = "yao_ot_cuda/results/real-data-rerun-v2--01a5b207e04f"
GPU_DATASETS: tuple[str, ...] = ("yao_10x_ent_l2l3",)


def read_scalar(path: Path, name: str):
    """One stored field of a task file, or ``None`` when absent."""
    with np.load(path, allow_pickle=False) as cache:
        if name not in cache.files:
            return None
        value = np.asarray(cache[name])
        return value.item() if value.ndim == 0 else value.tolist()


def _on_gpu(frame: pd.DataFrame) -> pd.Series:
    """Rows of tasks that ran on a GPU. Files that record no device ran on the CPU."""
    return frame["device"].fillna("cpu").astype(str) != "cpu"


def _select(frame: pd.DataFrame, policy: BenchmarkSelectionPolicy) -> pd.DataFrame:
    """The canonical configuration: rel_reg, OR symmetrisation for OT, no UOT or retired arms."""
    frame = frame[frame["timing_schema"].notna()].copy()
    frame["graph_symmetrization"] = [
        read_scalar(Path(path), "graph_symmetrization") for path in frame["task_path"]
    ]
    frame = frame[~frame["task_kind"].isin(policy.excluded_methods)]
    frame = frame[~frame["task_kind"].astype(str).str.startswith("ot-uot-")]
    has_rel_reg = frame["rel_reg"].notna()
    frame = frame[~has_rel_reg | np.isclose(frame["rel_reg"].astype(float), policy.rel_reg)]
    is_ot = frame["task_kind"].isin(policy.ot_methods)
    symmetrized = frame["graph_symmetrization"].astype(str) == policy.graph_symmetrization
    return frame[~is_ot | symmetrized]


def _method_keys(frame: pd.DataFrame) -> list[str]:
    return [
        multimap_key(
            str(source) == "supplementary:multimap_native"
            or bool(read_scalar(Path(path), "multimap_native"))
        )
        if str(kind) == "multimap"
        else str(kind)
        for kind, source, path in zip(
            frame["task_kind"], frame["result_source"], frame["task_path"], strict=True
        )
    ]


def load_runtime_frame(
    root: Path,
    *,
    datasets: tuple[str, ...] = CANONICAL_DATASETS,
    supplementary_root: Path | None = None,
    policy: BenchmarkSelectionPolicy = CURRENT_BENCHMARK,
) -> pd.DataFrame:
    """Canonical CPU task timings, with the corrected supplementary arms swapped in."""
    pieces = []
    for dataset in datasets:
        directory = Path(root) / dataset
        if not directory.is_dir():
            raise FileNotFoundError(f"missing benchmark dataset result directory: {directory}")
        pieces.append(scan_task_tree(directory))
    frame = pd.concat(pieces, ignore_index=True)
    # GPU tasks share the results root but are a separate timing pool.
    frame = frame[~_on_gpu(frame)].copy()
    frame["result_source"] = "frozen_benchmark"
    if supplementary_root is not None:
        supplementary_root = Path(supplementary_root).resolve()
        replacements = []
        for dataset in datasets:
            for method, (arm, directory) in SUPPLEMENTARY_REPLACEMENTS.items():
                method_root = supplementary_root / arm / directory / dataset
                if not method_root.is_dir():
                    raise FileNotFoundError(f"missing supplementary task tree: {method_root}")
                corrected = scan_task_tree(method_root)
                corrected = corrected[
                    corrected["timing_schema"].notna()
                    & (corrected["task_kind"].astype(str) == method)
                ].copy()
                if sorted(corrected["seed"].astype(int)) != [0, 1, 2]:
                    raise ValueError(f"{method_root} must contain corrected task seeds 0, 1, 2")
                corrected["result_source"] = f"supplementary:{arm}"
                replacements.append(corrected)
        replaced = frame["task_kind"].astype(str).isin(set(SUPPLEMENTARY_REPLACEMENTS))
        frame = pd.concat([frame.loc[~replaced], *replacements], ignore_index=True)
    frame = _select(frame, policy)
    frame["method_key"] = _method_keys(frame)
    expected = {dataset: EXPECTED_CANONICAL_ROWS for dataset in datasets}
    counts = frame.groupby("dataset").size().to_dict()
    if counts != expected:
        raise ValueError(f"canonical runtime selection expected {expected}, found {counts}")
    return frame.reset_index(drop=True)


def _gpu_rows(base: Path, datasets: tuple[str, ...]) -> pd.DataFrame:
    pieces = [scan_task_tree(base / dataset) for dataset in datasets if (base / dataset).is_dir()]
    if not pieces:
        return pd.DataFrame()
    frame = pd.concat(pieces, ignore_index=True)
    # The device is the whole point; never pass a CPU row off as a GPU one.
    return frame[_on_gpu(frame)].copy()


def load_gpu_medians(
    root: Path,
    supplementary_root: Path | None = None,
    *,
    datasets: tuple[str, ...] = GPU_DATASETS,
    policy: BenchmarkSelectionPolicy = CURRENT_BENCHMARK,
) -> pd.DataFrame:
    """Per-(method, dataset) medians of the CUDA tasks; empty when there are none.

    GPU tasks in the results root come first. Only when it has none is the October CUDA run in
    the supplementary root used.
    """
    frame = _gpu_rows(Path(root), datasets)
    frame["result_source"] = "gpu"
    if frame.empty and supplementary_root is not None:
        frame = _gpu_rows(Path(supplementary_root) / GPU_SUBPATH, datasets)
        frame["result_source"] = "supplementary:gpu"
    if frame.empty:
        return pd.DataFrame()
    frame = _select(frame, policy)
    if frame.empty:
        return pd.DataFrame()
    frame["method_key"] = _method_keys(frame)
    return frame.groupby(["method_key", "dataset"], as_index=False, dropna=False).agg(
        time_total=("time_total", "median"),
        time_pre_embedding=("time_pre_embedding", "median"),
        time_embedding=("time_embedding", "median"),
        n_runs=("time_total", "size"),
        n_obs=("n_obs", "max"),
        device=("device", "first"),
    )
