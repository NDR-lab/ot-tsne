from __future__ import annotations

import argparse
import csv
import pickle
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from ne_aligned_utils.analysis import NCovAnalysisData  # noqa: E402
from ne_aligned_utils.methods import (  # noqa: E402
    compute_eot_embedding_from_arrays,
    compute_ot_tsne_embedding_from_arrays,
)


ENT_ANALYSIS_SUBCLASSES = ("L2 IT ENTl", "L2 IT ENTm", "L3 IT ENT", "L2/3 IT ENTl")
ENT_ANALYSIS_DATA_DIR = "data/yao_10x_ent_l2l3"
ENT_ANALYSIS_BASE_DIR = "data/yao_10x"
ENT_ANALYSIS_BATCH_COL = "donor_sex_label"
ENT_ANALYSIS_LABEL_COL = "cluster_label"
ENT_ANALYSIS_MAX_CELLS_PER_BATCH = 5000
ENT_ANALYSIS_RANDOM_STATE = 0
ENT_ANALYSIS_DATASET_KEY = "ent_l2l3"
ENT_ANALYSIS_DISPLAY_NAME = "ENT L2/L3"

DEFAULT_PERCENTAGES = (0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2)
DEFAULT_EPS_VALUES = (1.0, 10.0, 100.0, 300.0)
DEFAULT_METRIC_K = 50
DEFAULT_EMBEDDING_RANDOM_STATE = 0
DEFAULT_N_EPOCHS = 200
DEFAULT_BINARIZE = True
DEFAULT_EOT_T = 0
DEFAULT_EOT_SOURCE = "raw_arrays"

SLURM_MANIFEST_FIELDNAMES = [
    "job_kind",
    "eps_times",
    "percentage",
    "job_id",
    "job_name",
    "output_path",
    "log_path",
    "command",
]


def bool_to_variant(binarize: bool) -> str:
    return "binarized" if bool(binarize) else "weighted"


def format_token(value: float | int | str) -> str:
    text = str(value).replace("-", "m").replace(".", "p")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in text)


def resolve_ent_dataset_path(data_path: str | Path = ENT_ANALYSIS_DATA_DIR) -> Path:
    path = Path(data_path).expanduser()
    candidates = [path]

    if not path.is_absolute():
        cwd = Path.cwd().resolve()
        for base in (cwd, cwd.parent, cwd.parent.parent, ROOT):
            candidates.append(base / path)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in seen:
            continue
        seen.add(resolved_candidate)
        if resolved_candidate.exists():
            return resolved_candidate

    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def _allocate_group_sample_sizes(counts: pd.Series, n_target: int) -> pd.Series:
    counts = counts.astype(int).sort_index()
    n_target = min(int(n_target), int(counts.sum()))
    if n_target <= 0:
        return pd.Series(0, index=counts.index, dtype=int)

    ideal = counts / counts.sum() * n_target
    sizes = np.floor(ideal).astype(int)
    min_per_group = 1 if n_target >= len(counts) else 0
    if min_per_group:
        sizes = np.maximum(sizes, min_per_group)
    sizes = np.minimum(sizes, counts)
    sizes = pd.Series(sizes, index=counts.index, dtype=int)

    remaining = int(n_target - sizes.sum())
    if remaining > 0:
        order = (ideal - sizes).sort_values(ascending=False).index.tolist()
        while remaining > 0:
            changed = False
            for group in order:
                if sizes[group] < counts[group]:
                    sizes[group] += 1
                    remaining -= 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                break
    elif remaining < 0:
        order = (ideal - sizes).sort_values(ascending=True).index.tolist()
        while remaining < 0:
            changed = False
            for group in order:
                if sizes[group] > min_per_group:
                    sizes[group] -= 1
                    remaining += 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                break

    return sizes.astype(int)


def stratified_subsample_frame(
    frame: pd.DataFrame,
    group_col: str,
    n_target: int | None,
    random_state: int = 0,
) -> pd.DataFrame:
    if n_target is None or len(frame) <= int(n_target):
        return frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    counts = frame[group_col].value_counts().sort_index()
    allocations = _allocate_group_sample_sizes(counts, int(n_target))
    sampled_parts = []
    for offset, (group, n_group) in enumerate(allocations.items()):
        if n_group <= 0:
            continue
        group_frame = frame[frame[group_col] == group]
        if n_group >= len(group_frame):
            sampled = group_frame.copy()
        else:
            sampled = group_frame.sample(n=int(n_group), random_state=random_state + offset)
        sampled_parts.append(sampled)

    if not sampled_parts:
        raise ValueError("Stratified sampling produced no rows.")

    sampled_frame = pd.concat(sampled_parts, ignore_index=False)
    sampled_frame = sampled_frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled_frame


def load_ent_dataset_arrays(
    subset_path: str | Path = ENT_ANALYSIS_DATA_DIR,
    base_path: str | Path = ENT_ANALYSIS_BASE_DIR,
    subclasses: tuple[str, ...] = ENT_ANALYSIS_SUBCLASSES,
) -> tuple[Path, np.ndarray, pd.DataFrame]:
    subset_dir = resolve_ent_dataset_path(subset_path)
    subset_pca_path = subset_dir / "ent_l2l3_pca50.npy"
    subset_meta_path = subset_dir / "ent_l2l3_meta.csv"

    if subset_pca_path.exists() and subset_meta_path.exists():
        pca50 = np.load(subset_pca_path)
        meta = pd.read_csv(subset_meta_path)
    else:
        base_dir = resolve_ent_dataset_path(base_path)
        base_meta_path = base_dir / "10x_metadata.csv"
        base_pca_path = base_dir / "yao_10x_v2_pca50.pkl"
        if not base_meta_path.exists() or not base_pca_path.exists():
            raise FileNotFoundError(
                "Could not find saved ENT subset files or the base yao_10x PCA/metadata files."
            )

        base_meta = pd.read_csv(base_meta_path)
        with base_pca_path.open("rb") as handle:
            compact = pickle.load(handle)
        mask = base_meta["subclass_label"].isin(subclasses).to_numpy()
        pca50 = np.asarray(compact["pca50"])[mask]
        meta = base_meta.loc[mask].reset_index(drop=True)

    meta = meta.reset_index(drop=True)
    pca50 = np.asarray(pca50, dtype=float)
    if len(meta) != pca50.shape[0]:
        raise ValueError(
            f"ENT metadata rows ({len(meta)}) do not match PCA rows ({pca50.shape[0]})."
        )
    return subset_dir, pca50, meta


def load_ent_analysis_data(
    data_path: str | Path = ENT_ANALYSIS_DATA_DIR,
    base_path: str | Path = ENT_ANALYSIS_BASE_DIR,
    batch_col: str = ENT_ANALYSIS_BATCH_COL,
    label_col: str = ENT_ANALYSIS_LABEL_COL,
    max_cells_per_batch: int | None = ENT_ANALYSIS_MAX_CELLS_PER_BATCH,
    random_state: int = ENT_ANALYSIS_RANDOM_STATE,
) -> tuple[NCovAnalysisData, pd.DataFrame, Path]:
    resolved_path, pca50, meta = load_ent_dataset_arrays(
        subset_path=data_path,
        base_path=base_path,
        subclasses=ENT_ANALYSIS_SUBCLASSES,
    )
    meta = meta.copy()
    meta["_row_index"] = np.arange(len(meta), dtype=int)

    batch_values = meta[batch_col].astype(str)
    unique_batches = sorted(batch_values.dropna().unique())
    preferred_batches = [value for value in ("F", "M") if value in unique_batches]
    if len(preferred_batches) == 2:
        batch_order = preferred_batches
    else:
        if len(unique_batches) < 2:
            raise ValueError(f"Expected at least two batches in {batch_col}, found {unique_batches}.")
        batch_order = unique_batches[:2]

    sampled_batches = []
    for batch_idx, batch_name in enumerate(batch_order):
        batch_frame = meta.loc[batch_values == batch_name].copy()
        sampled_batch = stratified_subsample_frame(
            batch_frame,
            group_col=label_col,
            n_target=max_cells_per_batch,
            random_state=random_state + 1000 * batch_idx,
        )
        sampled_batch[batch_col] = batch_name
        sampled_batches.append(sampled_batch)

    meta_first, meta_second = sampled_batches
    joint_meta = pd.concat([meta_first, meta_second], ignore_index=True)

    label_values = joint_meta[label_col].astype(str)
    cell_type_names = tuple(sorted(label_values.unique()))
    label_to_index = {name: idx for idx, name in enumerate(cell_type_names)}

    x1 = pca50[meta_first["_row_index"].to_numpy(dtype=int)]
    x2 = pca50[meta_second["_row_index"].to_numpy(dtype=int)]
    labels1 = meta_first[label_col].astype(str).map(label_to_index).to_numpy(dtype=int)
    labels2 = meta_second[label_col].astype(str).map(label_to_index).to_numpy(dtype=int)

    data = NCovAnalysisData(
        x1=np.asarray(x1, dtype=float),
        x2=np.asarray(x2, dtype=float),
        labels1=np.asarray(labels1, dtype=int),
        labels2=np.asarray(labels2, dtype=int),
        cell_type_names=cell_type_names,
        dataset_names=(
            f"{ENT_ANALYSIS_DATASET_KEY}-{batch_order[0]}",
            f"{ENT_ANALYSIS_DATASET_KEY}-{batch_order[1]}",
        ),
    )
    return data, joint_meta.reset_index(drop=True), resolved_path


def resolve_task_root(
    dataset_path: Path,
    *,
    variant: str,
    max_cells_per_batch: int | None,
    subsample_random_state: int,
    embedding_random_state: int,
    n_epochs: int,
    eot_t: int,
    eot_source: str,
) -> Path:
    n_token = "all" if max_cells_per_batch is None else str(int(max_cells_per_batch))
    config_token = "__".join(
        [
            f"variant_{format_token(variant)}",
            f"n_per_batch_{format_token(n_token)}",
            f"subsample_seed_{format_token(subsample_random_state)}",
            f"embed_seed_{format_token(embedding_random_state)}",
            f"epochs_{format_token(n_epochs)}",
            f"eot_t_{format_token(eot_t)}",
            f"eot_source_{format_token(eot_source)}",
        ]
    )
    return dataset_path / "joint_ot_tsne_task_cache" / config_token


def resolve_task_output_path(
    task_root: Path,
    *,
    task_kind: str,
    eps_times: float,
    percentage: float | None = None,
    eot_t: int = DEFAULT_EOT_T,
) -> Path:
    eps_token = format_token(f"{float(eps_times):g}")
    if task_kind == "eot":
        return task_root / f"eot_t_{int(eot_t)}_eps_times_{eps_token}.npz"
    if percentage is None:
        raise ValueError("percentage is required for ot-tsne tasks.")
    pct_token = format_token(f"{float(percentage):g}")
    return task_root / f"ot_tsne_eps_times_{eps_token}_pct_{pct_token}.npz"


def resolve_cache_output_path(dataset_path: Path, variant: str) -> Path:
    return dataset_path / f"joint_ot_tsne_sweep_embeddings_{variant}.npz"


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_common_args(
    *,
    data_path: str | Path,
    base_path: str | Path,
    max_cells_per_batch: int | None,
    subsample_random_state: int,
    embedding_random_state: int,
    n_epochs: int,
    binarize: bool,
    eot_t: int,
    eot_source: str,
    metric_k: int,
) -> list[str]:
    args = [
        "--data-path",
        str(data_path),
        "--base-path",
        str(base_path),
        "--subsample-random-state",
        str(int(subsample_random_state)),
        "--embedding-random-state",
        str(int(embedding_random_state)),
        "--n-epochs",
        str(int(n_epochs)),
        "--eot-t",
        str(int(eot_t)),
        "--eot-source",
        str(eot_source),
        "--metric-k",
        str(int(metric_k)),
    ]
    if max_cells_per_batch is not None:
        args.extend(["--max-cells-per-batch", str(int(max_cells_per_batch))])
    if binarize:
        args.append("--binarize")
    else:
        args.append("--weighted")
    return args


def build_python_command(python_executable: str, script_path: Path, args: list[str]) -> list[str]:
    return [python_executable, str(script_path), *args]


def run_task(args: argparse.Namespace) -> None:
    data, _, dataset_path = load_ent_analysis_data(
        data_path=args.data_path,
        base_path=args.base_path,
        max_cells_per_batch=args.max_cells_per_batch,
        random_state=args.subsample_random_state,
    )
    variant = bool_to_variant(args.binarize)
    task_root = resolve_task_root(
        dataset_path,
        variant=variant,
        max_cells_per_batch=args.max_cells_per_batch,
        subsample_random_state=args.subsample_random_state,
        embedding_random_state=args.embedding_random_state,
        n_epochs=args.n_epochs,
        eot_t=args.eot_t,
        eot_source=args.eot_source,
    )
    task_root.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.task_output
        if args.task_output is not None
        else resolve_task_output_path(
            task_root,
            task_kind=args.task_kind,
            eps_times=args.eps_times,
            percentage=args.percentage,
            eot_t=args.eot_t,
        )
    )

    if output_path.exists() and not args.force:
        print(f"Task output already exists at {output_path}; use --force to recompute.")
        return

    print(
        f"Using {ENT_ANALYSIS_DISPLAY_NAME} analysis data from {dataset_path} "
        f"with batch sizes {data.x1.shape[0]:,}/{data.x2.shape[0]:,}"
    )
    started = time.perf_counter()
    if args.task_kind == "eot":
        print(f"Computing paper-style EOT t={args.eot_t} from raw arrays for eps_times={args.eps_times:g}")
        embedding = compute_eot_embedding_from_arrays(
            data.x1,
            data.x2,
            t=args.eot_t,
            eps_times=args.eps_times,
        )
    else:
        print(
            "Computing joint OT-tSNE embedding "
            f"for eps_times={args.eps_times:g}, top {100.0 * args.percentage:g}% neighbors"
        )
        embedding = compute_ot_tsne_embedding_from_arrays(
            data.x1,
            data.x2,
            eps_times=args.eps_times,
            effective_k=None,
            percentage=args.percentage,
            binarize=args.binarize,
            random_state=args.embedding_random_state,
            n_epochs=args.n_epochs,
        )
    elapsed = time.perf_counter() - started

    np.savez_compressed(
        output_path,
        task_kind=np.asarray(args.task_kind),
        eps_times=np.asarray(float(args.eps_times), dtype=float),
        percentage=np.asarray(np.nan if args.percentage is None else float(args.percentage), dtype=float),
        embedding=np.asarray(embedding, dtype=float),
        variant=np.asarray(variant),
        binarize=np.asarray(bool(args.binarize)),
        eot_t=np.asarray(int(args.eot_t), dtype=int),
        eot_source=np.asarray(str(args.eot_source)),
        metric_k=np.asarray(int(args.metric_k), dtype=int),
        subsample_random_state=np.asarray(int(args.subsample_random_state), dtype=int),
        embedding_random_state=np.asarray(int(args.embedding_random_state), dtype=int),
        n_epochs=np.asarray(int(args.n_epochs), dtype=int),
        max_cells_per_batch=np.asarray(
            -1 if args.max_cells_per_batch is None else int(args.max_cells_per_batch),
            dtype=int,
        ),
        batch_sizes=np.asarray([data.x1.shape[0], data.x2.shape[0]], dtype=int),
        total_seconds=np.asarray(float(elapsed), dtype=float),
    )
    print(f"Saved {args.task_kind} task output to {output_path} in {elapsed:.2f}s")


def _load_task_embedding(
    path: Path,
    *,
    expected_kind: str,
    expected_variant: str,
    expected_eps_times: float,
    expected_percentage: float | None,
    expected_eot_t: int,
    expected_eot_source: str,
    expected_binarize: bool,
    expected_random_state: int,
    expected_n_epochs: int,
    expected_max_cells_per_batch: int | None,
    expected_joint_size: int,
) -> np.ndarray:
    with np.load(path) as cache:
        task_kind = str(cache["task_kind"].item())
        variant = str(cache["variant"].item())
        eps_times = float(np.asarray(cache["eps_times"]).item())
        percentage = float(np.asarray(cache["percentage"]).item())
        binarize = bool(np.asarray(cache["binarize"]).item())
        eot_t = int(np.asarray(cache["eot_t"]).item())
        eot_source = str(cache["eot_source"].item())
        embedding_random_state = int(np.asarray(cache["embedding_random_state"]).item())
        n_epochs = int(np.asarray(cache["n_epochs"]).item())
        stored_max_cells = int(np.asarray(cache["max_cells_per_batch"]).item())
        batch_sizes = np.asarray(cache["batch_sizes"], dtype=int)
        embedding = np.asarray(cache["embedding"], dtype=float)

    if task_kind != expected_kind:
        raise ValueError(f"{path} has task_kind={task_kind}, expected {expected_kind}.")
    if variant != expected_variant:
        raise ValueError(f"{path} has variant={variant}, expected {expected_variant}.")
    if not np.isclose(eps_times, expected_eps_times):
        raise ValueError(f"{path} has eps_times={eps_times}, expected {expected_eps_times}.")
    if expected_percentage is None:
        if not np.isnan(percentage):
            raise ValueError(f"{path} unexpectedly stores percentage={percentage}.")
    else:
        if not np.isclose(percentage, expected_percentage):
            raise ValueError(f"{path} has percentage={percentage}, expected {expected_percentage}.")
    if binarize != expected_binarize:
        raise ValueError(f"{path} has binarize={binarize}, expected {expected_binarize}.")
    if eot_t != expected_eot_t:
        raise ValueError(f"{path} has eot_t={eot_t}, expected {expected_eot_t}.")
    if eot_source != expected_eot_source:
        raise ValueError(f"{path} has eot_source={eot_source}, expected {expected_eot_source}.")
    if embedding_random_state != expected_random_state:
        raise ValueError(
            f"{path} has embedding_random_state={embedding_random_state}, "
            f"expected {expected_random_state}."
        )
    if n_epochs != expected_n_epochs:
        raise ValueError(f"{path} has n_epochs={n_epochs}, expected {expected_n_epochs}.")

    expected_max = -1 if expected_max_cells_per_batch is None else int(expected_max_cells_per_batch)
    if stored_max_cells != expected_max:
        raise ValueError(
            f"{path} has max_cells_per_batch={stored_max_cells}, expected {expected_max}."
        )
    if embedding.shape != (expected_joint_size, 2):
        raise ValueError(
            f"{path} has embedding shape {embedding.shape}, expected {(expected_joint_size, 2)}."
        )
    if batch_sizes.shape != (2,) or int(batch_sizes.sum()) != expected_joint_size:
        raise ValueError(
            f"{path} has batch_sizes={batch_sizes.tolist()}, expected a total of {expected_joint_size}."
        )

    return embedding


def aggregate_tasks(args: argparse.Namespace) -> None:
    data, _, dataset_path = load_ent_analysis_data(
        data_path=args.data_path,
        base_path=args.base_path,
        max_cells_per_batch=args.max_cells_per_batch,
        random_state=args.subsample_random_state,
    )
    variant = bool_to_variant(args.binarize)
    task_root = resolve_task_root(
        dataset_path,
        variant=variant,
        max_cells_per_batch=args.max_cells_per_batch,
        subsample_random_state=args.subsample_random_state,
        embedding_random_state=args.embedding_random_state,
        n_epochs=args.n_epochs,
        eot_t=args.eot_t,
        eot_source=args.eot_source,
    )
    cache_output_path = (
        resolve_cache_output_path(dataset_path, variant)
        if args.output_cache is None
        else Path(args.output_cache).expanduser()
    )

    eps_values = tuple(float(value) for value in args.eps_values)
    percentages = tuple(float(value) for value in args.percentages)
    expected_shape = (len(eps_values), len(percentages), data.joint_data.shape[0], 2)
    expected_eot_shape = (len(eps_values), data.joint_data.shape[0], 2)
    ot_tsne_embeddings = np.empty(expected_shape, dtype=float)
    eot_embeddings = np.empty(expected_eot_shape, dtype=float)

    missing_paths: list[Path] = []
    for eps_idx, eps_times in enumerate(eps_values):
        eot_path = resolve_task_output_path(
            task_root,
            task_kind="eot",
            eps_times=eps_times,
            eot_t=args.eot_t,
        )
        if not eot_path.exists():
            missing_paths.append(eot_path)
        else:
            eot_embeddings[eps_idx] = _load_task_embedding(
                eot_path,
                expected_kind="eot",
                expected_variant=variant,
                expected_eps_times=eps_times,
                expected_percentage=None,
                expected_eot_t=args.eot_t,
                expected_eot_source=args.eot_source,
                expected_binarize=args.binarize,
                expected_random_state=args.embedding_random_state,
                expected_n_epochs=args.n_epochs,
                expected_max_cells_per_batch=args.max_cells_per_batch,
                expected_joint_size=data.joint_data.shape[0],
            )

        for pct_idx, percentage in enumerate(percentages):
            ot_path = resolve_task_output_path(
                task_root,
                task_kind="ot-tsne",
                eps_times=eps_times,
                percentage=percentage,
                eot_t=args.eot_t,
            )
            if not ot_path.exists():
                missing_paths.append(ot_path)
                continue
            ot_tsne_embeddings[eps_idx, pct_idx] = _load_task_embedding(
                ot_path,
                expected_kind="ot-tsne",
                expected_variant=variant,
                expected_eps_times=eps_times,
                expected_percentage=percentage,
                expected_eot_t=args.eot_t,
                expected_eot_source=args.eot_source,
                expected_binarize=args.binarize,
                expected_random_state=args.embedding_random_state,
                expected_n_epochs=args.n_epochs,
                expected_max_cells_per_batch=args.max_cells_per_batch,
                expected_joint_size=data.joint_data.shape[0],
            )

    if missing_paths:
        preview = "\n".join(str(path) for path in missing_paths[:10])
        suffix = "" if len(missing_paths) <= 10 else f"\n... and {len(missing_paths) - 10} more"
        raise FileNotFoundError(
            "Cannot aggregate because some task outputs are missing:\n" + preview + suffix
        )

    cache_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_output_path,
        eps_values=np.asarray(eps_values, dtype=float),
        percentages=np.asarray(percentages, dtype=float),
        ot_tsne_embeddings=ot_tsne_embeddings,
        eot_embeddings=eot_embeddings,
        binarize=np.asarray(bool(args.binarize)),
        metric_k=np.asarray(int(args.metric_k), dtype=int),
        eot_t=np.asarray(int(args.eot_t), dtype=int),
        eot_source=np.asarray(str(args.eot_source)),
        random_state=np.asarray(int(args.embedding_random_state), dtype=int),
        n_epochs=np.asarray(int(args.n_epochs), dtype=int),
    )
    print(f"Saved aggregated notebook cache to {cache_output_path}")
    print(
        f"variant={variant}, eps_times={eps_values}, percentages={percentages}, "
        f"k={args.metric_k}, EOT source={args.eot_source}, "
        f"OT grid shape={ot_tsne_embeddings.shape}, EOT grid shape={eot_embeddings.shape}"
    )


def build_sbatch_command(
    *,
    job_name: str,
    log_path: Path,
    wrapped_command: list[str],
    chdir: Path,
    time_limit: str | None,
    partition: str | None,
    account: str | None,
    cpus_per_task: int | None,
    mem: str | None,
    extra_args: list[str],
) -> list[str]:
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        job_name,
        "--output",
        str(log_path),
        "--chdir",
        str(chdir),
    ]
    if time_limit is not None:
        command.extend(["--time", time_limit])
    if partition is not None:
        command.extend(["--partition", partition])
    if account is not None:
        command.extend(["--account", account])
    if cpus_per_task is not None:
        command.extend(["--cpus-per-task", str(int(cpus_per_task))])
    if mem is not None:
        command.extend(["--mem", mem])
    command.extend(extra_args)
    command.extend(["--wrap", shlex.join(wrapped_command)])
    return command


def submit_slurm(args: argparse.Namespace) -> None:
    _, _, dataset_path = load_ent_analysis_data(
        data_path=args.data_path,
        base_path=args.base_path,
        max_cells_per_batch=args.max_cells_per_batch,
        random_state=args.subsample_random_state,
    )
    variant = bool_to_variant(args.binarize)
    task_root = resolve_task_root(
        dataset_path,
        variant=variant,
        max_cells_per_batch=args.max_cells_per_batch,
        subsample_random_state=args.subsample_random_state,
        embedding_random_state=args.embedding_random_state,
        n_epochs=args.n_epochs,
        eot_t=args.eot_t,
        eot_source=args.eot_source,
    )
    task_root.mkdir(parents=True, exist_ok=True)

    output_dir = (
        dataset_path / "slurm_joint_ot_tsne_sweep"
        if args.output_dir is None
        else Path(args.output_dir).expanduser()
    )
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "slurm_jobs.csv"
    if args.overwrite_manifest and manifest_path.exists():
        manifest_path.unlink()

    python_executable = args.python_executable or sys.executable
    script_path = Path(__file__).resolve()
    common_args = build_common_args(
        data_path=args.data_path,
        base_path=args.base_path,
        max_cells_per_batch=args.max_cells_per_batch,
        subsample_random_state=args.subsample_random_state,
        embedding_random_state=args.embedding_random_state,
        n_epochs=args.n_epochs,
        binarize=args.binarize,
        eot_t=args.eot_t,
        eot_source=args.eot_source,
        metric_k=args.metric_k,
    )

    jobs: list[tuple[str, float, float | None]] = []
    for eps_times in args.eps_values:
        jobs.append(("eot", float(eps_times), None))
        for percentage in args.percentages:
            jobs.append(("ot-tsne", float(eps_times), float(percentage)))

    total_jobs = len(jobs)
    print(
        f"Preparing {total_jobs} Slurm jobs for {ENT_ANALYSIS_DISPLAY_NAME} "
        f"(variant={variant}, n_per_batch={args.max_cells_per_batch})."
    )
    for job_idx, (task_kind, eps_times, percentage) in enumerate(jobs, start=1):
        job_name = f"{args.slurm_job_prefix}-{job_idx:03d}"
        log_path = logs_dir / f"{job_name}.log"
        output_path = resolve_task_output_path(
            task_root,
            task_kind=task_kind,
            eps_times=eps_times,
            percentage=percentage,
            eot_t=args.eot_t,
        )

        task_args = [
            "run-task",
            "--task-kind",
            task_kind,
            "--eps-times",
            f"{eps_times:g}",
            "--task-output",
            str(output_path),
            *common_args,
        ]
        if percentage is not None:
            task_args.extend(["--percentage", f"{percentage:g}"])
        if args.force:
            task_args.append("--force")
        wrapped_command = build_python_command(python_executable, script_path, task_args)
        sbatch_command = build_sbatch_command(
            job_name=job_name,
            log_path=log_path,
            wrapped_command=wrapped_command,
            chdir=ROOT,
            time_limit=args.slurm_time,
            partition=args.slurm_partition,
            account=args.slurm_account,
            cpus_per_task=args.slurm_cpus_per_task,
            mem=args.slurm_mem,
            extra_args=args.slurm_extra_arg,
        )

        if args.dry_run:
            job_id = ""
            print(shlex.join(sbatch_command))
        else:
            result = subprocess.run(
                sbatch_command,
                check=True,
                capture_output=True,
                text=True,
            )
            job_id = result.stdout.strip()
            print(f"Submitted {job_name} as job {job_id}")

        append_csv_row(
            manifest_path,
            SLURM_MANIFEST_FIELDNAMES,
            {
                "job_kind": task_kind,
                "eps_times": f"{eps_times:g}",
                "percentage": "" if percentage is None else f"{percentage:g}",
                "job_id": job_id,
                "job_name": job_name,
                "output_path": str(output_path),
                "log_path": str(log_path),
                "command": shlex.join(wrapped_command),
            },
        )

    print(f"Wrote Slurm manifest to {manifest_path}")
    print(
        "After the jobs finish, combine them with:\n"
        + shlex.join(
            build_python_command(
                python_executable,
                script_path,
                [
                    "aggregate",
                    *common_args,
                    "--output-cache",
                    str(resolve_cache_output_path(dataset_path, variant)),
                    "--eps-values",
                    *(f"{value:g}" for value in args.eps_values),
                    "--percentages",
                    *(f"{value:g}" for value in args.percentages),
                ],
            )
        )
    )


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(ENT_ANALYSIS_DATA_DIR),
        help="Path to the ENT subset directory or a relative path resolvable from the repo.",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path(ENT_ANALYSIS_BASE_DIR),
        help="Fallback path to the base Yao 10x dataset directory.",
    )
    parser.add_argument(
        "--max-cells-per-batch",
        type=int,
        default=ENT_ANALYSIS_MAX_CELLS_PER_BATCH,
        help="Stratified sample size per batch. The notebook default is 5000.",
    )
    parser.add_argument(
        "--subsample-random-state",
        type=int,
        default=ENT_ANALYSIS_RANDOM_STATE,
        help="Random seed used for the stratified female/male subsampling.",
    )
    parser.add_argument(
        "--embedding-random-state",
        type=int,
        default=DEFAULT_EMBEDDING_RANDOM_STATE,
        help="Random seed used for the OT-tSNE initialization.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=DEFAULT_N_EPOCHS,
        help="Number of OT-tSNE optimization epochs.",
    )
    parser.add_argument(
        "--eot-t",
        type=int,
        default=DEFAULT_EOT_T,
        help="EOT singular-value power used for the reference embedding.",
    )
    parser.add_argument(
        "--eot-source",
        type=str,
        default=DEFAULT_EOT_SOURCE,
        help="Metadata tag stored in the aggregated cache.",
    )
    parser.add_argument(
        "--metric-k",
        type=int,
        default=DEFAULT_METRIC_K,
        help="Metadata field carried over from the notebook cache.",
    )
    parser.set_defaults(binarize=DEFAULT_BINARIZE)
    variant_group = parser.add_mutually_exclusive_group()
    variant_group.add_argument(
        "--binarize",
        dest="binarize",
        action="store_true",
        help="Use the notebook's binarized OT-tSNE graph variant.",
    )
    variant_group.add_argument(
        "--weighted",
        dest="binarize",
        action="store_false",
        help="Use the weighted OT-tSNE graph variant.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Yao 10x ENT joint OT-tSNE sweep outside the notebook, either one task at a "
            "time or as one Slurm job per sub-calculation."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_task_parser = subparsers.add_parser(
        "run-task",
        help="Compute one EOT or OT-tSNE sub-calculation and save it as a standalone task file.",
    )
    add_common_options(run_task_parser)
    run_task_parser.add_argument(
        "--task-kind",
        choices=("eot", "ot-tsne"),
        required=True,
        help="Whether to compute the paper-style EOT reference or one OT-tSNE embedding.",
    )
    run_task_parser.add_argument(
        "--eps-times",
        type=float,
        required=True,
        help="Notebook-style epsilon scaling value.",
    )
    run_task_parser.add_argument(
        "--percentage",
        type=float,
        default=None,
        help="Top-neighbor percentage in [0, 1]. Required for ot-tsne tasks.",
    )
    run_task_parser.add_argument(
        "--task-output",
        type=Path,
        default=None,
        help="Optional explicit output path for the task .npz file.",
    )
    run_task_parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if the task output already exists.",
    )

    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="Combine per-task .npz outputs back into the notebook cache file format.",
    )
    add_common_options(aggregate_parser)
    aggregate_parser.add_argument(
        "--eps-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_EPS_VALUES),
        help="eps_times values to aggregate.",
    )
    aggregate_parser.add_argument(
        "--percentages",
        nargs="+",
        type=float,
        default=list(DEFAULT_PERCENTAGES),
        help="Top-neighbor percentages to aggregate, expressed in [0, 1].",
    )
    aggregate_parser.add_argument(
        "--output-cache",
        type=Path,
        default=None,
        help="Optional output path for the aggregated notebook cache .npz file.",
    )

    submit_parser = subparsers.add_parser(
        "submit-slurm",
        help="Submit one Slurm job per EOT/OT-tSNE task for the notebook sweep.",
    )
    add_common_options(submit_parser)
    submit_parser.add_argument(
        "--eps-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_EPS_VALUES),
        help="eps_times values to submit.",
    )
    submit_parser.add_argument(
        "--percentages",
        nargs="+",
        type=float,
        default=list(DEFAULT_PERCENTAGES),
        help="Top-neighbor percentages to submit, expressed in [0, 1].",
    )
    submit_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for Slurm logs and the submission manifest.",
    )
    submit_parser.add_argument(
        "--python-executable",
        type=str,
        default=None,
        help="Python executable used inside each sbatch --wrap command. Defaults to the current interpreter.",
    )
    submit_parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Delete an existing Slurm manifest before writing a new one.",
    )
    submit_parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force through to each worker job.",
    )
    submit_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting them.",
    )
    submit_parser.add_argument(
        "--slurm-job-prefix",
        type=str,
        default="yao-ent-ottsne",
        help="Prefix used for submitted Slurm job names.",
    )
    submit_parser.add_argument(
        "--slurm-time",
        type=str,
        default=None,
        help="Optional sbatch --time value.",
    )
    submit_parser.add_argument(
        "--slurm-partition",
        type=str,
        default=None,
        help="Optional sbatch --partition value.",
    )
    submit_parser.add_argument(
        "--slurm-account",
        type=str,
        default=None,
        help="Optional sbatch --account value.",
    )
    submit_parser.add_argument(
        "--slurm-cpus-per-task",
        type=int,
        default=None,
        help="Optional sbatch --cpus-per-task value.",
    )
    submit_parser.add_argument(
        "--slurm-mem",
        type=str,
        default=None,
        help="Optional sbatch --mem value.",
    )
    submit_parser.add_argument(
        "--slurm-extra-arg",
        action="append",
        default=[],
        help="Additional raw sbatch argument. Repeat this flag to pass multiple values.",
    )

    args = parser.parse_args()
    if args.command == "run-task":
        if args.task_kind == "ot-tsne" and args.percentage is None:
            parser.error("--percentage is required when --task-kind=ot-tsne.")
        if args.task_kind == "eot" and args.percentage is not None:
            parser.error("--percentage should not be provided when --task-kind=eot.")
        if args.task_output is not None:
            args.task_output = args.task_output.expanduser()
    if hasattr(args, "output_cache") and args.output_cache is not None:
        args.output_cache = args.output_cache.expanduser()
    if hasattr(args, "output_dir") and args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser()
    if hasattr(args, "eps_values"):
        args.eps_values = tuple(float(value) for value in args.eps_values)
    if hasattr(args, "percentages"):
        args.percentages = tuple(float(value) for value in args.percentages)
    if hasattr(args, "task_output") and args.task_output is not None:
        args.task_output = Path(args.task_output)
    if args.max_cells_per_batch is not None and args.max_cells_per_batch <= 0:
        parser.error("--max-cells-per-batch must be positive when provided.")
    if args.n_epochs <= 0:
        parser.error("--n-epochs must be positive.")
    if args.metric_k <= 0:
        parser.error("--metric-k must be positive.")
    if hasattr(args, "eps_times") and args.eps_times <= 0:
        parser.error("--eps-times must be positive.")
    if hasattr(args, "eps_values") and any(value <= 0 for value in args.eps_values):
        parser.error("--eps-values must all be positive.")
    if hasattr(args, "percentages") and any(value <= 0 or value > 1 for value in args.percentages):
        parser.error("--percentages must lie in (0, 1].")
    if hasattr(args, "percentage") and args.percentage is not None and (
        args.percentage <= 0 or args.percentage > 1
    ):
        parser.error("--percentage must lie in (0, 1].")
    if hasattr(args, "slurm_cpus_per_task") and args.slurm_cpus_per_task is not None:
        if args.slurm_cpus_per_task <= 0:
            parser.error("--slurm-cpus-per-task must be positive.")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "run-task":
        if args.task_output is not None:
            args.task_output.parent.mkdir(parents=True, exist_ok=True)
        run_task(args)
        return
    if args.command == "aggregate":
        aggregate_tasks(args)
        return
    if args.command == "submit-slurm":
        submit_slurm(args)
        return
    raise ValueError(f"Unsupported command {args.command!r}.")


if __name__ == "__main__":
    main()
