from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runtime_benchmarking import create_ring_embedding


DETAIL_FIELDNAMES = [
    "n0",
    "n1",
    "seed",
    "epsilon",
    "top_pct",
    "median_squared_distance",
    "graph_nnz",
    "graph_density",
    "data_seconds",
    "sinkhorn_seconds",
    "four_block_seconds",
    "sparsify_seconds",
    "tsne_seconds",
    "total_seconds",
]

SUMMARY_FIELDNAMES = [
    "n0",
    "n1",
    "epsilon",
    "top_pct",
    "num_seeds",
    "data_seconds_mean",
    "data_seconds_sd",
    "sinkhorn_seconds_mean",
    "sinkhorn_seconds_sd",
    "four_block_seconds_mean",
    "four_block_seconds_sd",
    "sparsify_seconds_mean",
    "sparsify_seconds_sd",
    "tsne_seconds_mean",
    "tsne_seconds_sd",
    "total_seconds_mean",
    "total_seconds_sd",
    "graph_density_mean",
    "graph_density_sd",
]


@dataclass(frozen=True)
class GroupSpec:
    n0: int
    n1: int
    epsilon: float
    top_pct: float


def parse_pair(pair_text: str) -> tuple[int, int]:
    parts = [part.strip() for part in pair_text.split(",")]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid pair '{pair_text}'. Expected the format 'n0,n1', for example '1000,5000'."
        )
    n0, n1 = (int(part) for part in parts)
    if n0 <= 0 or n1 <= 0:
        raise ValueError(f"Pair values must be positive, got ({n0}, {n1}).")
    return n0, n1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep ring runtime benchmarks over sample-size pairs, epsilons, sparsification "
            "percentages, and seeds."
        )
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="List of sample-size pairs in the form 'n0,n1'. Example: --pairs 100,500 500,2500",
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        required=True,
        help="Explicit epsilon values to benchmark.",
    )
    parser.add_argument(
        "--percentages",
        nargs="+",
        type=float,
        required=True,
        help="Top-neighbor percentages to benchmark, given in percent units like 1 5 10.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help=(
            "Seeds to benchmark. Each seed is used both for ring data generation and t-SNE "
            "initialization."
        ),
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=200,
        help="Number of t-SNE optimization epochs for each run.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-12,
        help="Sinkhorn stopping tolerance.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=int(1e4),
        help="Maximum number of Sinkhorn iterations.",
    )
    parser.add_argument(
        "--save-csv",
        type=Path,
        default=None,
        help="Path for per-run results. Rows are appended immediately after each completed run.",
    )
    parser.add_argument(
        "--save-summary-csv",
        type=Path,
        default=None,
        help="Path for per-group summary results. Rows are appended when each group finishes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Base directory for worker-specific outputs. Helpful for tmux launches; each worker "
            "gets its own detail CSV, summary CSV, and log file."
        ),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete this worker's existing CSV/log files before starting.",
    )
    parser.add_argument(
        "--launch-tmux",
        action="store_true",
        help="Launch detached tmux worker sessions instead of running the sweep in the current shell.",
    )
    parser.add_argument(
        "--tmux-splits",
        type=int,
        default=1,
        help="Number of detached tmux worker sessions to launch.",
    )
    parser.add_argument(
        "--tmux-session-prefix",
        type=str,
        default="runtime-sweep",
        help="Prefix used for detached tmux session names.",
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    args.pairs = [parse_pair(pair_text) for pair_text in args.pairs]
    if any(epsilon <= 0 for epsilon in args.epsilons):
        raise ValueError(f"All epsilons must be positive, got {args.epsilons}.")
    if any(percentage <= 0 or percentage > 100 for percentage in args.percentages):
        raise ValueError(
            f"All percentages must lie in (0, 100], got {args.percentages}."
        )
    if args.n_epochs <= 0:
        raise ValueError(f"n_epochs must be positive, got {args.n_epochs}.")
    if args.tmux_splits <= 0:
        raise ValueError(f"tmux_splits must be positive, got {args.tmux_splits}.")
    if args.num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {args.num_workers}.")
    if args.worker_index < 0 or args.worker_index >= args.num_workers:
        raise ValueError(
            f"worker_index must satisfy 0 <= worker_index < num_workers, got "
            f"worker_index={args.worker_index}, num_workers={args.num_workers}."
        )
    if args.launch_tmux and args.output_dir is None:
        raise ValueError("--launch-tmux requires --output-dir so each worker can write separate files.")

    return args


def build_group_specs(args: argparse.Namespace) -> list[GroupSpec]:
    groups = []
    for n0, n1 in args.pairs:
        for epsilon in args.epsilons:
            for percentage in args.percentages:
                groups.append(
                    GroupSpec(
                        n0=int(n0),
                        n1=int(n1),
                        epsilon=float(epsilon),
                        top_pct=float(percentage),
                    )
                )
    return groups


def select_worker_groups(groups: list[GroupSpec], worker_index: int, num_workers: int) -> list[GroupSpec]:
    return [group for group_idx, group in enumerate(groups) if group_idx % num_workers == worker_index]


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def remove_if_exists(path: Path | None) -> None:
    if path is not None and path.exists():
        path.unlink()


def worker_tag(worker_index: int, num_workers: int) -> str:
    width = max(2, len(str(num_workers - 1)))
    return f"worker_{worker_index:0{width}d}_of_{num_workers:0{width}d}"


def resolve_worker_paths(
    args: argparse.Namespace,
    *,
    worker_index: int | None = None,
    num_workers: int | None = None,
) -> tuple[Path | None, Path | None, Path | None]:
    idx = args.worker_index if worker_index is None else int(worker_index)
    total = args.num_workers if num_workers is None else int(num_workers)

    detail_path = args.save_csv
    summary_path = args.save_summary_csv
    log_path = None

    if args.output_dir is not None:
        tag = worker_tag(idx, total)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if detail_path is None:
            detail_path = output_dir / f"{tag}_detail.csv"
        if summary_path is None:
            summary_path = output_dir / f"{tag}_summary.csv"
        log_path = output_dir / f"{tag}.log"

    return detail_path, summary_path, log_path


def make_detail_row(group: GroupSpec, seed: int, run) -> dict[str, float | int]:
    graph_nnz = int(np.count_nonzero(run.graph))
    return {
        "n0": int(group.n0),
        "n1": int(group.n1),
        "seed": int(seed),
        "epsilon": float(run.epsilon),
        "top_pct": float(group.top_pct),
        "median_squared_distance": float(run.median_squared_distance),
        "graph_nnz": graph_nnz,
        "graph_density": float(graph_nnz / run.graph.size),
        "data_seconds": float(run.data_seconds),
        "sinkhorn_seconds": float(run.sinkhorn_seconds),
        "four_block_seconds": float(run.four_block_seconds),
        "sparsify_seconds": float(run.sparsify_seconds),
        "tsne_seconds": float(run.tsne_seconds),
        "total_seconds": float(run.total_seconds),
    }


def build_summary_row(group: GroupSpec, rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    summary_row: dict[str, float | int] = {
        "n0": int(group.n0),
        "n1": int(group.n1),
        "epsilon": float(group.epsilon),
        "top_pct": float(group.top_pct),
        "num_seeds": int(len(rows)),
    }
    for metric_name in (
        "data_seconds",
        "sinkhorn_seconds",
        "four_block_seconds",
        "sparsify_seconds",
        "tsne_seconds",
        "total_seconds",
        "graph_density",
    ):
        values = np.asarray([float(row[metric_name]) for row in rows], dtype=float)
        summary_row[f"{metric_name}_mean"] = float(values.mean())
        summary_row[f"{metric_name}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return summary_row


def print_summary(summary_rows: list[dict[str, float | int]]) -> None:
    print("\nAcross-seed summary")
    for row in summary_rows:
        print(
            "  n0={n0}, n1={n1}, epsilon={epsilon:g}, top_pct={top_pct:g}, seeds={num_seeds}: "
            "sinkhorn_mean={sinkhorn_seconds_mean:.4f}s, tsne_mean={tsne_seconds_mean:.4f}s, "
            "total_mean={total_seconds_mean:.4f}s".format(**row)
        )


def run_worker(args: argparse.Namespace) -> list[dict[str, float | int]]:
    all_groups = build_group_specs(args)
    assigned_groups = select_worker_groups(all_groups, args.worker_index, args.num_workers)
    detail_path, summary_path, _ = resolve_worker_paths(args)

    if args.overwrite_output:
        remove_if_exists(detail_path)
        remove_if_exists(summary_path)

    total_runs = len(assigned_groups) * len(args.seeds)
    if total_runs == 0:
        print(
            f"No work assigned to worker {args.worker_index}/{args.num_workers - 1}. "
            "Nothing to do."
        )
        return []

    print(
        f"Worker {args.worker_index + 1}/{args.num_workers}: "
        f"{len(assigned_groups)} group(s), {total_runs} run(s)."
    )
    if detail_path is not None:
        print(f"  detail_csv={detail_path}")
    if summary_path is not None:
        print(f"  summary_csv={summary_path}")

    summary_rows: list[dict[str, float | int]] = []
    run_idx = 0

    for group_idx, group in enumerate(assigned_groups, start=1):
        group_rows: list[dict[str, float | int]] = []
        print(
            f"\nGroup {group_idx}/{len(assigned_groups)}: "
            f"n0={group.n0}, n1={group.n1}, epsilon={group.epsilon:g}, top_pct={group.top_pct:g}"
        )
        for seed in args.seeds:
            run_idx += 1
            print(f"  Run {run_idx}/{total_runs}: seed={seed}")
            run = create_ring_embedding(
                n0=group.n0,
                n1=group.n1,
                seed=int(seed),
                tsne_seed=int(seed),
                n_epochs=args.n_epochs,
                top_pct=group.top_pct,
                epsilon=group.epsilon,
                eps_times=1.0,
                tol=args.tol,
                max_iter=args.max_iter,
            )

            row = make_detail_row(group, int(seed), run)
            group_rows.append(row)
            if detail_path is not None:
                append_csv_row(detail_path, DETAIL_FIELDNAMES, row)

            print(
                "    sinkhorn={:.4f}s | tsne={:.4f}s | total={:.4f}s".format(
                    row["sinkhorn_seconds"],
                    row["tsne_seconds"],
                    row["total_seconds"],
                )
            )

        summary_row = build_summary_row(group, group_rows)
        summary_rows.append(summary_row)
        if summary_path is not None:
            append_csv_row(summary_path, SUMMARY_FIELDNAMES, summary_row)

    return summary_rows


def quote_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def make_worker_command(
    args: argparse.Namespace,
    *,
    worker_index: int,
    num_workers: int,
    detail_path: Path | None,
    summary_path: Path | None,
) -> list[str]:
    script_path = Path(__file__).resolve()
    python_bin = sys.executable

    command = [
        str(python_bin),
        str(script_path),
        "--pairs",
        *[f"{n0},{n1}" for n0, n1 in args.pairs],
        "--epsilons",
        *[f"{float(epsilon):g}" for epsilon in args.epsilons],
        "--percentages",
        *[f"{float(percentage):g}" for percentage in args.percentages],
        "--seeds",
        *[str(int(seed)) for seed in args.seeds],
        "--n-epochs",
        str(int(args.n_epochs)),
        "--tol",
        f"{float(args.tol):.12g}",
        "--max-iter",
        str(int(args.max_iter)),
        "--worker-index",
        str(int(worker_index)),
        "--num-workers",
        str(int(num_workers)),
    ]

    if detail_path is not None:
        command.extend(["--save-csv", str(detail_path)])
    if summary_path is not None:
        command.extend(["--save-summary-csv", str(summary_path)])
    if args.output_dir is not None:
        command.extend(["--output-dir", str(Path(args.output_dir).resolve())])
    if args.overwrite_output:
        command.append("--overwrite-output")

    return command


def launch_tmux_workers(args: argparse.Namespace) -> None:
    all_groups = build_group_specs(args)
    if not all_groups:
        raise ValueError("No groups were generated from the requested sweep.")

    worker_count = min(int(args.tmux_splits), len(all_groups))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "tmux_workers.csv"

    if args.overwrite_output:
        remove_if_exists(manifest_path)

    manifest_rows: list[dict[str, str | int]] = []
    for worker_index in range(worker_count):
        detail_path, summary_path, log_path = resolve_worker_paths(
            args,
            worker_index=worker_index,
            num_workers=worker_count,
        )
        session_name = f"{args.tmux_session_prefix}-{worker_index:02d}"

        has_session = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if has_session.returncode == 0:
            raise RuntimeError(
                f"tmux session '{session_name}' already exists. Use a different "
                f"--tmux-session-prefix or close that session first."
            )

        worker_command = make_worker_command(
            args,
            worker_index=worker_index,
            num_workers=worker_count,
            detail_path=detail_path,
            summary_path=summary_path,
        )
        shell_command = f"cd {shlex.quote(str(Path.cwd()))} && {quote_join(worker_command)}"
        if log_path is not None:
            if args.overwrite_output:
                remove_if_exists(log_path)
            shell_command = f"{shell_command} >> {shlex.quote(str(log_path))} 2>&1"

        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, "bash", "-lc", shell_command],
            check=True,
        )

        manifest_row = {
            "worker_index": worker_index,
            "session_name": session_name,
            "detail_csv": "" if detail_path is None else str(detail_path),
            "summary_csv": "" if summary_path is None else str(summary_path),
            "log_path": "" if log_path is None else str(log_path),
            "group_count": len(select_worker_groups(all_groups, worker_index, worker_count)),
        }
        manifest_rows.append(manifest_row)
        append_csv_row(
            manifest_path,
            ["worker_index", "session_name", "detail_csv", "summary_csv", "log_path", "group_count"],
            manifest_row,
        )

    print(f"Launched {worker_count} detached tmux session(s).")
    print(f"Manifest: {manifest_path}")
    print("Sessions:")
    for row in manifest_rows:
        print(
            f"  {row['session_name']}: detail={row['detail_csv']}, "
            f"summary={row['summary_csv']}, log={row['log_path']}"
        )
    print("\nUseful commands:")
    print(f"  tmux ls | grep {shlex.quote(args.tmux_session_prefix)}")
    print(f"  tmux attach -t {args.tmux_session_prefix}-00")


def main() -> None:
    args = parse_args()
    if args.launch_tmux:
        launch_tmux_workers(args)
        return

    summary_rows = run_worker(args)
    if summary_rows:
        print_summary(summary_rows)


if __name__ == "__main__":
    main()
