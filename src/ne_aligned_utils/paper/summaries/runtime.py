"""Runtime summaries and validation for the selected canonical timing rows."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ne_aligned_utils.paper import registry
from ne_aligned_utils.paper.rendering import runtime_drawing
from ne_aligned_utils.paper.rendering.runtime import (
    FAMILY_LABELS,
    PHASE_FAMILIES,
    RuntimeRenderContext,
)
from ne_aligned_utils.paper.summaries import task_timing

#: Largest relative gap between median phases and median total before falling back to one run.
PHASE_TOLERANCE = 0.05

SUMMARY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "dataset_label",
    "n_obs",
    "n_features_input",
    "task_kind",
    "method_key",
    "method_label",
    "family",
    "family_label",
    "backend",
    "seed",
    "rel_reg",
    "effective_k",
    "timing_schema",
    "time_total",
    "time_pre_embedding",
    "time_embedding",
    "time_wall_outer",
    "time_excluded_overhead",
    "total_seconds",
    "threads_requested",
    "n_jobs",
    "device",
    "blas_name",
    "slurm_cpus_per_task",
    "env_fingerprint",
    "hostname",
    "representation_reused",
    "included_in_plot",
    "exclusion_reason",
    "task_path",
)


def annotate(frame: pd.DataFrame, *, context: RuntimeRenderContext) -> pd.DataFrame:
    """Add panel, label and inclusion columns; a method outside every panel stays visible."""
    out = frame.copy()
    family_by_method = {
        key: family for family, members in context.families.items() for key in members
    }
    out["family"] = out["method_key"].map(family_by_method)
    out["included_in_plot"] = out["family"].notna()
    out["exclusion_reason"] = np.where(out["included_in_plot"], "", "not in a runtime panel")
    out["family_label"] = out["family"].map(FAMILY_LABELS)
    out["method_label"] = [registry.resolve(key).label for key in out["method_key"]]
    out["dataset_label"] = out["dataset"].map(context.dataset_labels).fillna(out["dataset"])
    return out


def family_of(key: str, *, context: RuntimeRenderContext) -> str:
    for family, members in context.families.items():
        if key in members:
            return family
    raise KeyError(f"{key!r} is not in any runtime panel")


def ordered_methods(
    family: str, gmeans: dict[str, float], *, context: RuntimeRenderContext
) -> list[str]:
    """Panel rows, top first: fastest first, except declared panels and 3b (mirrors 3a)."""
    members = list(context.families[family])
    if family in context.declared_order_families:
        return members
    if family == "3b":
        partner = {context.method_tools[key]: key for key in members}
        mirrored = [
            partner[context.method_tools[key]]
            for key in ordered_methods("3a", gmeans, context=context)
            if context.method_tools[key] in partner
        ]
        return mirrored + [key for key in members if key not in mirrored]
    return sorted(members, key=lambda key: (gmeans.get(key, math.inf), key))


def dataset_medians(frame: pd.DataFrame, *, context: RuntimeRenderContext) -> pd.DataFrame:
    """Median of each timing field per (method, dataset) across seeds, with the min/max spread."""
    included = frame.loc[frame["included_in_plot"]]
    grouped = included.groupby(["method_key", "dataset"], as_index=False, dropna=False).agg(
        time_total=("time_total", "median"),
        time_total_min=("time_total", "min"),
        time_total_max=("time_total", "max"),
        time_pre_embedding=("time_pre_embedding", "median"),
        time_embedding=("time_embedding", "median"),
        n_runs=("time_total", "size"),
        n_obs=("n_obs", "max"),
        n_features_input=("n_features_input", "max"),
        timing_schema=("timing_schema", "first"),
    )
    keys = grouped["method_key"].astype(str)
    grouped["family"] = [family_of(key, context=context) for key in keys]
    grouped["method_label"] = [context.method_labels[key] for key in keys]
    return grouped


def _geometric_mean(values) -> float:
    array = np.asarray([float(v) for v in values], dtype=float)
    array = array[np.isfinite(array) & (array > 0.0)]
    if array.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(array))))


def geometric_means(medians: pd.DataFrame) -> dict[str, float]:
    """Geometric mean of each method's dataset medians; a sort key only, never drawn."""
    return {
        str(key): _geometric_mean(group["time_total"])
        for key, group in medians.groupby("method_key")
    }


def panel_inputs(
    frame: pd.DataFrame, *, datasets: tuple[str, ...], context: RuntimeRenderContext
) -> dict:
    """Medians, shared axis limits, dataset order (by size) and row order for every panel."""
    medians = dataset_medians(frame, context=context)
    gmeans = geometric_means(medians)
    sizes = {
        str(dataset): int(group["n_obs"].max()) for dataset, group in medians.groupby("dataset")
    }
    return dict(
        medians=medians,
        xlim=runtime_drawing.axis_limits(medians, phase_families=PHASE_FAMILIES),
        sizes=sizes,
        datasets=runtime_drawing.order_by_size(tuple(datasets), sizes),
        method_order={
            family: ordered_methods(family, gmeans, context=context)
            for family in context.family_order
        },
    )


def ot_breakdown_table(
    frame: pd.DataFrame,
    *,
    datasets: tuple[str, ...],
    context: RuntimeRenderContext,
    method: str = "ot",
) -> pd.DataFrame:
    """Median OT and t-SNE durations per dataset, with spread and a phase-consistency check.

    When the two phase medians disagree with the median total by more than
    :data:`PHASE_TOLERANCE`, both phases come from the single run closest to the median total.
    """
    included = frame.loc[frame["included_in_plot"] & (frame["method_key"] == method)]
    if included.empty:
        raise ValueError(f"no plotted runs for {method!r}; nothing to break down.")
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        group = included.loc[included["dataset"] == dataset]
        if group.empty:
            continue
        total = group["time_total"].to_numpy(dtype=float)
        pre = group["time_pre_embedding"].to_numpy(dtype=float)
        emb = group["time_embedding"].to_numpy(dtype=float)
        median_total = float(np.median(total))
        median_pre = float(np.median(pre))
        median_emb = float(np.median(emb))
        disagreement = abs(median_pre + median_emb - median_total) / median_total
        source = "per-phase medians"
        if disagreement > PHASE_TOLERANCE:
            pick = int(np.argmin(np.abs(total - median_total)))
            median_pre, median_emb = float(pre[pick]), float(emb[pick])
            source = "median-total run"
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": runtime_drawing.dataset_label(
                    dataset, labels=context.dataset_labels
                ),
                "n_obs": int(group["n_obs"].max()),
                "n_runs": int(len(group)),
                "time_total": median_total,
                "time_total_min": float(total.min()),
                "time_total_max": float(total.max()),
                "time_pre_embedding": median_pre,
                "time_pre_embedding_min": float(pre.min()),
                "time_pre_embedding_max": float(pre.max()),
                "time_embedding": median_emb,
                "time_embedding_min": float(emb.min()),
                "time_embedding_max": float(emb.max()),
                "phase_disagreement": disagreement,
                "median_source": source,
            }
        )
    return pd.DataFrame(rows).sort_values("n_obs").reset_index(drop=True)


def environment_hosts(root: Path) -> dict[str, str]:
    """Environment fingerprint -> hostname, from every sidecar under ``root``."""
    hosts: dict[str, str] = {}
    for path in Path(root).rglob("environment_*.json"):
        fingerprint = path.stem.removeprefix("environment_")
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        hosts[fingerprint] = str(record.get("hostname", ""))
    return hosts


def validate(
    frame: pd.DataFrame,
    *,
    root: Path,
    datasets: tuple[str, ...],
    context: RuntimeRenderContext,
) -> list[str]:
    """Report lines on coverage, environments, totals and inputs.

    Raises only when plotted runs disagree on a resource field: those runtimes would compare
    machines rather than methods.
    """
    lines: list[str] = []
    plotted = frame.loc[frame["included_in_plot"]]

    lines.append(f"scanned {len(frame)} task files under {root}")
    lines.append(f"  {len(plotted)} included in the figures, {len(frame) - len(plotted)} excluded")
    reasons = frame.loc[~frame["included_in_plot"], "exclusion_reason"].value_counts()
    for reason, count in reasons.items():
        lines.append(f"    {count:5d}  {reason}")

    lines.append("")
    lines.append("resource fields (all plotted runs must agree):")
    for key in ("threads_requested", "n_jobs", "device", "blas_name", "slurm_cpus_per_task"):
        if key not in plotted.columns:
            lines.append(f"  {key}: absent from the task files")
            continue
        values = sorted({str(v) for v in plotted[key].tolist()})
        lines.append(f"  {key}: {values}")
        if len(values) > 1 and key in task_timing.RESOURCE_KEYS:
            raise task_timing.RuntimeComparabilityError(
                f"the plotted rows disagree on {key!r} ({values}), so their runtimes were not "
                "produced under comparable resources. Select one allocation rather than pooling."
            )

    hosts = environment_hosts(root)
    fingerprints = sorted({str(v) for v in plotted["env_fingerprint"].tolist()})
    resolved = [f for f in fingerprints if f in hosts]
    hostnames = sorted({hosts[f] for f in resolved if hosts[f]})
    lines.append("")
    lines.append(
        f"environment: {len(fingerprints)} distinct fingerprints, "
        f"{len(resolved)} with a sidecar, across {len(hostnames)} hosts"
    )
    lines.append(f"  hosts: {hostnames}")
    unresolved = [f for f in fingerprints if f not in hosts]
    if unresolved:
        lines.append(f"  WARNING no sidecar found for: {unresolved}")

    lines.append("")
    expected = [key for family in context.family_order for key in context.families[family]]
    lines.append(f"coverage: expecting {len(expected)} methods x {len(datasets)} datasets")
    counts = plotted.groupby(["method_key", "dataset"]).size()
    gaps: list[str] = []
    seed_oddities: list[str] = []
    for key in expected:
        for dataset in datasets:
            n = int(counts.get((key, dataset), 0))
            if n == 0:
                gaps.append(f"{key} x {dataset}")
            elif n != 3:
                seed_oddities.append(f"{key} x {dataset}: {n} runs")
    lines.append(f"  missing cells: {gaps if gaps else 'none'}")
    lines.append(f"  cells not at 3 seeds: {seed_oddities if seed_oddities else 'none'}")

    lines.append("")
    nonfinite = int((~np.isfinite(frame["time_total"].to_numpy(dtype=float))).sum())
    lines.append(f"time_total: {nonfinite} non-finite of {len(frame)} scanned")
    pre = plotted["time_pre_embedding"].to_numpy(dtype=float)
    emb = plotted["time_embedding"].to_numpy(dtype=float)
    total = plotted["time_total"].to_numpy(dtype=float)
    both = np.isfinite(pre) & np.isfinite(emb)
    disagreement = np.zeros_like(total)
    disagreement[both] = np.abs(pre[both] + emb[both] - total[both]) / total[both]
    n_bad = int((disagreement > PHASE_TOLERANCE).sum())
    lines.append(
        f"phase additivity: {int(both.sum())} plotted runs carry both phases; "
        f"{n_bad} exceed {PHASE_TOLERANCE:.0%}"
        + (f"; worst {disagreement.max():.2%}" if both.any() else "")
    )
    if n_bad:
        offenders = plotted.loc[disagreement > PHASE_TOLERANCE, ["method_key", "dataset", "seed"]]
        for row in offenders.to_dict("records"):
            lines.append(f"    {row['method_key']} {row['dataset']} seed {row['seed']}")

    lines.append("")
    lines.append("input width per dataset (gene-space tools read genes, the rest 50 PCs):")
    for dataset in datasets:
        subset = plotted.loc[plotted["dataset"] == dataset]
        widths = sorted({int(v) for v in subset["n_features_input"].tolist()})
        n_obs = sorted({int(v) for v in subset["n_obs"].tolist()})
        if len(widths) == 1:
            lines.append(f"  {dataset}: n_obs={n_obs[0]}, n_features_input={widths[0]}")
            continue
        lines.append(f"  {dataset}: n_obs={n_obs}, n_features_input={widths}")
        for width in widths:
            members = sorted(
                {str(k) for k in subset.loc[subset["n_features_input"] == width, "method_key"]}
            )
            lines.append(f"      {width:5d} features: {members}")
    return lines
