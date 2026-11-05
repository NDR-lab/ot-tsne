"""Build immutable drawing inputs for each figure family from the registry and styles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ne_aligned_utils.paper import layout, registry
from ne_aligned_utils.paper.figures.qualitative import QualitativeComposer, QualitativeStyle
from ne_aligned_utils.paper.figures.ranks import RankComposer, RankStyle, TopKComposer
from ne_aligned_utils.paper.rendering import legends, styles
from ne_aligned_utils.paper.rendering.context import LegendConfig, MethodPresentation, RenderContext


def method_presentation(key: str) -> MethodPresentation:
    """Scatter and legend-glyph values for one registered method."""
    entry = registry.resolve(key)
    return MethodPresentation(
        entry.label,
        entry.legend_label,
        entry.family,
        entry.subfamily,
        entry.scatter_kwargs(),
        key=key,
        tool=entry.tool,
        legend_glyph=legends.handle_kwargs(
            marker=entry.marker,
            color=entry.color,
            size=entry.size,
            label=entry.label,
            edge=(entry.edge_color, entry.edge_width),
            alpha=1.0,
            unfilled=entry.unfilled,
        ),
    )


def benchmark_render_context(
    output_root: Path,
    *,
    method_keys: Sequence[str],
    datasets: Sequence[str],
    dataset_labels: Mapping[str, str] | None = None,
) -> RenderContext:
    """Presentation for the metric grids; represents exactly ``method_keys``, selects nothing."""
    methods = {key: method_presentation(key) for key in method_keys}
    return RenderContext(
        methods=methods,
        method_order=tuple(registry.method_order(methods)),
        family_order=registry.FAMILY_ORDER,
        family_labels=registry.FAMILY_LABELS,
        dataset_order=tuple(datasets),
        dataset_labels=styles.GRID_STRIP_LABELS if dataset_labels is None else dataset_labels,
        legend=LegendConfig(
            1,
            True,
            registry.INTEGRATION_LEGEND_ORDER,
            dimension_size=registry.METHODS["scanorama-50"].size,
        ),
        output_root=output_root,
    )


def short_labels() -> dict[str, str]:
    return {key: entry.short_label for key, entry in registry.METHODS.items()}


def rank_composer() -> RankComposer:
    return RankComposer(
        style=RankStyle(
            rank_ramp=styles.RANK_RAMP,
            rank_mark_size=styles.RANK_MARK_SIZE,
            rank_ramp_offset_pt=styles.RANK_RAMP_OFFSET_PT,
            rank_ramp_gap_pt=styles.RANK_RAMP_GAP_PT,
            rank_ramp_thickness_pt=styles.RANK_RAMP_THICKNESS_PT,
            rank_ramp_tick_pad_pt=styles.RANK_RAMP_TICK_PAD_PT,
            rank_panel_width=styles.RANK_PANEL_WIDTH,
            rank_panel_gap=styles.RANK_PANEL_GAP,
            rank_side_blocks=styles.RANK_SIDE_BLOCKS,
            rank_value_format=styles.RANK_VALUE_FORMAT,
            rank_label_pad_pt=styles.RANK_LABEL_PAD_PT,
            rank_edge_pad_in=styles.RANK_EDGE_PAD_IN,
        ),
        method_labels=short_labels(),
        block_titles=styles.RANK_BLOCKS,
    )


def top_k_composer() -> TopKComposer:
    return TopKComposer(
        method_labels=short_labels(),
        method_order=tuple(registry.method_order()),
        palette=styles.DATASET_RAMP,
        dataset_labels=registry.DATASET_LABELS,
        parts=styles.BAR_PARTS,
        label_clearance=styles.BAR_LABEL_CLEARANCE,
        panel_fraction=styles.BAR_PANEL_FRACTION,
        legend_loc=styles.BAR_LEGEND_LOC,
        legend_pad=styles.BAR_LEGEND_PAD,
    )


def qualitative_composer() -> QualitativeComposer:
    return QualitativeComposer(
        style=QualitativeStyle(
            qualitative_rows=styles.QUALITATIVE_ROWS,
            qualitative_margins=styles.QUALITATIVE_MARGINS,
            qualitative_batch_colours=styles.QUALITATIVE_BATCH_COLOURS,
            qualitative_rule_at=styles.QUALITATIVE_RULE_AT,
            qualitative_legend_gap=styles.QUALITATIVE_LEGEND_GAP,
            qualitative_batch_labels=styles.QUALITATIVE_BATCH_LABELS,
            qualitative_handletextpad=styles.QUALITATIVE_HANDLETEXTPAD,
            qualitative_columnspacing=styles.QUALITATIVE_COLUMNSPACING,
            qualitative_group_band=styles.QUALITATIVE_GROUP_BAND,
            qualitative_group_lift=styles.QUALITATIVE_GROUP_LIFT,
            qualitative_group_tick=styles.QUALITATIVE_GROUP_TICK,
            qualitative_group_text_gap=styles.QUALITATIVE_GROUP_TEXT_GAP,
            qualitative_group_pt=styles.QUALITATIVE_GROUP_PT,
            point_ref_size=styles.POINT_REF_SIZE,
            point_ref_n=styles.POINT_REF_N,
            panel_trim=styles.PANEL_TRIM,
            seed=layout.SEED,
        ),
        method_labels=short_labels(),
    )


__all__ = [
    "benchmark_render_context",
    "method_presentation",
    "qualitative_composer",
    "rank_composer",
    "top_k_composer",
]
