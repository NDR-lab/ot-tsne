"""Shared legend semantics tested directly, without retired assembly variants."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ne_aligned_utils.paper.rendering import legends, metrics
from ne_aligned_utils.paper.rendering.presentation import benchmark_render_context


@pytest.fixture
def presentation(tmp_path):
    baseline = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "tests"
            / "fixtures"
            / "figures"
            / "FIGURE_BASELINE.json"
        ).read_text()
    )
    ctx = benchmark_render_context(
        tmp_path, method_keys=baseline["method_order"], datasets=("nCoV",)
    )
    return legends.LegendPresentation(
        ctx.methods, ctx.family_order, ctx.legend.integration_tool_order
    )


def test_integration_entries_are_paired_and_dimension_suffix_is_shared(presentation):
    keys = [key for key, value in presentation.methods.items() if value.family == "integration"]
    handles, labels = legends._integration_entries(keys, presentation=presentation)
    assert all(isinstance(handle, tuple) and len(handle) == 2 for handle in handles)
    assert not any("50D" in label or "2D" in label for label in labels)
    assert len(handles) == 6


def test_an_unpaired_integration_entry_is_refused(presentation):
    with pytest.raises(ValueError, match="folded 50D/2D pair"):
        legends._integration_entries(["seurat-50"], presentation=presentation)


def test_dimension_key_glyphs_are_adjacent_to_their_own_labels():
    handles, labels = legends._dimension_key_entries()
    assert len(handles) == len(labels) == 2
    assert [h.get_marker() for h in handles] == [entry[0] for entry in legends.DIMENSION_KEY]
    assert labels == [entry[1] for entry in legends.DIMENSION_KEY]
    assert "$" in labels[0] and "\\rightarrow" in labels[0]
    assert all("→" not in label for label in labels)


def test_legend_reservation_adds_exactly_the_requested_rows(presentation):
    ordered = list(presentation.methods)
    base = legends.legend_rows(ordered, presentation=presentation)
    assert legends.legend_rows(ordered, presentation=presentation, extra_rows=1) == base + 1


def test_error_bars_and_metric_span_never_leave_the_metric_range():
    np.testing.assert_allclose(metrics.clipped_error(0.9, 0.3), (0.3, 0.1))
    frame = pd.DataFrame({"x_mean": [0.1, 0.9], "x_sd": [0.3, 0.3]})
    assert metrics.metric_span(frame, "x").between(0, 1).all()
