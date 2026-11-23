"""The single paper registry: names, colours, markers and figure order."""

from __future__ import annotations

import re

import pytest

from ne_aligned_utils import config
from ne_aligned_utils.paper import layout, registry as reg
from ne_aligned_utils.paper.rendering import presentation

REPORTED = (
    "ot", "ot-selfotmedian", "ot-zerodiag", "ot-gramdiag", "ot-knndiag",
    "multimap", "bc-tsne", "ref", "rc-tsne",
    "seurat-50", "seurat-2d", "liger-50", "liger-2d", "scanorama-50", "scanorama-2d",
    "harmony-50", "harmony-2d", "scvi-50", "scvi-2d", "eot-50", "eot-t0",
    "joint-pca", "joint",
)  # fmt: skip


def test_reported_methods_are_registered_in_figure_order():
    assert reg.method_order(REPORTED) == list(REPORTED)


def test_a_key_is_the_benchmark_method_name_unless_it_is_a_special_configuration():
    for key, entry in reg.METHODS.items():
        if key == "multimap-k15":
            assert entry.method == "multimap"
        else:
            assert entry.method == key
        assert config.method_spec(entry.method)


def test_no_retired_or_suffixed_default_keys_remain():
    retired = {"ref-our", "seurat3-2d", "seurat3-50", "multimap@nn15", "multimap@nn30"}
    assert not retired & set(reg.METHODS)
    assert not any("@" in key for key in reg.METHODS)


def test_unknown_method_raises_rather_than_falling_back():
    with pytest.raises(KeyError, match="not in the paper registry"):
        reg.resolve("no-such-method")
    with pytest.raises(KeyError):
        reg.method_order(["ot", "no-such-method"])


def test_colours_are_valid_hex_and_one_tool_means_one_colour():
    by_tool: dict[str, set[str]] = {}
    for entry in reg.METHODS.values():
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", entry.color), entry.key
        if entry.family == "integration":
            by_tool.setdefault(entry.tool, set()).add(entry.color)
    assert all(len(colours) == 1 for colours in by_tool.values())


def test_integration_forms_differ_by_marker_only():
    for tool in {e.tool for e in reg.METHODS.values() if e.family == "integration"}:
        forms = [e for e in reg.METHODS.values() if e.tool == tool]
        assert sorted(e.marker for e in forms) == ["o", "s"]
        circle = next(e for e in forms if e.marker == "o")
        assert config.has_backend_axis(circle.method)


def test_the_contribution_is_identifiable_by_shape_alone():
    diamonds = {e.family for e in reg.METHODS.values() if e.marker == "D"}
    assert diamonds == {"ot"}


def test_ot_variants_run_dark_to_light_in_figure_order():
    ot = [key for key in reg.method_order() if reg.METHODS[key].family == "ot"]
    lightness = [sum(int(reg.METHODS[k].color[i : i + 2], 16) for i in (1, 3, 5)) for k in ot]
    assert lightness[:5] == sorted(lightness[:5])


def test_unfilled_markers_carry_colour_on_the_stroke():
    for entry in reg.METHODS.values():
        kwargs = entry.scatter_kwargs()
        if entry.unfilled:
            assert "edgecolors" not in kwargs and kwargs["linewidths"] == 0.9
        else:
            assert kwargs["edgecolors"] == entry.edge_color


def test_marker_areas_respect_the_print_floor():
    for entry in reg.METHODS.values():
        assert entry.size**0.5 >= layout.MIN_MARKER_PT, entry.key


def test_ot_variants_have_full_short_and_legend_names():
    entry = reg.METHODS["ot-gramdiag"]
    assert (entry.label, entry.short_label, entry.legend_label) == (
        "OT-tSNE (Gram diagonal)",
        "OT-tSNE (Gram)",
        "Gram diagonal",
    )
    plain = reg.METHODS["seurat-50"]
    assert plain.label == plain.short_label == plain.legend_label == "Seurat 50D"


def test_multimap_default_and_matched_k_arms_are_distinct():
    default, matched = reg.METHODS["multimap"], reg.METHODS["multimap-k15"]
    assert default.label == "MultiMAP" and matched.label == "MultiMAP k=15"
    assert default.color != matched.color


def test_integration_legend_order_names_every_integration_tool():
    tools = {e.tool for e in reg.METHODS.values() if e.family == "integration"}
    assert tools == set(reg.INTEGRATION_LEGEND_ORDER)


def test_metrics_split_into_bio_and_mixing_halves():
    assert set(reg.BIO_METRICS) | set(reg.MIXING_METRICS) == set(reg.METRICS)
    assert set(reg.METRIC_SOURCE_COLUMNS.values()) == set(reg.METRICS)


def test_presentation_follows_the_registry(tmp_path):
    ctx = presentation.benchmark_render_context(tmp_path, method_keys=REPORTED, datasets=["nCoV"])
    assert list(ctx.method_order) == list(REPORTED)
    for key in REPORTED:
        entry, shown = reg.METHODS[key], ctx.methods[key]
        assert shown.display_label == entry.legend_label
        assert shown.scatter["c"] == entry.color and shown.scatter["marker"] == entry.marker
    with pytest.raises(TypeError):
        ctx.methods["ot"] = None
    labels = presentation.rank_composer().method_labels
    assert labels["ot-knndiag"] == "OT-tSNE (kNN)"
