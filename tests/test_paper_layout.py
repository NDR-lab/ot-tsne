"""Check physical figure geometry, export settings, and print legibility from rendered files."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ne_aligned_utils.paper import layout as L  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_rcparams():
    """``plt.style.use`` is global and does not restore itself.

    Without this, activating the paper style here changes the font sizes and tick geometry for
    every test that runs later in the same process -- which is exactly what happened the first
    time these tests ran, taking twelve unrelated ``test_rank_plots`` assertions with them.
    """
    saved = matplotlib.rcParams.copy()
    yield
    matplotlib.rcParams.update(saved)


# The text block


def test_widths_match_the_manuscript_text_block():
    assert L.FULL_WIDTH == 5.500


def test_width_helpers_return_exact_inches():
    assert L.full_width(2.0) == (5.500, 2.0)


# square_grid


@pytest.mark.parametrize(
    "ncols,expected",
    [(2, 2.490), (3, 1.633), (4, 1.205), (5, 0.948)],
)
def test_panel_sides_match_the_documented_table(ncols, expected):
    """docs/figure_plan.md section 3.4 quotes these; captions quote them from the same source."""
    assert L.square_grid(ncols, 1).panel == pytest.approx(expected, abs=0.001)


def test_gallery_margins_give_a_wider_panel():
    """No ticks and no labels buy back most of the margin."""
    geom = L.square_grid(6, 1, **L.GALLERY_MARGINS)
    assert geom.panel == pytest.approx(0.872, abs=0.001)


def test_height_makes_panels_square():
    geom = L.square_grid(4, 3)
    expected = L.MARGIN_TOP + L.MARGIN_BOTTOM + 3 * geom.panel + 2 * L.GAP
    assert geom.figsize[1] == pytest.approx(expected, abs=1e-9)
    assert geom.figsize[0] == L.FULL_WIDTH


def test_legend_reservation_adds_to_height():
    without = L.square_grid(4, 1).figsize[1]
    with_legend = L.square_grid(4, 1, legend=0.6).figsize[1]
    assert with_legend - without == pytest.approx(0.6)


def test_too_many_columns_raises_rather_than_returning_a_negative_panel():
    with pytest.raises(ValueError, match="do not fit"):
        L.square_grid(200, 1)


def test_zero_rows_raises():
    with pytest.raises(ValueError):
        L.square_grid(4, 0)


def test_geometry_describes_itself():
    text = L.square_grid(4, 2, legend=0.6).describe()
    assert "1.205" in text and "4x2" in text


# Style


def test_style_files_exist():
    assert L.BASE_STYLE.exists()
    assert L.APPENDIX_STYLE.exists()


def test_base_style_sets_the_manuscript_type_ladder():
    L.use_paper_style()
    assert plt.rcParams["font.size"] == 9
    assert plt.rcParams["axes.labelsize"] == 9
    assert plt.rcParams["xtick.labelsize"] == 8
    assert plt.rcParams["legend.fontsize"] == 8
    assert plt.rcParams["figure.titlesize"] == 10


def test_base_style_sets_the_export_settings_that_matter():
    L.use_paper_style()
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["ps.fonttype"] == 42
    # `tight` would re-crop on save and break the author-at-final-size rule.
    assert plt.rcParams["savefig.bbox"] != "tight"
    assert plt.rcParams["font.family"] == ["sans-serif"]
    assert plt.rcParams["font.sans-serif"][0] == "Arial"
    # Math in the same face as the words beside it, not a second sans.
    assert plt.rcParams["mathtext.fontset"] == "custom"
    assert plt.rcParams["mathtext.rm"] == "Arial"


def test_style_sets_no_colour_cycle_beyond_the_default():
    """Colour comes from the registry.

    A default cycle silently gives two unrelated methods the same color when a panel's method
    set changes.
    """
    L.use_paper_style()
    text = L.BASE_STYLE.read_text()
    assert "axes.prop_cycle" not in text


def test_appendix_style_only_overrides():
    """It inherits by list order; anything it restates unchanged is dead weight."""
    L.use_paper_style(appendix=True)
    assert plt.rcParams["font.size"] == 10
    assert plt.rcParams["xtick.labelsize"] == 9
    # inherited from the base, not restated
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["savefig.bbox"] != "tight"


def test_paper_style_context_manager_restores_rcparams():
    """The non-leaking form, for anything sharing a process with other matplotlib code."""
    before = plt.rcParams["font.size"]
    with L.paper_style():
        assert plt.rcParams["font.size"] == 9
    assert plt.rcParams["font.size"] == before


# save_figure and the legibility gate


def test_save_figure_takes_no_bbox_argument():
    """Not an oversight: exposing it would let a caller break the sizing rule."""
    import inspect

    assert "bbox_inches" not in inspect.signature(L.save_figure).parameters


def _figure(panel_points: float = 30.0, font: float = 9.0):
    L.use_paper_style()
    geom = L.square_grid(4, 1)
    fig, axes = plt.subplots(1, 4, figsize=geom.figsize)
    for ax in axes:
        ax.scatter([0.2, 0.8], [0.3, 0.9], s=panel_points, zorder=-1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("kBET", fontsize=font)
    return fig, geom


def test_preview_measures_the_declared_size(tmp_path, monkeypatch):
    from ne_aligned_utils import config

    monkeypatch.setattr(config, "PAPER_FIGURES_DIR", tmp_path)
    fig, geom = _figure()
    report = L.print_preview(fig, tmp_path / "probe.pdf")
    assert report.measured[0] == pytest.approx(geom.figsize[0], abs=0.01)
    assert report.measured[1] == pytest.approx(geom.figsize[1], abs=0.01)
    assert not report.problems
    plt.close(fig)


def test_preview_rejects_text_below_the_floor(tmp_path):
    fig, _ = _figure(font=5.0)
    report = L.print_preview(fig, tmp_path / "small_text.pdf")
    assert any("below the 7 pt floor" in p for p in report.problems)
    with pytest.raises(L.LegibilityError):
        report.raise_for_problems()
    plt.close(fig)


def test_preview_rejects_markers_that_stop_reading_as_shapes(tmp_path):
    fig, _ = _figure(panel_points=4.0)  # 2 pt across
    report = L.print_preview(fig, tmp_path / "small_marker.pdf")
    assert any("smallest marker" in p for p in report.problems)
    plt.close(fig)


def test_density_kind_exempts_the_gallery(tmp_path):
    """F6 draws thousands of points at s=0.8, where a point is a pixel, not a shape."""
    fig, _ = _figure(panel_points=0.8)
    report = L.print_preview(fig, tmp_path / "gallery.pdf", kind="density")
    assert not any("smallest marker" in p for p in report.problems)
    plt.close(fig)


def test_preview_rejects_an_unknown_kind(tmp_path):
    fig, _ = _figure()
    with pytest.raises(ValueError, match="glyph"):
        L.print_preview(fig, tmp_path / "x.pdf", kind="raster")
    plt.close(fig)


def test_save_figure_writes_a_png_at_the_declared_size(tmp_path, monkeypatch):
    from ne_aligned_utils import config

    monkeypatch.setattr(config, "PAPER_FIGURES_DIR", tmp_path)
    fig, geom = _figure()
    path = L.save_figure(fig, "f1c_probe")
    assert path == tmp_path / "f1c_probe.png"
    assert path.exists()


def test_appendix_flag_does_not_rewrite_the_filename(tmp_path, monkeypatch):
    """Ordering is carried by the name, not by a prefix the helper adds.

    ``save_figure`` used to prepend "a" for appendix figures. That did not sort: "a3" comes before
    "f1", so the appendix listed ahead of the main text, and "a10" sorted between "a1" and "a3".
    Callers now pass a zero-padded sequence prefix and get exactly the filename they asked for;
    ``appendix`` is semantic only.
    """
    from ne_aligned_utils import config

    monkeypatch.setattr(config, "PAPER_FIGURES_DIR", tmp_path)
    fig, _ = _figure()
    path = L.save_figure(fig, "06_a1_full_benchmark", appendix=True)
    assert path.name == "06_a1_full_benchmark.png"


def test_sequence_prefixes_sort_into_reading_order():
    """Main text before appendix, and stable past nine figures."""
    names = [
        "01_f1a_benchmark_direct",
        "05_f4_gallery_nCoV_lupus",
        "06_a1_full_benchmark",
        "09_a4_delta_ot_tsne_eps10",
        "10_a5_delta_ot_tsne_eps1_with_old_panel",
        "19_a11_rank_aggregate",
    ]
    assert sorted(names) == names


def test_save_figure_leaves_rasterisation_alone(tmp_path, monkeypatch):
    """The rasterisation-zorder pass went out with the PDF.

    It existed so a 14,000-point scatter stayed raster inside a vector PDF while axes and text
    stayed vector. The output is a PNG now, raster all the way through, so setting it would
    change nothing and the figure the caller handed in comes back untouched.
    """
    from ne_aligned_utils import config

    monkeypatch.setattr(config, "PAPER_FIGURES_DIR", tmp_path)
    fig, _ = _figure()
    axes = fig.get_axes()
    before = [ax.get_rasterization_zorder() for ax in axes]
    path = L.save_figure(fig, "raster_probe", close=False)
    assert path == tmp_path / "raster_probe.png"
    assert [ax.get_rasterization_zorder() for ax in axes] == before
    plt.close(fig)


# Axis helpers


def test_style_axes_offsets_the_spines():
    fig, ax = plt.subplots()
    L.style_axes(ax)
    assert ax.spines["left"].get_position() == ("outward", 2.0)
    assert not ax.spines["top"].get_visible()
    plt.close(fig)


def test_legend_columns_shrink_as_labels_grow():
    L.use_paper_style()
    short = ["EOT t=0", "BC-tSNE", "Joint PCA", "Ref tSNE"]
    long = ["MultiMAP (K=15, matched)", "MultiMAP (K=30, controlled)", "OT-tSNE", "Joint PCA"]
    assert L.legend_columns(short) > L.legend_columns(long)


def test_legend_columns_never_overflow_the_page():
    """Overflow is invisible: constrained_layout resolves it by shrinking the axes instead."""
    L.use_paper_style()
    labels = ["MultiMAP (K=30, controlled)"] * 8
    ncol = L.legend_columns(labels)
    fontsize = plt.rcParams["legend.fontsize"]
    entry_in = (len(labels[0]) * L._EM_PER_CHAR + L._LEGEND_FURNITURE_EM) * fontsize / 72
    assert ncol * entry_in <= L.FULL_WIDTH


def test_grid_with_legend_does_not_eat_panel_width():
    """The legend takes a gridspec row, so panel width is unchanged by adding one."""
    L.use_paper_style()
    labels = ["MultiMAP (K=15, matched)"] * 8
    fig, axes, legend_ax, geom = L.grid_with_legend(4, 1, labels)
    assert geom.panel == pytest.approx(L.square_grid(4, 1).panel, abs=1e-9)
    assert len(axes) == 4
    assert not legend_ax.axison
    plt.close(fig)


def test_panels_stay_square_once_rendered():
    """The direct regression: the first F1 rendered four panels at near-zero width.

    Anything that participates in constrained_layout and is wider than its cell -- an oversized
    figure legend, an over-wide annotation left in the layout -- is resolved by shrinking the
    axes, silently. Measuring the rendered axes is the only check that catches it.
    """
    L.use_paper_style()
    labels = ["MultiMAP (K=15, matched)"] * 8
    fig, axes, legend_ax, geom = L.grid_with_legend(4, 1, labels)
    for ax in axes:
        text = ax.annotate("a very wide label indeed", (0.5, 0.5), fontsize=7)
        text.set_in_layout(False)
    handles = [Line2D([], [], linestyle="none", marker="o") for _ in labels]
    legend_ax.legend(handles=handles, labels=labels, loc="center", ncol=L.legend_columns(labels))
    fig.canvas.draw()
    widths = []
    for ax in axes:
        box = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        assert box.width > 0.6 * geom.panel, f"panel collapsed to {box.width:.3f} in"
        widths.append(box.width)
    # and all four stay equal, so no single panel absorbs the squeeze
    assert max(widths) - min(widths) < 0.01
    plt.close(fig)
