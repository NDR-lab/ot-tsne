"""Family headings and the default figure root."""

from ne_aligned_utils import config
from ne_aligned_utils.paper import layout, registry as reg


def test_the_family_titles_are_the_papers_wording():
    """Every heading takes two lines, so the four share one title band; the words are pinned."""
    assert set(reg.FAMILY_LABELS) == set(reg.FAMILY_ORDER)
    assert all(title.count("\n") == 1 for title in reg.FAMILY_LABELS.values())
    words = {k: " ".join(v.split()) for k, v in reg.FAMILY_LABELS.items()}
    assert words["ot"] == "OT-tSNE (ours)"
    assert words["dedicated"] == "Integrated visualization"
    assert words["integration"] == "Integrated in 2D or 50D $\\rightarrow$ t-SNE"
    assert words["reference"] == "No integration"


def test_save_figure_still_defaults_to_the_working_figures_root(tmp_path, monkeypatch):
    """The `root` parameter must not have moved anything that does not pass it."""
    import matplotlib.pyplot as plt

    monkeypatch.setattr(config, "PAPER_FIGURES_DIR", tmp_path)
    with layout.paper_style():
        fig = plt.figure(figsize=layout.full_width(1.5))
        fig.add_subplot(111).plot([0, 1], [0, 1])
        path = layout.save_figure(fig, "unrooted", preview=False)
    assert path == tmp_path / "unrooted.png"
