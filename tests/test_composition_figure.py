"""Cell-type composition blocks from aggregate caches, and their figure."""

import matplotlib.pyplot as plt
import numpy as np

from ne_aligned_utils.paper import layout
from ne_aligned_utils.paper.figures import composition


def _cache(labels0, labels1, names):
    return dict(
        label_names=np.array(names), labels0=np.array([labels0]), labels1=np.array([labels1])
    )


def test_block_puts_shared_types_first_then_largest_and_drops_empty_types():
    cache = _cache([0, 0, 1, 2, 2, 2], [0, 2, 2], ["a", "b", "c", "unused"])
    block = composition.composition_block("d", "D", cache)
    assert block.names.tolist() == ["c", "a", "b"]
    assert block.counts.tolist() == [[3, 2, 1], [2, 1, 0]]
    assert block.shared.tolist() == [True, True, False]
    assert block.total == 9


def test_blocks_split_smallest_first_into_figures_of_four():
    blocks = [
        composition.CompositionBlock(str(n), str(n), np.array(["a"]), np.array([[n], [n]]))
        for n in (5, 1, 7, 3, 9, 2, 4)
    ]
    parts = composition.split_by_size(blocks)
    assert [[b.total for b in part] for part in parts] == [[2, 4, 6, 8], [10, 14, 18]]


def test_axes_end_on_a_round_tick_enclosing_the_data():
    limit, ticks = composition._count_axis(1437)
    assert limit >= 1437 and ticks[-1] == limit and 4 <= len(ticks) <= 6
    limit, ticks = composition._fraction_axis(0.21, "nCoV")
    assert limit == 0.30


def test_figure_draws_two_panels_per_dataset():
    blocks = [
        composition.composition_block(key, key, _cache([0, 1, 1, 2], [0, 0, 1], ["x", "y", "z"]))
        for key in ("one", "two")
    ]
    try:
        with layout.paper_style():
            fig = composition.batch_composition(
                blocks, name_column_in=composition.name_column_width(blocks)
            )
        assert len(fig.axes) == 4
        assert any("one" in t.get_text() for t in fig.texts)
    finally:
        plt.close("all")
