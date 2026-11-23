"""Runtime manual geometry must not inherit the application's layout engine."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ne_aligned_utils.paper import layout
from ne_aligned_utils.paper.figures import runtime
from ne_aligned_utils.paper.rendering import runtime_drawing


@pytest.mark.parametrize("rows", (1, 2, 7))
def test_breakdown_title_legend_and_labels_fit_without_layout_warnings(rows):
    labels = (
        "nCoV",
        "Hematopoietic",
        "Mouse atlas",
        "Human brain",
        "Lupus",
        "PBMC",
        "Mouse cortex",
    )
    totals = np.array([8, 11, 27, 41, 137, 197, 1393])[:rows]
    table = pd.DataFrame(
        {
            "dataset_label": labels[:rows],
            "n_obs": [4117, 4649, 6954, 9215, 13999, 15476, 26588][:rows],
            "time_total": totals,
            "time_pre_embedding": totals * 0.6,
            "time_embedding": totals * 0.4,
        }
    )
    with layout.paper_style(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig = runtime.ot_breakdown(
            table,
            method_label="OT-tSNE",
            segments=(
                ("time_pre_embedding", "Optimal transport", "#602003"),
                ("time_embedding", "t-SNE", "#c8bba9"),
            ),
            figsize=(5.5, 2.95),
        )
        try:
            FigureCanvasAgg(fig)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            title = fig._suptitle.get_window_extent(renderer)
            legend = fig.axes[0].get_legend().get_window_extent(renderer)
            assert not title.overlaps(legend)
            artists = [fig._suptitle, *fig.axes[0].get_yticklabels()]
            for ax in fig.axes:
                artists.extend([ax.xaxis.label, *ax.get_xticklabels(), *ax.texts])
            for artist in artists:
                box = artist.get_window_extent(renderer)
                assert box.x0 >= 0 and box.y0 >= 0, artist.get_text()
                assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, artist.get_text()
        finally:
            plt.close(fig)
    assert not [w for w in caught if "layout" in str(w.message).lower()]


def test_manual_runtime_axes_do_not_create_a_constrained_layout_engine():
    with layout.paper_style(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig, axes = runtime_drawing.family_axes([2, 3], 0.1, (5.5, 8.6))
        try:
            FigureCanvasAgg(fig)
            fig.canvas.draw()
            assert fig.get_layout_engine() is None
            assert len(axes) == 2
        finally:
            plt.close(fig)
    assert not [w for w in caught if "layout" in str(w.message).lower()]
