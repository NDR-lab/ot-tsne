"""Metric drawing primitives: bounded error bars and round ticks."""

from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ne_aligned_utils.paper.rendering import metrics


@pytest.mark.parametrize("sd", [None, np.nan, np.inf, 0.0, 0.4])
def test_clipped_error_matches_bounded_contract(sd):
    spread = 0 if sd is None or not np.isfinite(sd) else sd
    expected = (0.1 - max(0, 0.1 - spread), min(1, 0.1 + spread) - 0.1)
    assert metrics.clipped_error(0.1, sd) == expected


def test_point_preserves_artist_arguments_and_rank_has_no_bars():
    ax = Mock()
    row = pd.Series(dict(x_mean=0.1, y_mean=0.9, x_sd=0.4, y_sd=0.3))
    style = dict(c="red", marker="x", alpha=0.85, zorder=4)
    metrics.draw_metric_point(ax, row, x="x_mean", y="y_mean", scatter=style, error_color="red")
    kwargs = ax.errorbar.call_args.kwargs
    np.testing.assert_allclose(kwargs.pop("xerr"), [[0.1], [0.4]])
    np.testing.assert_allclose(kwargs.pop("yerr"), [[0.3], [0.1]])
    assert kwargs == dict(
        fmt="none", ecolor="red", elinewidth=0.55, alpha=0.55, zorder=-1, clip_on=False
    )
    ax.scatter.assert_called_once_with(0.1, 0.9, clip_on=False, **style)
    ax.reset_mock()
    metrics.draw_metric_point(
        ax, row, x="x_mean", y="y_mean", scatter=style, error_color="red", errorbars=False
    )
    ax.errorbar.assert_not_called()


def test_span_preserves_missing_spread_and_indices():
    frame = pd.DataFrame(dict(x_mean=[0.1, 0.9], x_sd=[np.nan, 0.3]), index=[5, 7])
    np.testing.assert_allclose(metrics.metric_span(frame, "x"), [0.1, 0.9, 0.1, 0.6, 0.1, 1.0])
    assert list(metrics.metric_span(frame.drop(columns="x_sd"), "x")) == [0.1, 0.9] * 3


def test_ticks_preserve_rounding_padding_and_label_precision():
    fig, ax = plt.subplots()
    try:
        metrics.round_three_ticks(ax, [np.nan, 0.12, 0.88], [0.5], step_x=0.1)
        np.testing.assert_allclose(ax.get_xticks(), [0.1, 0.5, 0.9])
        np.testing.assert_allclose(ax.get_xlim(), [0.06, 0.94])
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0.1", "0.5", "0.9"]
        np.testing.assert_allclose(ax.get_ylim(), [0.4975, 0.5525])
    finally:
        plt.close(fig)
