"""Characterize seed summaries independently of registry, selection and plotting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ne_aligned_utils import config
from ne_aligned_utils.paper.selection.benchmark import reported_metric_rows
from ne_aligned_utils.paper.summaries.metrics import summarise_seeds

ROOT = Path(__file__).resolve().parents[1]


def _original_aggregation(frame, metrics):
    """Frozen pre-extraction body, with registry input supplied explicitly."""
    keys = ["dataset", "method_key", "method_label", "family"]
    metrics = list(metrics)
    mean = frame.groupby(keys, dropna=False)[metrics].mean().add_suffix("_mean")
    std = frame.groupby(keys, dropna=False)[metrics].std(ddof=1).add_suffix("_sd")
    count = frame.groupby(keys, dropna=False)["seed"].nunique().rename("n")
    return mean.join(std).join(count).reset_index()


def _frame():
    return pd.DataFrame(
        {
            "dataset": ["z", "a", "a", "a", "a"],
            "method_key": ["m"] * 5,
            "method_label": ["Method"] * 5,
            "family": ["family"] * 5,
            "seed": [0, 0, 0, 1, np.nan],
            "x": [9.0, 1.0, 3.0, 5.0, np.nan],
            "y": [2.0, 4.0, np.nan, 8.0, 12.0],
        }
    )


def test_mean_sample_sd_unique_seed_count_and_explicit_column_order():
    result = summarise_seeds(_frame(), metrics=("y", "x"))
    expected = pd.DataFrame(
        {
            "dataset": ["a", "z"],
            "method_key": ["m", "m"],
            "method_label": ["Method", "Method"],
            "family": ["family", "family"],
            "y_mean": [8.0, 2.0],
            "x_mean": [3.0, 9.0],
            "y_sd": [4.0, np.nan],
            "x_sd": [2.0, np.nan],
            "n": [2, 1],
        }
    )
    assert_frame_equal(result, expected, check_exact=True)


@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "empty",
        "singleton",
        "all-missing-metric",
        "missing-group",
        "float32",
        "nullable",
        "categorical",
        "shuffled-index",
        "no-metrics",
    ],
)
def test_exact_original_behavior_dtypes_order_and_input_preservation(case):
    frame = _frame()
    metrics = ["x", "y"]
    if case == "empty":
        frame = frame.iloc[:0]
    elif case == "singleton":
        frame = frame.iloc[:1]
    elif case == "all-missing-metric":
        frame["x"] = np.nan
    elif case == "missing-group":
        frame.loc[1:2, ["dataset", "method_label", "family"]] = None
    elif case == "float32":
        frame = frame.astype({"x": "float32", "y": "float32"})
    elif case == "nullable":
        frame = frame.astype({"x": "Float64", "y": "Float64", "seed": "Int64"})
    elif case == "categorical":
        frame["family"] = pd.Categorical(frame.family)
    elif case == "shuffled-index":
        frame = frame.iloc[::-1].set_axis([9, 2, 2, 0, 4])
    elif case == "no-metrics":
        metrics = []
    before = frame.copy(deep=True)
    # The original categorical-groupby default warns; this extraction retains it.
    with pd.option_context("mode.chained_assignment", "raise"):
        if case == "categorical":
            with pytest.warns(FutureWarning):
                expected = _original_aggregation(frame, metrics)
            with pytest.warns(FutureWarning):
                actual = summarise_seeds(frame, metrics=metrics)
        else:
            expected = _original_aggregation(frame, metrics)
            actual = summarise_seeds(frame, metrics=metrics)
    assert_frame_equal(actual, expected, check_exact=True)
    assert_frame_equal(frame, before, check_exact=True)


@pytest.mark.parametrize("group", ["dataset", "method_key", "method_label", "family"])
def test_each_identity_grouping_key_is_preserved(group):
    frame = _frame().iloc[1:3].copy()
    frame.loc[frame.index[-1], group] = "different"
    assert len(summarise_seeds(frame, metrics=["x"])) == 2


def test_missing_columns_still_raise_without_new_validation():
    with pytest.raises(KeyError):
        summarise_seeds(_frame(), metrics=["absent"])
    with pytest.raises(KeyError):
        summarise_seeds(_frame().drop(columns="seed"), metrics=["x"])


@pytest.mark.parametrize("backend,rel_reg", [("tsne", 1.0), ("umap", 1.0), ("tsne", 0.1)])
def test_explicit_selection_then_summary_preserves_original_calculation(backend, rel_reg):
    frame = _frame().assign(
        backend=["tsne", "umap", "na", "tsne", "tsne"],
        rel_reg=[1.0, 1.0, np.nan, 0.1, 1.0],
        effective_k=[15, 15, np.nan, 15, 30],
    )
    selected = reported_metric_rows(
        frame,
        backend=backend,
        no_backend=config.NO_BACKEND,
        rel_reg=rel_reg,
        effective_k=15,
    )
    expected = _original_aggregation(selected, ["y", "x"])
    assert_frame_equal(summarise_seeds(selected, metrics=["y", "x"]), expected, check_exact=True)
    # The pure layer does no hidden configuration selection of its own.
    assert_frame_equal(
        summarise_seeds(frame, metrics=["y", "x"]),
        _original_aggregation(frame, ["y", "x"]),
        check_exact=True,
    )
