"""The main synthetic graph panel averages node purity from both batches."""
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ne_aligned_utils.experiments.synthetic import SyntheticExperiment
from ne_aligned_utils.paper.figures import synthetic as drawing


def test_two_batch_purity_is_invariant_to_swapping_batches():
    experiment = object.__new__(SyntheticExperiment)
    support = np.array([[1, 1, 1], [1, 1, 0]], dtype=bool)
    labels0, labels1 = np.array([0, 1]), np.array([0, 1, 1])
    q01 = experiment.support_stats(support, labels0, labels1)[0]
    q10 = experiment.support_stats(support.T, labels1, labels0)[0]
    assert q01 == pytest.approx(5 / 12)
    assert q10 == pytest.approx(1 / 3)
    assert q01 != q10
    assert .5 * (q01 + q10) == pytest.approx(3 / 8)
    swapped = .5 * (experiment.support_stats(support.T, labels1, labels0)[0]
                    + experiment.support_stats(support, labels0, labels1)[0])
    assert swapped == pytest.approx(3 / 8)


def test_main_graph_curves_use_symmetric_fields():
    base = dict(mode=['original'] * 2, magnitude=[1., 10.], corruption_seed=[0, 0])
    affinity = pd.DataFrame(dict(base,
        eot_shared_or_digit_purity=[.1, .1],
        eot_shared_or_symmetric_digit_purity=[.45, .46]))
    knn = pd.DataFrame(dict(base,
        shared_correct_class_neighbor_fraction=[.2, .2],
        mutual_correct_class_neighbor_fraction=[.3, .3],
        shared_symmetric_correct_class_neighbor_fraction=[.65, .66],
        mutual_symmetric_correct_class_neighbor_fraction=[.85, .86]))
    runs = SimpleNamespace(configs={'original': {'magnitudes': [1., 10.]}}, knn=knn)
    fig = plt.figure()
    try:
        drawing.mechanism_panels(fig, (fig.add_gridspec(1, 1)[0],),
            'two_sided_independent_scales', runs, affinity, directed_panel=False)
        curves = [line for line in fig.axes[0].lines if len(line.get_xdata()) == 2]
        # Error-bar cap lines may also have length two; identify the method markers.
        curves = [line for line in curves if line.get_linestyle() not in ('None', 'none', '')]
        assert len(curves) == 3
        for line, expected in zip(curves, ([.45, .46], [.65, .66], [.85, .86])):
            np.testing.assert_allclose(np.asarray(line.get_ydata(), dtype=float), expected)
    finally:
        plt.close(fig)
