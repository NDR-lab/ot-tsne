"""Reported-embedding selection and gallery composition from preloaded coordinates."""

from dataclasses import FrozenInstanceError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ne_aligned_utils.analysis import cell_type_palette
from ne_aligned_utils.paper import layout
from ne_aligned_utils.paper.figures.qualitative import GalleryBlock
from ne_aligned_utils.paper.inputs import embeddings
from ne_aligned_utils.paper.rendering import styles
from ne_aligned_utils.paper.rendering.presentation import qualitative_composer


def _cached(columns, n=40):
    rng = np.random.default_rng(42)
    return {
        key: (
            rng.normal(size=(n, 2)),
            dict(
                batch_ids=np.repeat([0, 1], n // 2),
                labels0=np.array([np.arange(n // 2) % 3]),
                labels1=np.array([np.arange(n // 2) % 3]),
                label_names=np.array(["a", "b", "c"]),
            ),
        )
        for key, _heading in columns
    }


COLUMNS = (("joint", "Joint t-SNE"), ("ot", "OT-tSNE"), ("seurat-50", "Seurat"))


@pytest.mark.parametrize("grouped", (False, True))
def test_gallery_draws_two_rows_per_method_without_touching_its_inputs(monkeypatch, grouped):
    groups = (("Baseline", 1), ("Integrated", 2)) if grouped else None
    cached = _cached(COLUMNS)
    before = {
        key: (embedding.copy(), {k: v.copy() for k, v in cache.items()})
        for key, (embedding, cache) in cached.items()
    }
    monkeypatch.setattr(layout, "save_figure", lambda *a, **k: pytest.fail("composer saved"))
    try:
        with layout.paper_style():
            fig = qualitative_composer().embedding_panels(
                cached, cell_type_palette(("a", "b", "c")), columns=COLUMNS, groups=groups
            )
        panels = [ax for ax in fig.axes if ax.collections]
        assert len(panels) == 2 * len(COLUMNS)
        assert [ax.get_title() for ax in panels[: len(COLUMNS)]] == [h for _k, h in COLUMNS]
        for key, (embedding, cache) in cached.items():
            assert np.array_equal(embedding, before[key][0])
            for name, value in cache.items():
                assert np.array_equal(value, before[key][1][name])
    finally:
        plt.close("all")


@pytest.mark.parametrize("braces", ("once", "each"))
def test_stacked_gallery_titles_once_and_letters_every_block(braces):
    blocks = []
    for name in ("First", "Second"):
        cached = _cached(COLUMNS)
        blocks.append(
            GalleryBlock(
                name=name,
                embeddings={k: v[0] for k, v in cached.items()},
                caches={k: v[1] for k, v in cached.items()},
                palette=cell_type_palette(("a", "b", "c")),
                labels=np.concatenate([np.arange(20) % 3, np.arange(20) % 3]),
            )
        )
    groups = (("Baseline", 1), ("Integrated", 2))
    try:
        with layout.paper_style():
            fig = qualitative_composer().stacked_panels(
                blocks, columns=COLUMNS, groups=groups, braces=braces
            )
        panels = [ax for ax in fig.axes if ax.collections]
        assert len(panels) == 2 * 2 * len(COLUMNS)
        assert sum(bool(ax.get_title()) for ax in panels) == len(COLUMNS)
        texts = [t.get_text() for t in fig.texts]
        assert sum(t.startswith("First") or t.startswith("Second") for t in texts) == 2
        assert texts.count("Integrated") == (2 if braces == "each" else 1)
        with pytest.raises(ValueError, match="braces"):
            qualitative_composer().stacked_panels(
                blocks, columns=COLUMNS, groups=groups, braces="x"
            )
    finally:
        plt.close("all")


def test_gallery_style_is_detached(monkeypatch):
    composer = qualitative_composer()
    margins = dict(composer.style.qualitative_margins)
    with pytest.raises(TypeError):
        composer.style.qualitative_margins["left"] = 99
    with pytest.raises(FrozenInstanceError):
        composer.style.seed = 99
    monkeypatch.setitem(styles.QUALITATIVE_MARGINS, "left", 99)
    assert dict(composer.style.qualitative_margins) == margins


@pytest.mark.parametrize(("key", "has_axis"), (("rc-tsne", False), ("ot", True)))
def test_embedding_selection_preserves_backend_seed_init_and_regularization(
    monkeypatch, key, has_axis
):
    frame = pd.DataFrame(
        [
            dict(
                dataset="nCoV",
                method_key=key,
                backend=backend,
                seed=seed,
                init=init,
                rel_reg=reg,
                source_cache=f"{backend}-{seed}-{init}-{reg}.npz",
            )
            for backend in ("tsne", "umap")
            for seed in (1, 0)
            for init in ("spectral", "random")
            for reg in (10.0, 1.0)
        ]
    )
    before = frame.copy(deep=True)
    paths = []
    cache = {"test": "cache"}
    selected = np.array([[1.0, 2.0]])

    def load(path):
        paths.append(path.name)
        return cache

    monkeypatch.setattr(embeddings, "load_aggregate_cache", load)
    monkeypatch.setattr(
        embeddings,
        "iter_aggregate_embeddings",
        lambda cache, methods: [
            dict(seed_index=1, rel_reg=1.0, embedding=np.zeros((1, 2))),
            dict(seed_index=0, rel_reg=10.0, embedding=np.zeros((1, 2))),
            dict(seed_index=0, rel_reg=1.0, embedding=selected),
        ],
    )
    actual, actual_cache = embeddings.load_reported_embedding(
        frame, "nCoV", key, rel_reg=1.0, backend="umap", init="random"
    )
    assert actual is selected
    assert actual_cache is cache
    assert paths == [f"{'umap' if has_axis else 'tsne'}-0-random-1.0.npz"]
    assert_frame_equal(frame, before)
    # Two inits and no choice: refuse rather than pick one.
    with pytest.raises(ValueError, match=r"inits \['random', 'spectral'\]; pass init="):
        embeddings.load_reported_embedding(frame, "nCoV", key, rel_reg=1.0, backend="umap")
    # One init in the frame (here a non-random one): it is used without being named.
    paths.clear()
    embeddings.load_reported_embedding(
        frame[frame["init"] == "spectral"], "nCoV", key, rel_reg=1.0, backend="umap"
    )
    assert paths == [f"{'umap' if has_axis else 'tsne'}-0-spectral-1.0.npz"]


def test_embedding_selection_refuses_missing_rows_and_ambiguous_records(monkeypatch):
    frame = pd.DataFrame(
        [
            dict(
                dataset="nCoV",
                method_key="ot",
                backend="tsne",
                seed=0,
                init="random",
                rel_reg=1.0,
                source_cache="cache.npz",
            )
        ]
    )
    kwargs = dict(rel_reg=1.0)
    with pytest.raises(FileNotFoundError, match="no seed-0 reported row"):
        embeddings.load_reported_embedding(frame, "absent", "ot", **kwargs)
    monkeypatch.setattr(embeddings, "load_aggregate_cache", lambda path: {})
    monkeypatch.setattr(
        embeddings,
        "iter_aggregate_embeddings",
        lambda cache, methods: [
            dict(seed_index=0, rel_reg=1.0),
            dict(seed_index=0, rel_reg=1.0),
        ],
    )
    with pytest.raises(ValueError, match="expected one embedding.*found 2"):
        embeddings.load_reported_embedding(frame, "nCoV", "ot", **kwargs)


def test_seed_gallery_draws_two_rows_per_seed_and_labels_each_seed():
    by_seed = {seed: _cached(COLUMNS) for seed in (0, 1, 2)}
    try:
        with layout.paper_style():
            fig = qualitative_composer().seed_panels(
                by_seed,
                cell_type_palette(("a", "b", "c")),
                columns=COLUMNS,
                groups=(("Baseline", 1), ("Integrated", 2)),
            )
        panels = [ax for ax in fig.axes if ax.collections]
        assert len(panels) == 3 * 2 * len(COLUMNS)
        assert sum(bool(ax.get_title()) for ax in panels) == len(COLUMNS)
        labels = [ax.get_ylabel() for ax in panels]
        assert [label for label in labels if label.startswith("Seed")] == [
            "Seed 0",
            "Seed 1",
            "Seed 2",
        ]
        assert labels.count("Batch") == labels.count("Cell type") == 3
        with pytest.raises(ValueError, match="at least one seed"):
            qualitative_composer().seed_panels({}, cell_type_palette(("a",)), columns=COLUMNS)
    finally:
        plt.close("all")


def test_seed_embeddings_read_each_cache_once_and_select_every_seed(monkeypatch):
    frame = pd.DataFrame(
        [
            dict(
                dataset="nCoV",
                method_key="ot",
                backend="tsne",
                seed=seed,
                init="random",
                rel_reg=reg,
                source_cache="cache.npz",
            )
            for seed in (0, 1, 2)
            for reg in (1.0, 10.0)
        ]
    )
    loads = []
    monkeypatch.setattr(
        embeddings, "load_aggregate_cache", lambda path: loads.append(path) or {"c": 1}
    )
    records = [
        dict(seed_index=seed, rel_reg=reg, embedding=np.full((4, 2), seed + reg))
        for seed in (0, 1, 2)
        for reg in (1.0, 10.0)
    ]
    monkeypatch.setattr(embeddings, "iter_aggregate_embeddings", lambda cache, methods: records)
    chosen, cache = embeddings.load_seed_embeddings(
        frame, "nCoV", "ot", rel_reg=1.0, seeds=(0, 1, 2)
    )
    assert cache == {"c": 1} and len(loads) == 1
    assert {seed: float(e[0, 0]) for seed, e in chosen.items()} == {0: 1.0, 1: 2.0, 2: 3.0}
    with pytest.raises(FileNotFoundError, match="seed-5"):
        embeddings.load_seed_embeddings(frame, "nCoV", "ot", rel_reg=1.0, seeds=(5,))
    records[2]["embedding"] = np.zeros((3, 2))
    with pytest.raises(ValueError, match="different numbers of cells"):
        embeddings.load_seed_embeddings(frame, "nCoV", "ot", rel_reg=1.0, seeds=(0, 1))
