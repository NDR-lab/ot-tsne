"""Every metric row says which run it came from: its settings and a readable configuration."""

import json

import pandas as pd

from ne_aligned_utils.benchmark.pipeline import configuration_label, with_run_settings


def test_configuration_names_what_tells_runs_apart():
    assert configuration_label("seurat-50") == "seurat-50"
    assert configuration_label("eot-50", rel_reg=1.0) == "eot-50 rel_reg=1"
    assert (
        configuration_label("ot", rel_reg=0.1, effective_k=30.0, graph_symmetrization="or")
        == "ot rel_reg=0.1 k=30 sym=or"
    )
    assert (
        configuration_label("multimap", method_settings={"native_profile": True})
        == "multimap native"
    )
    assert (
        configuration_label(
            "multimap", method_settings={"native_profile": False, "n_neighbors": 15}
        )
        == "multimap k=15"
    )
    assert configuration_label("rc-tsne", method_settings={"beta": 0.0001}) == "rc-tsne beta=0.0001"
    # A non-random OT-tSNE init is named; random, the benchmark's default, is not.
    for init in ("spectral", "eot", "pca"):
        assert (
            configuration_label(
                "ot-knndiag",
                rel_reg=1.0,
                effective_k=15,
                method_settings={"ot_initialization": init},
            )
            == f"ot-knndiag rel_reg=1 k=15 init={init}"
        )
    assert (
        configuration_label("ot", rel_reg=1.0, method_settings={"ot_initialization": "random"})
        == "ot rel_reg=1"
    )
    # eot-t0 shares the OT settings but never reads the init.
    assert (
        configuration_label("eot-t0", rel_reg=1.0, method_settings={"ot_initialization": "pca"})
        == "eot-t0 rel_reg=1"
    )
    # A method without a setting never shows it, even if the column holds a placeholder.
    assert configuration_label("seurat-50", graph_symmetrization="or") == "seurat-50"


def test_older_tables_get_settings_from_task_metadata():
    metrics = pd.DataFrame(
        dict(
            method=["multimap", "multimap", "seurat-50"],
            arm_id=["mm-a", "mm-b", "unlisted"],
            rel_reg=float("nan"),
            effective_k=float("nan"),
        )
    )
    tasks = pd.DataFrame(
        dict(
            arm_id=["mm-a", "mm-b"],
            method_settings=[
                json.dumps({"native_profile": True}),
                json.dumps({"native_profile": False, "n_neighbors": 15}),
            ],
        )
    )
    filled = with_run_settings(metrics, tasks)
    assert filled["configuration"].tolist()[:2] == ["multimap native", "multimap k=15"]
    # A row whose arm is not listed stays unresolved rather than guessed.
    assert pd.isna(filled["method_settings"].iloc[2]) and pd.isna(filled["configuration"].iloc[2])
    # Rows that already carry settings are left as they are.
    again = with_run_settings(filled, tasks.assign(method_settings="{}"))
    assert again["configuration"].tolist()[:2] == ["multimap native", "multimap k=15"]
