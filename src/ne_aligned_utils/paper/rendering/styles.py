"""Presentation constants for paper figures; no scientific selection lives here."""

from __future__ import annotations

from matplotlib.colors import LinearSegmentedColormap

from ne_aligned_utils.paper import layout, registry

# Datasets

#: Categorical dataset colours for scatter marks (runtime figure).
DATASET_COLORS: dict[str, str] = {
    "nCoV": "#2a78d6",
    "human_brain": "#eb6834",
    "lupus": "#1baf7a",
    "yao_10x_ent_l2l3": "#eda100",
    "PBMC": "#e87ba4",
    "mouse_atlas": "#008300",
    "hematopoietic": "#4a3aa7",
    "hematopoietic_shared": "#e34948",
}

#: Ordered violet-to-orange dataset ramp for stacked bars (top-r figures).
DATASET_RAMP: dict[str, str] = {
    "nCoV": "#3f2b7a",
    "human_brain": "#2a55b0",
    "lupus": "#1683c0",
    "yao_10x_ent_l2l3": "#0f9e8e",
    "PBMC": "#4aab3a",
    "mouse_atlas": "#c9a227",
    "hematopoietic": "#e2671c",
}

#: Dataset names rotated into one panel's height on the right of a metric grid.
GRID_STRIP_LABELS: dict[str, str] = dict(registry.DATASET_LABELS) | {
    "hematopoietic": "Hemato-\npoietic",
    "hematopoietic_shared": "Hemato.\n(shared)",
}

# Ranks

#: Rank panels: score block -> heading.
RANK_BLOCKS: dict[str, str] = {
    "all": "All four metrics",
    "bio": "Bio conservation",
    "mixing": "Batch mixing",
}

RANK_RAMP = LinearSegmentedColormap.from_list("rank", ["#4A1B04", "#CC4C02", "#FE9929"])
RANK_MARK_SIZE = 46
RANK_RAMP_OFFSET_PT = 2.0
RANK_RAMP_GAP_PT = 0.0
RANK_RAMP_THICKNESS_PT = 4.0
RANK_RAMP_TICK_PAD_PT = 2.5
RANK_PANEL_WIDTH = 1.353
RANK_PANEL_GAP = 0.059
RANK_SIDE_BLOCKS = 1.323
RANK_VALUE_FORMAT = "{:.1f}"
RANK_LABEL_PAD_PT = 3.0
RANK_EDGE_PAD_IN = 0.06

# Top-r bars

BAR_LEGEND_LOC = "lower right"
BAR_PANEL_FRACTION = 0.70
BAR_LABEL_CLEARANCE = 0.30
BAR_LEGEND_PAD = 0.2
#: (count column, hatch): bio placings hatched, mixing placings solid.
BAR_PARTS = (("n_top_k_bio", "////"), ("n_top_k_mixing", None))

# Qualitative galleries

#: Shared column headings and the groups they fall into, left to right.
QUALITATIVE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("joint", "Joint t-SNE"),
    ("ot", "OT-tSNE"),
    ("multimap", "MultiMAP"),
    ("rc-tsne", "RC t-SNE"),
    ("seurat-50", "Seurat"),
    ("liger-50", "LIGER"),
    ("scanorama-50", "Scanorama"),
    ("scvi-50", "scVI"),
    ("eot-50", "EOT"),
)
QUALITATIVE_GROUPS: tuple[tuple[str, int], ...] = (
    ("Unintegrated", 1),
    ("Integrated visualisation", 3),
    ("Integrated embeddings in 50D → t-SNE", 5),
)

QUALITATIVE_ROWS: tuple[tuple[str, str], ...] = (
    ("batch", "Batch"),
    ("cell type", "Cell type"),
)
QUALITATIVE_MARGINS: dict[str, float] = {
    **layout.GALLERY_MARGINS,
    "left": 0.05,
    "right": 0.20,
    "top": 0.30,
}
QUALITATIVE_BATCH_COLOURS: tuple[str, str] = ("#377eb8", "#e66101")
QUALITATIVE_RULE_AT = 0.5
QUALITATIVE_LEGEND_GAP = 0.02
QUALITATIVE_BATCH_LABELS: tuple[str, str] = ("Batch 0", "Batch 1")
QUALITATIVE_HANDLETEXTPAD = 0.25
QUALITATIVE_COLUMNSPACING = 1.2
QUALITATIVE_GROUP_BAND = 0.08
QUALITATIVE_GROUP_LIFT = 0.055
QUALITATIVE_GROUP_TICK = 0.035
QUALITATIVE_GROUP_TEXT_GAP = 0.025
QUALITATIVE_GROUP_PT = 7.0
#: Point size at the reference cell count; galleries scale it by sqrt(POINT_REF_N / n).
POINT_REF_SIZE = 3.5
POINT_REF_N = 4060
PANEL_TRIM = 0.002
#: Datasets in the stacked gallery, top to bottom (lupus is left out of the stack).
STACKED_GALLERY_DATASETS: tuple[str, ...] = (
    "nCoV",
    "hematopoietic",
    "mouse_atlas",
    "human_brain",
    "PBMC",
    "yao_10x_ent_l2l3",
)
