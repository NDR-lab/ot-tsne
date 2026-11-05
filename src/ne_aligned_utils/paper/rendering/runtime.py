"""Runtime figure presentation: panel recipe, drawing constants and their snapshots."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ne_aligned_utils.paper import layout, registry
from ne_aligned_utils.paper.rendering import styles


@dataclass(frozen=True)
class RuntimeRenderContext:
    """Panel membership, row names and pairing metadata for the runtime figures."""

    families: Mapping[str, tuple[str, ...]]
    family_order: tuple[str, ...]
    declared_order_families: frozenset[str]
    method_labels: Mapping[str, str]
    method_tools: Mapping[str, str]
    dataset_labels: Mapping[str, str]
    dataset_markers: Mapping[str, str]

    def __post_init__(self):
        object.__setattr__(
            self,
            "families",
            MappingProxyType({family: tuple(members) for family, members in self.families.items()}),
        )
        for name in ("method_labels", "method_tools", "dataset_labels", "dataset_markers"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(self, "family_order", tuple(self.family_order))
        object.__setattr__(self, "declared_order_families", frozenset(self.declared_order_families))


@dataclass(frozen=True)
class RuntimeFigureStyle:
    """Resolved all-method runtime drawing values, separate from timing analysis."""

    palette: Mapping[str, str]
    family_labels: Mapping[str, str]
    phase_families: tuple[str, ...]
    row_spread: float
    row_band: str
    point_size: float
    hollow_size: float
    dataset_legend_pt: float
    phase_legend_pt: float
    legend_pad_in: float
    device_names: tuple[str, str]
    device_bridge_dash: tuple[float, tuple[float, ...]]
    device_mark_pt: float
    device_mark_pad: float
    device_mark_rise: float
    device_key_gap_in: float

    def __post_init__(self):
        for name in ("palette", "family_labels"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        for name in ("phase_families", "device_names"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        offset, pattern = self.device_bridge_dash
        object.__setattr__(self, "device_bridge_dash", (offset, tuple(pattern)))


# Runtime figure recipe

#: Panels as id -> methods. Panel 1 keeps this order; the others sort by geometric-mean runtime,
#: and 3b mirrors 3a so each tool's two forms share a row position.
RUNTIME_FAMILIES: dict[str, tuple[str, ...]] = {
    "1": ("ot", "ot-zerodiag", "ot-selfotmedian", "ot-gramdiag", "ot-knndiag"),
    "2": ("ref", "rc-tsne", "multimap", "bc-tsne"),
    "3a": ("harmony-50", "scanorama-50", "eot-50", "liger-50", "seurat-50", "scvi-50"),
    "3b": ("harmony-2d", "scanorama-2d", "eot-t0", "liger-2d", "seurat-2d", "scvi-2d"),
    "4": ("joint-pca", "joint"),
}
FAMILY_ORDER: tuple[str, ...] = ("1", "2", "3a", "3b", "4")
DECLARED_ORDER_FAMILIES = frozenset({"1"})
FAMILY_LABELS: dict[str, str] = {
    "1": "OT-tSNE (ours)",
    "2": "Integrated visualization",
    "3a": "Integrated embeddings $\\rightarrow$ t-SNE",
    "3b": "Integrated embeddings in 2D",
    "4": "No integration",
}
#: Panels that draw the integration stage (hollow) as well as the total (filled).
PHASE_FAMILIES: tuple[str, ...] = ("1", "3a")

DATASET_MARKERS: dict[str, str] = {
    "nCoV": "o",
    "human_brain": "s",
    "lupus": "^",
    "yao_10x_ent_l2l3": "D",
    "PBMC": "v",
    "mouse_atlas": "P",
    "hematopoietic": "X",
    "hematopoietic_shared": "*",
}

ROW_SPREAD = 0.86
ROW_BAND = "#f4f4f4"
POINT_SIZE = 22.0
HOLLOW_SIZE = 21.0
DATASET_LEGEND_PT = 7.5
PHASE_LEGEND_PT = 7.0
LEGEND_PAD_IN = 0.10

FIGURE_ONE_SIZE = (layout.FULL_WIDTH, 8.6)
FIGURE_ONE_SIZE_NARROW = (layout.FULL_WIDTH * 0.95, 8.6 * 0.95)
FIGURE_TWO_SIZE = (layout.FULL_WIDTH, 2.95)

DEVICE_MARKS_LETTERS: tuple[str, str] = ("CPU", "GPU")
DEVICE_MARKS_STARS: tuple[str, str] = ("*", "**")
DEVICE_NAMES: tuple[str, str] = ("CPU", "GPU")
DEVICE_BRIDGE_DASH: tuple[float, tuple[float, float]] = (0.0, (1.8, 1.4))
DEVICE_MARK_PT = layout.MIN_FONT_PT
DEVICE_MARK_PAD = 2.6
DEVICE_MARK_RISE = 2.2
DEVICE_KEY_GAP_IN = 0.22

#: Device-overlay outputs: file stem -> (CPU/GPU marks, figure size).
DEVICE_VARIANTS: dict[str, tuple[tuple[str, str], tuple[float, float]]] = {
    "runtime_all_methods_device": (DEVICE_MARKS_LETTERS, FIGURE_ONE_SIZE),
    "runtime_all_methods_device_stars": (DEVICE_MARKS_STARS, FIGURE_ONE_SIZE),
    "runtime_all_methods_device_narrow": (DEVICE_MARKS_LETTERS, FIGURE_ONE_SIZE_NARROW),
    "runtime_all_methods_device_stars_narrow": (DEVICE_MARKS_STARS, FIGURE_ONE_SIZE_NARROW),
}

#: OT-tSNE stage bars: (timing column, legend label, colour).
OT_SEGMENTS: tuple[tuple[str, str, str], ...] = (
    ("time_pre_embedding", "Optimal transport", registry.METHODS["ot"].color),
    ("time_embedding", "t-SNE", "#c8b9a6"),
)


def runtime_render_context() -> RuntimeRenderContext:
    keys = [key for members in RUNTIME_FAMILIES.values() for key in members]
    return RuntimeRenderContext(
        families=RUNTIME_FAMILIES,
        family_order=FAMILY_ORDER,
        declared_order_families=DECLARED_ORDER_FAMILIES,
        method_labels={key: registry.METHODS[key].short_label for key in keys},
        method_tools={key: registry.METHODS[key].tool for key in keys},
        dataset_labels=registry.DATASET_LABELS,
        dataset_markers=DATASET_MARKERS,
    )


def runtime_figure_style() -> RuntimeFigureStyle:
    return RuntimeFigureStyle(
        palette=styles.DATASET_COLORS,
        family_labels=FAMILY_LABELS,
        phase_families=PHASE_FAMILIES,
        row_spread=ROW_SPREAD,
        row_band=ROW_BAND,
        point_size=POINT_SIZE,
        hollow_size=HOLLOW_SIZE,
        dataset_legend_pt=DATASET_LEGEND_PT,
        phase_legend_pt=PHASE_LEGEND_PT,
        legend_pad_in=LEGEND_PAD_IN,
        device_names=DEVICE_NAMES,
        device_bridge_dash=DEVICE_BRIDGE_DASH,
        device_mark_pt=DEVICE_MARK_PT,
        device_mark_pad=DEVICE_MARK_PAD,
        device_mark_rise=DEVICE_MARK_RISE,
        device_key_gap_in=DEVICE_KEY_GAP_IN,
    )
