"""Method, metric and dataset identities shared by every paper figure.

One entry per method the benchmark can report. A key is the benchmark's own method name; only a
non-default configuration of a method gets a suffixed key (``multimap-k15``). Every figure, the
synthetic and notebook figures included, draws a method with the colour and marker stored here.
The table order is the figure order: families first, then the order of the rows below.
"""

from __future__ import annotations

from dataclasses import dataclass

from ne_aligned_utils import config

#: Families in the order they appear as figure columns and legend blocks.
FAMILY_ORDER: tuple[str, ...] = ("ot", "dedicated", "integration", "reference")

#: Two-line column headings, so the four headings share one title band.
FAMILY_LABELS: dict[str, str] = {
    "ot": "OT-tSNE\n(ours)",
    "dedicated": "Integrated\nvisualization",
    "integration": "Integrated in 2D or\n50D $\\rightarrow$ t-SNE",
    "reference": "No\nintegration",
}

#: Markers drawn as a stroke only: no face to fill and no edge to outline.
UNFILLED_MARKERS = frozenset({"+", "x", "|", "_"})

#: Integration tools in legend order. Seurat sits last, as in the submitted figures; the colour
#: ramp alone would place it between LIGER and scVI.
INTEGRATION_LEGEND_ORDER: tuple[str, ...] = (
    "scanorama",
    "cca",
    "eot",
    "harmony",
    "liger",
    "scvi",
    "seurat",
)


@dataclass(frozen=True)
class MethodStyle:
    """How one method is named and drawn."""

    key: str
    label: str
    family: str
    subfamily: str
    tool: str
    color: str
    marker: str
    size: float = 34.0
    edge_color: str = "#ffffff"
    edge_width: float = 0.4
    alpha: float = 1.0
    #: Name inside a family legend block whose heading already says "OT-tSNE".
    legend_label: str = ""
    #: Compact name for crowded row labels (rank, top-r and runtime figures).
    short_label: str = ""
    #: Benchmark method name; differs from ``key`` only for a non-default configuration.
    method: str = ""
    #: Embedding backend whose cache holds the reported layout.
    backend_source: str = "tsne"

    def __post_init__(self):
        for name in ("legend_label", "short_label"):
            if not getattr(self, name):
                object.__setattr__(self, name, self.label)
        if not self.method:
            object.__setattr__(self, "method", self.key)
        if self.family not in FAMILY_ORDER:
            raise ValueError(f"{self.key}: unknown family {self.family!r}")

    @property
    def unfilled(self) -> bool:
        return self.marker in UNFILLED_MARKERS

    def scatter_kwargs(self, **overrides: object) -> dict[str, object]:
        """Keyword arguments for ``ax.scatter``."""
        kwargs: dict[str, object] = {
            "c": self.color,
            "marker": self.marker,
            "s": self.size,
            "alpha": self.alpha,
            "label": self.label,
        }
        if self.unfilled:
            # The colour rides on the stroke; a heavier line keeps a thin glyph legible.
            kwargs["linewidths"] = 0.9
        else:
            kwargs["edgecolors"] = self.edge_color
            kwargs["linewidths"] = self.edge_width
        kwargs.update(overrides)
        return kwargs


def _ot(key, variant, short, tool, color) -> MethodStyle:
    """An OT-tSNE diagonal ablation: full, legend-block and short names from its variant."""
    return MethodStyle(
        key,
        f"OT-tSNE ({variant})",
        "ot",
        "ot",
        tool,
        color,
        "D",
        legend_label=variant,
        short_label=f"OT-tSNE ({short})",
    )


def _dedicated(key, label, tool, color, marker="v", **kw) -> MethodStyle:
    return MethodStyle(
        key,
        label,
        "dedicated",
        "dedicated",
        tool,
        color,
        marker,
        edge_color="none",
        edge_width=0.0,
        **kw,
    )


def _integration(tool, name, subfamily, color, keys=None) -> tuple[MethodStyle, MethodStyle]:
    """The 50D form (integrate, then t-SNE; circle) and the direct 2D form (square)."""
    high, low = keys or (f"{tool}-50", f"{tool}-2d")
    return (
        MethodStyle(high, f"{name} 50D", "integration", subfamily, tool, color, "o"),
        MethodStyle(low, f"{name} 2D", "integration", subfamily, tool, color, "s"),
    )


_ENTRIES: tuple[MethodStyle, ...] = (
    # OT-tSNE and its diagonal ablations, dark-to-light along one brown ramp.
    MethodStyle(
        "ot",
        "OT-tSNE",
        "ot",
        "ot",
        "ot",
        "#4A1B04",
        "D",
        size=30.0,
        edge_color="#1f1f1f",
        edge_width=0.3,
    ),
    _ot("ot-selfotmedian", "self OT diagonal", "self OT", "ot-selfot", "#993404"),
    _ot("ot-zerodiag", "zero diagonal", "zero", "ot-zero", "#CC4C02"),
    _ot("ot-gramdiag", "Gram diagonal", "Gram", "ot-gram", "#FE9929"),
    _ot("ot-knndiag", "kNN diagonal", "kNN", "ot-knn", "#FEC44F"),
    _ot("ot-zerodiag-split", "zero diagonal, split", "zero, split", "ot-zero-split", "#EC7014"),
    # Methods that integrate while producing the 2D layout.
    _dedicated(
        "multimap-k15",
        "MultiMAP k=15",
        "multimap",
        "#7A0C17",
        marker="^",
        method="multimap",
        backend_source="umap",
    ),
    _dedicated("multimap", "MultiMAP", "multimap", "#D42A2A", marker="^"),
    _dedicated("bc-tsne", "BC t-SNE", "bctsne", "#C51B8A"),
    _dedicated("ref", "Ref t-SNE", "ref", "#F768A1"),
    _dedicated("rc-tsne", "RC t-SNE", "rctsne", "#FBB4B9"),
    # Integrate first, then embed. Each tool's two forms share a colour.
    *_integration("seurat", "Seurat", "anchor", "#5E4FA2"),
    *_integration("liger", "LIGER", "anchor", "#253494"),
    *_integration("scanorama", "Scanorama", "anchor", "#35A2C4"),
    *_integration("harmony", "Harmony", "model", "#2171B5"),
    *_integration("scvi", "scVI", "model", "#8C6BB1"),
    *_integration("eot", "EOT", "ot-baseline", "#005824", keys=("eot-50", "eot-t0")),
    *_integration("cca", "CCA", "spectral", "#1B9E77"),
    # No integration: the floor.
    MethodStyle(
        "joint-pca",
        "Joint PCA",
        "reference",
        "reference",
        "none",
        "#111111",
        "+",
        size=30.0,
        edge_color="none",
        edge_width=0.0,
        alpha=0.85,
    ),
    MethodStyle(
        "joint",
        "Joint t-SNE",
        "reference",
        "reference",
        "none",
        "#B0B0B0",
        "x",
        size=30.0,
        edge_color="none",
        edge_width=0.0,
        alpha=0.85,
    ),
)

METHODS: dict[str, MethodStyle] = {entry.key: entry for entry in _ENTRIES}
if len(METHODS) != len(_ENTRIES):
    raise RuntimeError("duplicate method key in the paper registry")


def resolve(key: str) -> MethodStyle:
    """The entry for ``key``; an unknown method is an error, never a grey fallback."""
    try:
        return METHODS[key]
    except KeyError:
        raise KeyError(f"{key!r} is not in the paper registry; add an entry for it") from None


def method_order(keys=None) -> list[str]:
    """``keys`` (default: all) in figure order: by family, then by table row."""
    wanted = set(METHODS) if keys is None else set(keys)
    for key in wanted:
        resolve(key)
    row = {key: index for index, key in enumerate(METHODS)}
    return sorted(wanted, key=lambda k: (FAMILY_ORDER.index(METHODS[k].family), row[k]))


#: Metric keys in summary-column order, with axis labels. All four are higher-is-better in [0, 1].
METRICS: dict[str, str] = {
    "purity": "Neighborhood purity",
    "kbet": "kBET",
    "silhouette": "Silhouette (ASW)",
    "ilisi": "iLISI",
}

#: Biological-conservation and batch-mixing halves of the score.
BIO_METRICS: tuple[str, ...] = ("purity", "silhouette")
MIXING_METRICS: tuple[str, ...] = ("ilisi", "kbet")

#: Score columns in the benchmark metric tables, mapped to the keys above.
METRIC_SOURCE_COLUMNS: dict[str, str] = {
    "neighborhood_purity_score": "purity",
    "silhouette_score": "silhouette",
    "ilisi_score": "ilisi",
    "kbet_score": "kbet",
}

DATASET_LABELS: dict[str, str] = dict(config.DATASET_LABELS)
