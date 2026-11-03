"""Immutable presentation snapshots for an already selected benchmark comparison."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class MethodPresentation:
    """Only drawing metadata for one identity; not another scientific registry."""

    label: str
    display_label: str
    family: str
    subfamily: str
    scatter: Mapping[str, str | float]
    key: str = ""
    tool: str = ""
    legend_glyph: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "scatter", MappingProxyType(dict(self.scatter)))
        object.__setattr__(self, "legend_glyph", MappingProxyType(dict(self.legend_glyph)))


@dataclass(frozen=True)
class LegendConfig:
    """Current legend reservation and pairing recipe."""

    extra_rows: int
    dimension_key_on_own_row: bool
    integration_tool_order: tuple[str, ...]
    dimension_size: float = 34.0


@dataclass(frozen=True)
class RenderContext:
    """Selected identities and presentation values, never selection/aggregation policy.

    All maps are detached read-only snapshots. No module, mutable registry, or
    callback closing over module globals is stored here. Dataset order describes
    panel composition only; it does not choose a scientific comparison pool.
    """

    methods: Mapping[str, MethodPresentation]
    method_order: tuple[str, ...]
    family_order: tuple[str, ...]
    family_labels: Mapping[str, str]
    dataset_order: tuple[str, ...]
    dataset_labels: Mapping[str, str]
    legend: LegendConfig
    output_root: Path

    def __post_init__(self):
        for name in ("methods", "family_labels", "dataset_labels"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        for name in ("method_order", "family_order", "dataset_order"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "output_root", Path(self.output_root))

    def order_methods(self, keys: Sequence[str]) -> list[str]:
        """Order the supplied identities without adding any competitors."""
        keys = set(keys)
        for key in keys:
            if key not in self.methods:
                raise KeyError(key)
        return [key for key in self.method_order if key in keys]

    def scatter_kwargs(self, key: str, **overrides) -> dict:
        return dict(self.methods[key].scatter) | overrides
