"""nCoV scientific data loading and the shared cell-type palette."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ne_aligned_utils.benchmark.data import load_named_dataset


@dataclass(frozen=True)
class NCovAnalysisData:
    x1: np.ndarray
    x2: np.ndarray
    labels1: np.ndarray
    labels2: np.ndarray
    cell_type_names: tuple[str, ...]
    dataset_names: tuple[str, str]

    @property
    def joint_data(self) -> np.ndarray:
        return np.concatenate([self.x1, self.x2], axis=0)

    @property
    def joint_labels(self) -> np.ndarray:
        return np.concatenate([self.labels1, self.labels2], axis=0)

    @property
    def batch_ids(self) -> np.ndarray:
        return np.concatenate(
            [
                np.zeros(self.x1.shape[0], dtype=int),
                np.ones(self.x2.shape[0], dtype=int),
            ]
        )

    @property
    def short_dataset_names(self) -> tuple[str, str]:
        return tuple(_short_dataset_name(name) for name in self.dataset_names)


def _resolve_ncov_path(data_path: str | Path) -> Path:
    path = Path(data_path)
    repo_root = Path(__file__).resolve().parents[2]
    base_candidates = [path]
    if not path.is_absolute():
        base_candidates.append(repo_root / path)

    candidates: list[Path] = []
    for base in base_candidates:
        candidates.append(base)
        if base.name != "nCoV":
            candidates.append(base / "nCoV")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find nCoV data. Checked: {checked}")


def _short_dataset_name(name: str) -> str:
    return name.split("-")[-1]


def cell_type_palette(cell_type_names: tuple[str, ...], cmap_name: str = "tab20") -> np.ndarray:
    cmap = plt.get_cmap(cmap_name, len(cell_type_names))
    return np.array([cmap(i) for i in range(len(cell_type_names))])


def load_ncov_analysis_data(data_path: str | Path = "data/nCoV") -> NCovAnalysisData:
    resolved_path = _resolve_ncov_path(data_path)
    data_root = resolved_path.parent if resolved_path.name == "nCoV" else resolved_path
    loaded = load_named_dataset("nCoV", data_root=data_root, seed=0)
    if loaded.labels0 is None or loaded.labels1 is None:
        raise ValueError("nCoV cell-type labels are missing or invalid.")
    return NCovAnalysisData(
        x1=np.asarray(loaded.x0),
        x2=np.asarray(loaded.x1),
        labels1=np.asarray(loaded.labels0, dtype=int),
        labels2=np.asarray(loaded.labels1, dtype=int),
        cell_type_names=loaded.label_names,
        dataset_names=loaded.dataset_names,
    )
