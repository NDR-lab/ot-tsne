"""Saved-array methods schematic: composition, preflight and export."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

from ne_aligned_utils.paper.inputs.methods import ARRAY_NAMES, load_method_arrays


@dataclass(frozen=True)
class MethodsComposer:
    """The notebook's fixed presentation, detached from numerical method stages."""

    blue: str = "#1F77B4"
    orange: str = "#FF7F0E"
    navy: str = "#183B56"
    text: str = "#17212B"
    muted: str = "#687684"
    heatmap_cmap: str = "magma"

    @contextmanager
    def style(self):
        """Scope the original notebook style without changing surrounding figures."""
        with plt.rc_context():
            self.set_style()
            yield

    def compose(self, signal0, signal1, plan, affinity, sparse, embedding) -> Figure:
        """Render supplied arrays; no OT, sparsification or embedding is recomputed."""
        with self.style():
            return self._compose(signal0, signal1, plan, affinity, sparse, embedding)

    def clean_axis(self, ax: plt.Axes) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def draw_boundaries(self, ax: plt.Axes, n0: int, total: int) -> None:
        boundary = n0 - 0.5
        ax.axhline(boundary, color="white", lw=1.25)
        ax.axvline(boundary, color="white", lw=1.25)
        ax.add_patch(
            Rectangle((-0.5, -0.5), total, total, fill=False, edgecolor=self.navy, lw=0.85)
        )

    def heatmap_norm(self, values: np.ndarray) -> mcolors.Normalize:
        """Expand the central value range so diffuse EOT structure stays visible."""
        positive = np.asarray(values)[np.asarray(values) > 0]
        (low, high) = np.quantile(positive, (0.005, 0.995))
        if np.isclose(low, high):
            (low, high) = (float(positive.min()), float(positive.max()))
        return mcolors.Normalize(vmin=low, vmax=high, clip=True)

    def draw_blocks(self, ax: plt.Axes, plan: np.ndarray) -> None:
        """Draw the symbolic diagonal blocks with the actual plans off diagonal."""
        (n0, n1) = plan.shape
        total = n0 + n1
        split = n0 / total
        diag_fill = "#FFFFFF"
        plan_norm = self.heatmap_norm(plan)
        ax.imshow(
            plan,
            cmap=self.heatmap_cmap,
            norm=plan_norm,
            interpolation="nearest",
            origin="upper",
            extent=(split, 1.0, 1.0 - split, 1.0),
            aspect="auto",
        )
        ax.imshow(
            plan.T,
            cmap=self.heatmap_cmap,
            norm=plan_norm,
            interpolation="nearest",
            origin="upper",
            extent=(0.0, split, 0.0, 1.0 - split),
            aspect="auto",
        )
        diagonal_blocks = (
            (0.0, 1.0 - split, split, split, "$S_0$"),
            (split, 0.0, 1.0 - split, 1.0 - split, "$S_1$"),
        )
        for x, y, width, height, label in diagonal_blocks:
            ax.add_patch(
                Rectangle((x, y), width, height, facecolor=diag_fill, edgecolor=self.navy, lw=1.0)
            )
            ax.text(
                x + width / 2,
                y + height / 2,
                label,
                ha="center",
                va="center",
                fontsize=min(25.0, max(16.0, 110.0 * min(width, height))),
                color=self.text,
            )
        for x, y, width, height, label in (
            (split, 1.0 - split, 1.0 - split, split, "$P$"),
            (0.0, 0.0, split, 1.0 - split, "$P^{\\mathsf{T}}$"),
        ):
            ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=self.navy, lw=1.0))
            ax.text(
                x + width / 2,
                y + height / 2,
                label,
                ha="center",
                va="center",
                fontsize=min(27.0, max(16.0, 115.0 * min(width, height))),
                color="white",
                path_effects=[path_effects.withStroke(linewidth=1.6, foreground="black")],
            )
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        self.clean_axis(ax)

    def set_style(self) -> None:
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "mathtext.fontset": "stixsans",
                "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                "figure.constrained_layout.use": False,
                "font.size": 9,
                "axes.labelsize": 9,
                "legend.fontsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "axes.titlepad": 6,
                "axes.linewidth": 0.8,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
            }
        )

    def set_layout(self, fig: plt.Figure, axes: np.ndarray, n0: int, n1: int) -> None:
        """A single reading line, with the portrait plan sized to its actual shape."""
        fig.set_size_inches(16.0, 4.0)
        (a, b, c, d, e, f) = axes
        matrix_height = 0.12 * 16.0 / 4.0
        bottom = 0.48 - matrix_height / 2
        positions = (
            (0.018, bottom, 0.142, matrix_height),
            (0.204, bottom, 0.06, matrix_height),
            (0.323, bottom, 0.12, matrix_height),
            (0.5, bottom, 0.12, matrix_height),
            (0.678, bottom, 0.12, matrix_height),
            (0.858, bottom, 0.12, matrix_height),
        )
        for ax, pos in zip(axes, positions):
            ax.set_position(pos)
        for ax in axes:
            for annotation in ax.texts:
                annotation.set_fontsize(annotation.get_fontsize() * 1.5)
        a.set_xlim(-1.8, 1.8)
        a.set_ylim(-2.52, 2.52)
        a.set_aspect("equal", adjustable="box")
        a.get_legend().remove()
        batch_offsets = np.array([[0.4, 0.88], [-0.4, -0.88]])
        angle = np.deg2rad(-25)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for collection in a.collections:
            if isinstance(collection, LineCollection):
                collection.set_segments(
                    (np.asarray(collection.get_segments()) + batch_offsets) @ rotation.T
                )
                colors = collection.get_colors()
                colors[:, :3] = mcolors.to_rgb("#53616B")
                colors[:, 3] = 0.1
                collection.set_colors(colors)
                collection.set_linewidths(collection.get_linewidths() * np.array(0.85))
            else:
                batch_index = 0 if collection.get_label() == "Batch 0" else 1
                collection.set_offsets(
                    (np.asarray(collection.get_offsets()) + batch_offsets[batch_index]) @ rotation.T
                )
                collection.set_sizes([6.5])
                collection.set_linewidths([0.12])
        b.set_xlabel("")
        b.set_ylabel("")
        a.text(
            -1.95,
            1.18,
            "Batch 0",
            ha="center",
            va="center",
            rotation=90,
            color=self.blue,
            fontsize=16.5,
            clip_on=False,
        )
        a.text(
            -1.95,
            -1.18,
            "Batch 1",
            ha="center",
            va="center",
            rotation=90,
            color=self.orange,
            fontsize=16.5,
            clip_on=False,
        )
        fig.canvas.draw()
        (arrow_y, label_y) = (0.48, 0.493)
        arrow_gap = 0.006

        def visible_edge(ax, side):
            """Visible plot edge along the arrow corridor, in figure coordinates."""
            if ax in (a, f):
                points = []
                for collection in ax.collections:
                    if isinstance(collection, LineCollection):
                        continue
                    xy = fig.transFigure.inverted().transform(
                        ax.transData.transform(np.asarray(collection.get_offsets()))
                    )
                    near = np.abs(xy[:, 1] - arrow_y) <= 0.035
                    xy = xy[near] if near.any() else xy
                    radius = (
                        np.sqrt(np.max(collection.get_sizes())) / 2
                        + np.max(collection.get_linewidths()) / 2
                    ) / (72 * fig.get_figwidth())
                    points.append(xy[:, 0] + (radius if side == "right" else -radius))
                values = np.concatenate(points)
                return values.max() if side == "right" else values.min()
            box = ax.get_position()
            return box.x1 if side == "right" else box.x0 - 0.0045

        operation_labels = (
            "=",
            "Assemble",
            "Fill\ndiagonal\nblocks",
            "Sparsify",
            "Embed",
        )
        for left, right, label in zip(axes[:-1], axes[1:], operation_labels):
            probe = fig.text(0, 0, label, fontsize=15, linespacing=1.15)
            fig.canvas.draw()
            label_width = probe.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width
            probe.remove()
            arrow_length = max(0.0392, label_width + 0.002)
            shift = (
                visible_edge(left, "right")
                + 2 * arrow_gap
                + arrow_length
                - visible_edge(right, "left")
            )
            box = right.get_position(original=True)
            right.set_position((box.x0 + shift, box.y0, box.width, box.height))
            fig.canvas.draw()
        edge_box, plan_box = a.get_position(), b.get_position()
        group_left = edge_box.x0 - 0.022
        group_right = plan_box.x1
        group_center = (group_left + group_right) / 2
        heading_y = 0.825
        heading_line_step = 16.5 * 1.05 / (72 * fig.get_figheight())
        heading = fig.text(
            group_center,
            heading_y,
            "Cross-batch transport plan",
            ha="center",
            va="baseline",
            fontsize=16.5,
            color=self.text,
        )
        fig.canvas.draw()
        heading_box = heading.get_window_extent(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted()
        )
        fig.text(
            heading_box.x0 - 0.009,
            heading_y,
            "a",
            ha="right",
            va="baseline",
            fontsize=16.5,
            weight="bold",
            color=self.text,
        )
        for ax, title in ((a, "as edges"), (b, "as matrix")):
            box = ax.get_position()
            fig.text(
                (box.x0 + box.x1) / 2,
                heading_y - heading_line_step,
                title,
                ha="center",
                va="baseline",
                fontsize=14,
                color=self.text,
            )

        for ax, letter, title in zip(
            (c, d, e, f),
            "bcde",
            (
                "Block\nconstruction",
                "Dense\naffinity",
                "Sparsified\naffinity",
                "t-SNE\noptimization",
            ),
        ):
            box = ax.get_position()
            heading_lines = [
                fig.text(
                    (box.x0 + box.x1) / 2,
                    heading_y - line_index * heading_line_step,
                    line,
                    ha="center",
                    va="baseline",
                    fontsize=16.5,
                    color=self.text,
                )
                for line_index, line in enumerate(title.splitlines())
            ]
            fig.canvas.draw()
            heading_left = min(
                line.get_window_extent(fig.canvas.get_renderer())
                .transformed(fig.transFigure.inverted())
                .x0
                for line in heading_lines
            )
            letter_right = min(box.x0 - 0.004, heading_left - 0.009)
            fig.text(
                letter_right,
                heading_y,
                letter,
                ha="right",
                va="baseline",
                fontsize=16.5,
                weight="bold",
                color=self.text,
            )

        def strip(x, y, width, height, color):
            fig.add_artist(
                Rectangle(
                    (x, y),
                    width,
                    height,
                    transform=fig.transFigure,
                    facecolor=color,
                    edgecolor="none",
                    clip_on=False,
                )
            )

        for ax in (b, c, d, e):
            box = ax.get_position()
            if ax is b:
                strip(box.x0, box.y1 + 0.012, box.width, 0.008, self.orange)
                strip(box.x0 - 0.0045, box.y0, 0.0018, box.height, self.blue)
            else:
                split = n0 / (n0 + n1)
                strip(box.x0, box.y1 + 0.012, box.width * split, 0.008, self.blue)
                strip(
                    box.x0 + box.width * split,
                    box.y1 + 0.012,
                    box.width * (1 - split),
                    0.008,
                    self.orange,
                )
                strip(
                    box.x0 - 0.0045,
                    box.y0 + box.height * (1 - split),
                    0.0018,
                    box.height * split,
                    self.blue,
                )
                strip(box.x0 - 0.0045, box.y0, 0.0018, box.height * (1 - split), self.orange)
        for left, right, label in zip(axes[:-1], axes[1:], operation_labels):
            start = visible_edge(left, "right") + arrow_gap
            end = visible_edge(right, "left") - arrow_gap
            assert end > start, "Insufficient room for a procedure arrow"
            if left is a:
                fig.text(
                    (start + end) / 2,
                    arrow_y,
                    "$=$",
                    fontsize=25,
                    ha="center",
                    va="center",
                    color=self.text,
                )
                continue
            fig.add_artist(
                FancyArrowPatch(
                    (start, arrow_y),
                    (end, arrow_y),
                    transform=fig.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=1.05,
                    color=self.text,
                    shrinkA=0,
                    shrinkB=0,
                )
            )
            fig.text(
                (start + end) / 2,
                label_y,
                label,
                fontsize=15,
                ha="center",
                va="bottom",
                color=self.text,
                linespacing=1.15,
            )

    def _compose(
        self,
        signal0: np.ndarray,
        signal1: np.ndarray,
        plan: np.ndarray,
        affinity: np.ndarray,
        sparse: np.ndarray,
        embedding: np.ndarray,
    ) -> Figure:
        """Render supplied pipeline results without recomputing any method stages."""
        (n0, n1) = (len(signal0), len(signal1))
        plt.rcParams["figure.dpi"] = 100
        fig = Figure(figsize=(16.4, 3.75), dpi=100)
        FigureCanvasAgg(fig)
        axes = fig.subplots(
            1,
            6,
            gridspec_kw={"width_ratios": [1.3, 1.1, 1.12, 1.12, 1.12, 1.3], "wspace": 0.08},
        )
        fig.patch.set_facecolor("white")
        (ax_a, ax_b, ax_c, ax_d, ax_e, ax_f) = axes
        visible = np.zeros(plan.shape, dtype=bool)
        row_targets = np.argsort(plan, axis=1)[:, -min(2, n1) :]
        visible[np.arange(n0)[:, None], row_targets] = True
        column_sources = np.argsort(plan, axis=0)[-min(2, n0) :, :]
        visible[column_sources, np.arange(n1)[None, :]] = True
        assert visible.any(axis=0).all() and visible.any(axis=1).all()
        (sources, targets) = np.nonzero(visible)
        masses = plan[sources, targets]
        weights = masses / masses.max()
        segments = np.stack((signal0[sources, :2], signal1[targets, :2]), axis=1)
        colors = np.tile(mcolors.to_rgba(self.muted), (len(sources), 1))
        colors[:, 3] = 0.065
        ax_a.add_collection(
            LineCollection(
                segments, colors=colors, linewidths=0.65 * weights, zorder=0.5, rasterized=True
            )
        )
        for signal, color, label, point_size, alpha, edgecolor, linewidth in (
            (signal0, self.blue, "Batch 0", 12.0, 0.84, "white", 0.18),
            (signal1, self.orange, "Batch 1", 12.0, 0.86, "white", 0.18),
        ):
            ax_a.scatter(
                signal[:, 0],
                signal[:, 1],
                s=point_size,
                color=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
                rasterized=True,
            )
        ax_a.set_xlim(-2.48, 2.48)
        ax_a.set_aspect("equal", adjustable="datalim")
        ax_a.set_axis_off()
        ax_a.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.17),
            ncol=2,
            frameon=False,
            handlelength=0.8,
            handletextpad=0.35,
            columnspacing=1.0,
            fontsize=9,
        )
        plan_norm = self.heatmap_norm(plan)
        ax_b.imshow(
            plan, cmap=self.heatmap_cmap, norm=plan_norm, interpolation="nearest", aspect="auto"
        )
        ax_b.set_xlabel("Batch 1", fontsize=8.5, color=self.muted, labelpad=4)
        ax_b.set_ylabel("Batch 0", fontsize=8.5, color=self.muted, labelpad=4)
        ax_b.text(
            0.5,
            0.5,
            "$P$",
            transform=ax_b.transAxes,
            ha="center",
            va="center",
            fontsize=31,
            color="white",
            alpha=1.0,
            path_effects=[path_effects.withStroke(linewidth=1.6, foreground="black")],
        )
        self.clean_axis(ax_b)
        for spine in ax_b.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor(self.navy)
        ax_b.set_box_aspect(n0 / n1)
        self.draw_blocks(ax_c, plan)
        dense_norm = self.heatmap_norm(affinity)
        ax_d.imshow(affinity, cmap=self.heatmap_cmap, norm=dense_norm, interpolation="nearest")
        self.draw_boundaries(ax_d, n0, n0 + n1)
        self.clean_axis(ax_d)
        sparse_cmap = mcolors.ListedColormap(["#FFFFFF", "#6A1B6A"])
        ax_e.imshow(sparse > 0, cmap=sparse_cmap, interpolation="nearest", vmin=0, vmax=1)
        self.draw_boundaries(ax_e, n0, n0 + n1)
        self.clean_axis(ax_e)
        draw_order = np.random.RandomState(0).permutation(n0 + n1)
        batch_colors = np.asarray([self.blue] * n0 + [self.orange] * n1)
        display_embedding = np.column_stack((embedding[:, 1], -embedding[:, 0]))
        ax_f.scatter(
            display_embedding[draw_order, 0],
            display_embedding[draw_order, 1],
            s=6.0,
            color=batch_colors[draw_order],
            edgecolor="none",
            linewidth=0.0,
            alpha=0.76,
            rasterized=True,
        )
        ax_f.set_aspect("equal", adjustable="datalim")
        self.clean_axis(ax_f)
        fig.subplots_adjust(left=0.028, right=0.988, top=0.76, bottom=0.18)
        self.set_layout(fig, axes, n0, n1)
        return fig


def save_method_figure(
    fig, *, output_pdf: Path, output_png: Path, composer: MethodsComposer | None = None
) -> None:
    """Preserve the methods notebook's separate 300-DPI tight-bbox export contract."""
    composer = MethodsComposer() if composer is None else composer
    output_pdf, output_png = Path(output_pdf), Path(output_png)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        with composer.style():
            fig.savefig(output_pdf, dpi=300, bbox_inches="tight", pad_inches=0.08)
            fig.savefig(output_png, dpi=300, bbox_inches="tight", pad_inches=0.08)
    finally:
        plt.close(fig)


def create_figure(signal0, signal1, plan, affinity, sparse, embedding, *, output_pdf, output_png):
    """Compose and save the methods figure from its six stage arrays (used by the notebook)."""
    composer = MethodsComposer()
    fig = composer.compose(signal0, signal1, plan, affinity, sparse, embedding)
    save_method_figure(fig, output_pdf=output_pdf, output_png=output_png, composer=composer)


@dataclass(frozen=True)
class PreparedMethods:
    source: Path
    output: Path
    source_hash: str
    arrays: dict[str, np.ndarray]


def preflight_methods(source: Path, output: Path) -> PreparedMethods:
    """Read and validate the saved stages before creating any output."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source.parent):
        raise ValueError("methods output must not overlap its saved-array input directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output must be new or empty: {output}")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    arrays = load_method_arrays(source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != before:
        raise RuntimeError("methods arrays changed during preflight")
    return PreparedMethods(source, output, before, arrays)


def build_methods(prepared: PreparedMethods) -> Path:
    """Write the original PNG/PDF contract and a success-only manifest."""
    if prepared.output.exists() and (
        not prepared.output.is_dir() or any(prepared.output.iterdir())
    ):
        raise FileExistsError(f"output must be new or empty: {prepared.output}")
    if hashlib.sha256(prepared.source.read_bytes()).hexdigest() != prepared.source_hash:
        raise RuntimeError("methods arrays changed after preflight")
    composer = MethodsComposer()
    figure = composer.compose(*(prepared.arrays[name] for name in ARRAY_NAMES))
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        save_method_figure(
            figure,
            output_pdf=prepared.output / "method.pdf",
            output_png=prepared.output / "method.png",
            composer=composer,
        )
    finally:
        if epoch is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = epoch
    if hashlib.sha256(prepared.source.read_bytes()).hexdigest() != prepared.source_hash:
        raise RuntimeError("methods arrays changed during rendering")
    outputs = {}
    for name in ("method.pdf", "method.png"):
        path = prepared.output / name
        record = dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size
        )
        if path.suffix == ".png":
            with Image.open(path) as image:
                record["dimensions_px"] = list(image.size)
                record["dpi"] = list(image.info.get("dpi", ()))
        outputs[name] = record
    manifest = dict(
        schema_version=1,
        families=["methods"],
        source_arrays=str(prepared.source),
        source_sha256=prepared.source_hash,
        source_date_epoch="0",
        arrays={
            name: dict(
                shape=list(value.shape),
                dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            )
            for name, value in prepared.arrays.items()
        },
        environment={
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name) for name in ("numpy", "matplotlib", "pillow")
            },
        },
        outputs=outputs,
    )
    path = prepared.output / "figure-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
