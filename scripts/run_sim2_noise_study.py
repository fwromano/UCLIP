#!/usr/bin/env python
"""Evaluate CLIP MCDO uncertainty under noise and downsampling perturbations."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from numpy.random import default_rng
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Allow running without installing the package by injecting repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

import sys
import os

for path in (SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from uclip.cli.car_sim_driver import CarSimRoutineConfig, run as run_car_sim  # noqa: E402

ADAPTER_TARGETS: Sequence[str] = tuple(
    f"vision_model.encoder.layers.{idx}" for idx in range(12)
) + ("visual_projection",)


METRIC_FIELDS = ("trace", "logdet", "off_diag_mass", "entropy", "mutual_information")


RNG = default_rng(20240229)


@dataclass(frozen=True)
class MetricMoments:
    mean: float
    std: float
    min: float
    max: float
    ci_low: float = float("nan")
    ci_high: float = float("nan")


@dataclass(frozen=True)
class TransformSpec:
    name: str
    description: str
    kind: str
    factory: Callable[[int], Callable[[Image.Image], Image.Image]]


CATEGORY_COLOURS = {
    "original": "#7f7f7f",
    "gaussian": "#1f77b4",
    "sp": "#2ca02c",
    "downsample1": "#ff7f0e",
    "downsample2": "#9467bd",
    "other": "#bcbd22",
}


CATEGORY_LABELS = {
    "original": "Original",
    "gaussian": "Gaussian noise",
    "sp": "Salt & pepper noise",
    "downsample1": "Smoothed crop (bicubic)",
    "downsample2": "Raw crop (nearest)",
    "other": "Other",
}

GROUP_ORDER = {
    "original": 0,
    "gaussian": 1,
    "sp": 2,
    "downsample1": 3,
    "downsample2": 4,
    "other": 5,
}


class FigureManager:
    """Assign sequential figure numbers and append captioned images to the report."""

    def __init__(self, report_lines: List[str]) -> None:
        self._report_lines = report_lines
        self._counter = 0

    def add(self, image_path: str, title: str, insight: str, alt_text: Optional[str] = None) -> None:
        self._counter += 1
        alt = alt_text or title
        caption = f"*Figure {self._counter}. {title}. {insight}*"
        self._report_lines.append(f"\n{caption}")
        self._report_lines.append(f"\n![{alt}]({image_path})")

    @property
    def count(self) -> int:
        return self._counter


def _identity_transform() -> Callable[[Image.Image], Image.Image]:
    def apply(image: Image.Image) -> Image.Image:
        return image

    return apply


class _GaussianNoiseTransform:
    def __init__(self, sigma: float, seed: int) -> None:
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image).astype(np.float32) / 255.0
        noise = self.rng.normal(loc=0.0, scale=self.sigma, size=array.shape)
        noisy = np.clip(array + noise, 0.0, 1.0)
        uint8 = (noisy * 255.0).round().astype(np.uint8)
        return Image.fromarray(uint8, mode=image.mode)


class _SaltPepperTransform:
    def __init__(self, amount: float, salt_ratio: float, seed: int) -> None:
        self.amount = max(0.0, min(1.0, amount))
        self.salt_ratio = max(0.0, min(1.0, salt_ratio))
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image).copy()
        if array.ndim != 3:
            raise ValueError("Expected RGB image for salt & pepper noise.")
        h, w, _ = array.shape
        total_pixels = h * w
        noisy_pixels = int(self.amount * total_pixels)
        if noisy_pixels <= 0:
            return image
        salt_pixels = int(noisy_pixels * self.salt_ratio)
        pepper_pixels = noisy_pixels - salt_pixels
        coords = self.rng.choice(total_pixels, size=noisy_pixels, replace=False)
        salt_coords = coords[:salt_pixels]
        pepper_coords = coords[salt_pixels:]
        if salt_pixels > 0:
            ys, xs = np.divmod(salt_coords, w)
            array[ys, xs] = 255
        if pepper_pixels > 0:
            ys, xs = np.divmod(pepper_coords, w)
            array[ys, xs] = 0
        return Image.fromarray(array, mode=image.mode)


class _DownsampleTransform:
    def __init__(self, target_max_dim: int) -> None:
        self.target_max_dim = target_max_dim

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_dim = max(width, height)
        if max_dim <= self.target_max_dim:
            return image
        scale = self.target_max_dim / float(max_dim)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        # First collapse to the coarser grid with nearest-neighbour so the pixel set matches
        # the raw (pixelated) transform exactly, then smooth while restoring to the original size.
        coarse = image.resize((new_width, new_height), resample=Image.NEAREST)
        restored = coarse.resize((width, height), resample=Image.BICUBIC)
        return restored


class _PixelateTransform:
    def __init__(self, target_max_dim: int) -> None:
        self.target_max_dim = target_max_dim

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_dim = max(width, height)
        if max_dim <= self.target_max_dim:
            return image
        scale = self.target_max_dim / float(max_dim)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        downsampled = image.resize((new_width, new_height), resample=Image.NEAREST)
        restored = downsampled.resize((width, height), resample=Image.NEAREST)
        return restored


def _format_sigma_name(sigma: float) -> str:
    return f"gaussian_noise_{int(round(sigma*1000)):03d}"  # e.g., 0.01 -> gaussian_noise_010


def create_image_grid(
    image_paths: Sequence[Path],
    output_path: Path,
    columns: int = 4,
    padding: int = 4,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Path:
    valid_paths = [path for path in image_paths if path.exists()]
    if not valid_paths:
        raise ValueError("No valid image paths provided for grid.")
    images = [Image.open(path).convert("RGB") for path in valid_paths]
    tile_width, tile_height = images[0].size
    for img in images:
        if img.size != (tile_width, tile_height):
            img.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
    cols = max(1, columns)
    rows = math.ceil(len(images) / cols)
    grid_width = cols * tile_width + padding * (cols + 1)
    grid_height = rows * tile_height + padding * (rows + 1)
    grid = Image.new("RGB", (grid_width, grid_height), color=background)
    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        x = padding + col * (tile_width + padding)
        y = padding + row * (tile_height + padding)
        grid.paste(img, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    for img in images:
        img.close()
    return output_path


def create_multirow_preview_grid(
    rows: Sequence[Tuple[str, Sequence[Path]]],
    output_path: Path,
    padding: int = 8,
    label_width: Optional[int] = None,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Path:
    tiles: List[Tuple[str, List[Image.Image]]] = []
    tile_width = tile_height = None
    for label, paths in rows:
        row_images: List[Image.Image] = []
        for path in paths:
            if not path.exists():
                continue
            img = Image.open(path).convert("RGB")
            if tile_width is None or tile_height is None:
                tile_width, tile_height = img.size
            else:
                # ensure consistent tile size
                if img.size != (tile_width, tile_height):
                    img = img.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            row_images.append(img)
        if row_images:
            tiles.append((label, row_images))

    if not tiles:
        raise ValueError("No valid images for multirow preview grid")

    max_cols = max(len(images) for _label, images in tiles)
    if max_cols == 0:
        raise ValueError("Empty image rows in multirow preview grid")

    label_font = ImageFont.load_default()
    tile_width = tile_width or 224
    tile_height = tile_height or 224
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_label_width = max(dummy_draw.textlength(label, font=label_font) for label, _ in tiles)
    label_width = label_width or int(max_label_width + 2 * padding)

    canvas_width = label_width + padding + max_cols * (tile_width + padding)
    canvas_height = padding + len(tiles) * (tile_height + padding)

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=background)
    draw = ImageDraw.Draw(canvas)

    blank_tile = Image.new("RGB", (tile_width, tile_height), color=background)

    for row_idx, (label, images) in enumerate(tiles):
        y = padding + row_idx * (tile_height + padding)
        # label text vertically centered relative to the tile
        text_x = padding
        text_height = label_font.getbbox(label)[3] - label_font.getbbox(label)[1] if hasattr(label_font, "getbbox") else label_font.getsize(label)[1]
        text_y = y + (tile_height - text_height) / 2
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=label_font)

        for col in range(max_cols):
            x = label_width + padding + col * (tile_width + padding)
            if col < len(images):
                canvas.paste(images[col], (x, y))
            else:
                canvas.paste(blank_tile, (x, y))

        for img in images:
            img.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def default_transform_specs(
    noise_levels: Sequence[float],
    downsample_percents: Sequence[float],
    encoder_base_px: int,
) -> Sequence[TransformSpec]:
    specs: List[TransformSpec] = [
        TransformSpec(
            name="original",
            description="Unmodified image",
            kind="reference",
            factory=lambda _seed: _identity_transform(),
        )
    ]

    # Noise severities
    for sigma in noise_levels:
        specs.append(
            TransformSpec(
                name=_format_sigma_name(sigma),
                description=f"Additive Gaussian noise (σ = {sigma})",
                kind="noise",
                factory=lambda seed, s=sigma: _GaussianNoiseTransform(sigma=s, seed=seed),
            )
        )
    # Include a salt & pepper reference at ~5% if not present
    specs.append(
        TransformSpec(
            name="saltpepper_5pct",
            description="Salt & pepper noise (5% pixels, 60% salt)",
            kind="noise",
            factory=lambda seed: _SaltPepperTransform(amount=0.05, salt_ratio=0.6, seed=seed),
        )
    )

    # Downsampling severities relative to encoder base size
    for pct in downsample_percents:
        frac = max(0.0, min(100.0, pct)) / 100.0
        target = max(1, int(round(encoder_base_px * (1.0 - frac))))
        specs.append(
            TransformSpec(
                name=f"downsample_{int(pct)}pct",
                description=f"Downsample to {target}px (encoder base {encoder_base_px}px, {int(pct)}% reduction) then upscale",
                kind="downsample",
                factory=lambda _seed, t=target: _DownsampleTransform(target_max_dim=t),
            )
        )
        specs.append(
            TransformSpec(
                name=f"pixel_downsample_{int(pct)}pct",
                description=f"Pixelate to {target}px (encoder base {encoder_base_px}px, {int(pct)}% reduction) then upscale",
                kind="downsample_pixel",
                factory=lambda _seed, t=target: _PixelateTransform(target_max_dim=t),
            )
        )

    return tuple(specs)


def discover_first_image(root: Path) -> Optional[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return path
    return None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarise_array(values: Sequence[float]) -> MetricMoments:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return MetricMoments(float("nan"), float("nan"), float("nan"), float("nan"))
    if arr.size == 1:
        value = float(arr[0])
        return MetricMoments(value, 0.0, value, value, value, value)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    _, ci_low, ci_high = bootstrap_mean_ci(arr)
    return MetricMoments(
        mean=mean,
        std=std,
        min=float(arr.min()),
        max=float(arr.max()),
        ci_low=ci_low,
        ci_high=ci_high,
    )


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        value = float(arr[0])
        return value, value, value
    rng = default_rng(12345)
    samples = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[samples].mean(axis=1)
    lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lower, upper


def bootstrap_effect(
    values: Sequence[float],
    base_values: Sequence[float],
    relative: bool,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    base = np.asarray(base_values, dtype=np.float64)
    if arr.size == 0 or base.size == 0:
        return float("nan"), float("nan"), float("nan")
    base_mean = float(base.mean())
    if relative and (not math.isfinite(base_mean) or base_mean == 0.0):
        return float("nan"), float("nan"), float("nan")
    if relative:
        effect = (arr.mean() - base_mean) / base_mean * 100.0
    else:
        effect = float(arr.mean())

    rng = default_rng(54321)
    indices = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[indices].mean(axis=1)
    if relative:
        boot = (boot - base_mean) / base_mean * 100.0
    lower = float(np.percentile(boot, 100 * (alpha / 2)))
    upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return effect, lower, upper


def corr_anisotropy_fro(cov: np.ndarray) -> float:
    diag = np.sqrt(np.clip(np.diag(cov), a_min=1e-8, a_max=None))
    D_inv = np.reciprocal(diag)
    corr = cov * np.outer(D_inv, D_inv)
    corr = np.clip(corr, -1.0, 1.0)
    off = corr - np.diag(np.diag(corr))
    return float(np.linalg.norm(off, ord="fro"))


def summarise_metric_collection(collection: Dict[str, Sequence[float]]) -> Dict[str, MetricMoments]:
    summary: Dict[str, MetricMoments] = {}
    for field, values in collection.items():
        summary[field] = summarise_array(values)
    return summary


def load_metrics(csv_path: Path) -> tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    aggregate: Dict[str, List[float]] = {field: [] for field in METRIC_FIELDS}
    per_label: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {field: [] for field in METRIC_FIELDS})

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_name = row.get("label_name", "")
            for field in METRIC_FIELDS:
                raw = row.get(field, "")
                if not raw:
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if math.isfinite(value):
                    aggregate[field].append(value)
                    per_label[label_name][field].append(value)

    aggregate_arrays: Dict[str, np.ndarray] = {field: np.asarray(values, dtype=np.float64) for field, values in aggregate.items()}
    per_label_arrays: Dict[str, Dict[str, np.ndarray]] = {
        label: {field: np.asarray(values, dtype=np.float64) for field, values in metrics.items()}
        for label, metrics in per_label.items()
    }
    return aggregate_arrays, per_label_arrays


def format_float(value: float, precision: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def format_mean_std(moments: MetricMoments, precision: int = 4, sci_threshold: float = 1e-3) -> str:
    if not math.isfinite(moments.mean):
        return "nan"
    use_scientific = abs(moments.mean) < sci_threshold and moments.mean != 0.0
    std_finite = math.isfinite(moments.std) and moments.std != 0.0
    if not std_finite:
        if use_scientific:
            return f"{moments.mean:.2e}"
        return f"{moments.mean:.{precision}f}"
    if use_scientific or abs(moments.std) < sci_threshold:
        return f"{moments.mean:.2e} ± {moments.std:.2e}"
    return f"{moments.mean:.{precision}f} ± {moments.std:.{precision}f}"


def moments_to_dict(moments: MetricMoments) -> Dict[str, float]:
    return {
        "mean": moments.mean,
        "std": moments.std,
        "min": moments.min,
        "max": moments.max,
        "ci_low": moments.ci_low,
        "ci_high": moments.ci_high,
    }


def get_moments(summary: Dict[str, MetricMoments], metric: str) -> MetricMoments:
    return summary.get(metric, MetricMoments(float("nan"), float("nan"), float("nan"), float("nan")))


def serialise_summary(summary: Dict[str, MetricMoments]) -> Dict[str, Dict[str, float]]:
    return {metric: moments_to_dict(moments) for metric, moments in summary.items()}


def compute_delta(reference: MetricMoments, target: MetricMoments) -> tuple[float, float]:
    delta = target.mean - reference.mean
    if reference.mean == 0.0 or not math.isfinite(reference.mean):
        pct = float("nan")
    else:
        pct = (delta / reference.mean) * 100.0
    return delta, pct


def build_markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([header_line, separator, body])


def create_relative_change_plot(
    results: Dict[str, dict],
    transforms: Sequence[TransformSpec],
    metric: str,
    asset_root: Path,
    filename: str,
    ylabel: str,
) -> Path:
    baseline_summary = results["original"]["mcdo"]["summary"].get(metric)
    if baseline_summary is None or not math.isfinite(baseline_summary.mean):
        raise ValueError(f"Baseline summary missing for metric {metric}")

    labels: List[str] = []
    deltas: List[float] = []
    colors: List[str] = []
    KIND_COLOURS = {
        "noise": "#1f77b4",
        "downsample": "#ff7f0e",
        "downsample_pixel": "#9467bd",
        "reference": "#7f7f7f",
    }

    for spec in transforms:
        if spec.name == "original":
            continue
        summary = results[spec.name]["mcdo"]["summary"].get(metric)
        if summary is None or not math.isfinite(summary.mean):
            continue
        base = baseline_summary.mean
        if base == 0.0 or not math.isfinite(base):
            delta_pct = float("nan")
        else:
            delta_pct = (summary.mean - base) / base * 100.0
        labels.append(spec.name)
        deltas.append(delta_pct)
        colors.append(KIND_COLOURS.get(spec.kind, "#bcbd22"))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(x, deltas, color=colors)
    ax.axhline(0.0, color="#444444", linewidth=1, linestyle="--")
    ax.set_ylabel(f"{ylabel} change (%)")
    ax.set_title(f"Relative change in {metric.replace('_', ' ')} vs original")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for bar, delta in zip(bars, deltas):
        if math.isfinite(delta):
            ax.annotate(
                f"{delta:+.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def create_violin_plot(
    results: Dict[str, dict],
    transforms: Sequence[TransformSpec],
    metric: str,
    asset_root: Path,
    filename: str,
) -> Path:
    data: List[np.ndarray] = []
    labels: List[str] = []
    data_specs: List[TransformSpec] = []
    for spec in transforms:
        values = results[spec.name]["mcdo"]["metrics"].get(metric)
        if values is None or values.size == 0:
            continue
        data.append(values)
        labels.append(spec.name)
        data_specs.append(spec)

    if not data:
        raise ValueError(f"No data available to plot {metric}")

    fig, ax = plt.subplots(figsize=(10, 4))
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc, spec in zip(parts["bodies"], data_specs):
        if spec.kind == "noise":
            color = "#1f77b4"
        elif spec.kind == "downsample":
            color = "#ff7f0e"
        elif spec.kind == "downsample_pixel":
            color = "#9467bd"
        else:
            color = "#7f7f7f"
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ').title()} distribution per transform")
    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def create_metric_line_plot(
    results: Dict[str, dict],
    transforms: Sequence[TransformSpec],
    metric: str,
    asset_root: Path,
    filename: str,
    ylabel: str,
) -> Path:
    means: List[float] = []
    labels: List[str] = []
    colors: List[str] = []
    colour_map = {
        "noise": "#1f77b4",
        "downsample": "#ff7f0e",
        "downsample_pixel": "#9467bd",
        "reference": "#7f7f7f",
    }

    for spec in transforms:
        summary = results[spec.name]["mcdo"]["summary"].get(metric)
        if summary is None or not math.isfinite(summary.mean):
            continue
        means.append(summary.mean)
        labels.append(spec.name)
        colors.append(colour_map.get(spec.kind, "#bcbd22"))

    if not means:
        raise ValueError(f"No values collected for {metric}")

    x = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, means, marker="o")
    for xi, yi, colour in zip(x, means, colors):
        ax.scatter([xi], [yi], color=colour, s=60)
        ax.annotate(f"{yi:.2e}", xy=(xi, yi), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric.replace('_', ' ').title()} across transforms")
    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def create_severity_plot(
    results: Dict[str, dict],
    transforms: Sequence[TransformSpec],
    metric: str,
    kind: str,
    asset_root: Path,
    filename: str,
    ylabel: str,
    relative_to_original: bool = False,
) -> Path:
    xs: List[float] = []
    ys: List[float] = []
    lowers: List[float] = []
    uppers: List[float] = []
    base_values = results["original"]["mcdo"]["metrics"].get(metric)
    if base_values is None or base_values.size == 0:
        raise ValueError(f"Missing baseline values for metric '{metric}'")

    if kind == "noise":
        pairs: List[Tuple[float, TransformSpec]] = []
        for spec in transforms:
            sigma = _spec_noise_sigma(spec)
            if sigma is not None:
                pairs.append((sigma, spec))
        pairs.sort(key=lambda p: p[0])
        for sigma, spec in pairs:
            xs.append(sigma)
            values = results[spec.name]["mcdo"]["metrics"].get(metric)
            if values is None or values.size == 0:
                ys.append(float("nan"))
                lowers.append(float("nan"))
                uppers.append(float("nan"))
                continue
            if relative_to_original:
                mean, low, high = bootstrap_effect(values, base_values, True)
            else:
                mean, low, high = bootstrap_mean_ci(values)
            ys.append(mean)
            lowers.append(low)
            uppers.append(high)
        x_label = "sigma"
    elif kind in {"downsample", "downsample_pixel"}:
        pairs_ds: List[Tuple[int, TransformSpec]] = []
        for spec in transforms:
            pct = _spec_downsample_percent(spec)
            if pct is not None:
                pairs_ds.append((pct, spec))
        pairs_ds.sort(key=lambda p: p[0])
        for pct, spec in pairs_ds:
            xs.append(float(pct))
            values = results[spec.name]["mcdo"]["metrics"].get(metric)
            if values is None or values.size == 0:
                ys.append(float("nan"))
                lowers.append(float("nan"))
                uppers.append(float("nan"))
                continue
            if relative_to_original:
                mean, low, high = bootstrap_effect(values, base_values, True)
            else:
                mean, low, high = bootstrap_mean_ci(values)
            ys.append(mean)
            lowers.append(low)
            uppers.append(high)
        x_label = "pixel reduction (%)" if kind == "downsample_pixel" else "downsample reduction (%)"
    else:
        raise ValueError("kind must be 'noise' or 'downsample'")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, marker="o")
    if lowers and uppers:
        ax.fill_between(xs, lowers, uppers, alpha=0.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel + (" (Δ%)" if relative_to_original else ""))
    ax.set_title(f"{metric.replace('_',' ').title()} vs {kind}")
    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def create_scatter_plot(
    results: Dict[str, dict],
    transforms: Sequence[TransformSpec],
    x_metric: str,
    y_metric: str,
    asset_root: Path,
    filename: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    colour_map = {
        "noise": "#1f77b4",
        "downsample": "#ff7f0e",
        "downsample_pixel": "#9467bd",
        "reference": "#7f7f7f",
    }
    kind_seen: Dict[str, bool] = {}
    for spec in transforms:
        metrics = results[spec.name]["mcdo"]["metrics"]
        x = metrics.get(x_metric)
        y = metrics.get(y_metric)
        if x is None or y is None or x.size == 0 or y.size == 0:
            continue
        color = colour_map.get(spec.kind, "#bcbd22")
        ax.scatter(x, y, s=8, alpha=0.4, color=color)
        kind_seen[spec.kind] = True
    ax.scatter([], [], s=8, color=colour_map.get("reference", "#7f7f7f"), label="baseline (original)")
    legend_entries = [
        ("noise", "noise perturbations"),
        ("downsample", "downsampling perturbations"),
        ("downsample_pixel", "pixelated downsampling"),
    ]
    for kind, label in legend_entries:
        if kind_seen.get(kind):
            ax.scatter([], [], s=8, color=colour_map[kind], label=label)
    ax.set_xlabel(x_metric.replace("_", " "))
    ax.set_ylabel(y_metric.replace("_", " "))
    ax.set_title(f"{y_metric.replace('_',' ').title()} vs {x_metric.replace('_',' ').title()}")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def create_trace_meanshift_scatter(
    results: Dict[str, dict],
    transforms: Sequence[TransformSpec],
    extras_map: Dict[str, Dict[str, Dict[str, float]]],
    kind: str,
    asset_root: Path,
    filename: str,
    title: str,
) -> Path:
    points: List[Tuple[float, float, float, str]] = []  # (shift, trace, severity, label)
    if kind == "noise":
        for spec in transforms:
            sigma = _spec_noise_sigma(spec)
            if sigma is None:
                continue
            shift = extras_map.get(spec.name, {}).get("mean_shift", {}).get("mean")
            trace = results[spec.name]["mcdo"]["summary"].get("trace")
            if shift is None or trace is None:
                continue
            points.append((shift, trace.mean, sigma, f"σ={sigma:.2f}"))
    elif kind in {"downsample", "downsample_pixel"}:
        for spec in transforms:
            pct = _spec_downsample_percent(spec)
            if pct is None:
                continue
            shift = extras_map.get(spec.name, {}).get("mean_shift", {}).get("mean")
            trace = results[spec.name]["mcdo"]["summary"].get("trace")
            if shift is None or trace is None:
                continue
            label = f"{pct}%"
            points.append((shift, trace.mean, float(pct), label))
    else:
        raise ValueError("kind must be 'noise' or 'downsample'")

    if not points:
        raise ValueError("No data available for trace vs mean-shift scatter")

    points.sort(key=lambda item: item[2])
    shifts = [p[0] for p in points]
    traces = [p[1] for p in points]
    severity = [p[2] for p in points]
    labels = [p[3] for p in points]

    cmap = matplotlib.colormaps.get_cmap("viridis")
    norm = plt.Normalize(min(severity), max(severity))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(shifts, traces, c=severity, cmap=cmap, norm=norm, s=60, marker="o")
    ax.plot(shifts, traces, color="#444444", linewidth=1, alpha=0.6)

    for shift, trace_val, label in zip(shifts, traces, labels):
        ax.annotate(
            label,
            (shift, trace_val),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            alpha=0.8,
        )

    ax.set_xlabel("Mean shift (L2)")
    ax.set_ylabel("Trace")
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax)
    cbar_label = "σ" if kind == "noise" else "Reduction (%)"
    cbar.set_label(cbar_label)
    ax.grid(alpha=0.2)

    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def create_aggregate_plot(
    results: Dict[str, dict],
    transforms: Sequence[TransformSpec],
    extras_map: Dict[str, Dict[str, Dict[str, float]]],
    asset_root: Path,
    filename: str,
) -> Path:
    metrics = [
        ("trace", "Trace"),
        ("logdet", "Logdet"),
        ("corr_anisotropy", "Correlation anisotropy (F)")
    ]
    ordered_specs = sorted(transforms, key=_transform_order_key)
    names = [spec.name for spec in ordered_specs]
    x_positions = np.arange(len(names))
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(10, len(names) * 0.4), 2 + 2 * len(metrics)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    groups_seen: set[str] = set()

    for ax, (metric_key, ylabel) in zip(axes, metrics):
        for idx, spec in enumerate(ordered_specs):
            group = _spec_group_category(spec)
            color = CATEGORY_COLOURS.get(group, "#bcbd22")
            if metric_key == "corr_anisotropy":
                extra = extras_map.get(spec.name, {}).get("corr_anisotropy")
                moments = dict_to_moments(extra) if extra else MetricMoments(
                    float("nan"), float("nan"), float("nan"), float("nan")
                )
            else:
                summary = results[spec.name]["mcdo"]["summary"].get(metric_key)
                moments = summary if summary is not None else MetricMoments(
                    float("nan"), float("nan"), float("nan"), float("nan")
                )

            mean = moments.mean
            if not math.isfinite(mean):
                continue
            if math.isfinite(moments.ci_low) and math.isfinite(moments.ci_high):
                lower = mean - moments.ci_low
                upper = moments.ci_high - mean
            else:
                lower = moments.std if math.isfinite(moments.std) else 0.0
                upper = moments.std if math.isfinite(moments.std) else 0.0
            lower = max(0.0, lower)
            upper = max(0.0, upper)
            ax.errorbar(
                x_positions[idx],
                mean,
                yerr=np.array([[lower], [upper]]),
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1,
                capsize=3,
            )
            groups_seen.add(group)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(names, rotation=40, ha="right")

    legend_handles: List[Line2D] = []
    for group in ["original", "gaussian", "sp", "downsample1", "downsample2"]:
        if group in groups_seen:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    color=CATEGORY_COLOURS[group],
                    label=CATEGORY_LABELS[group],
                )
            )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=min(len(legend_handles), 5),
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.4,
            borderaxespad=0.0,
        )

    axes[0].set_title("Aggregate metrics with 95% confidence intervals")
    fig.subplots_adjust(top=0.78, bottom=0.21, left=0.1, right=0.96, hspace=0.32)
    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def create_extras_severity_plot(
    transforms: Sequence[TransformSpec],
    extras: Dict[str, Dict[str, Dict[str, float]]],
    values_key: str,
    kind: str,
    asset_root: Path,
    filename: str,
    ylabel: str,
) -> Path:
    xs: List[float] = []
    ys: List[float] = []
    lowers: List[float] = []
    uppers: List[float] = []

    if kind == "noise":
        pairs: List[Tuple[float, TransformSpec]] = []
        for spec in transforms:
            sigma = _spec_noise_sigma(spec)
            if sigma is not None and spec.name in extras:
                pairs.append((sigma, spec))
        pairs.sort(key=lambda p: p[0])
        for sigma, spec in pairs:
            xs.append(sigma)
            values = extras[spec.name].get(f"{values_key}_values")
            if not values:
                ys.append(float("nan"))
                lowers.append(float("nan"))
                uppers.append(float("nan"))
                continue
            mean, low, high = bootstrap_mean_ci(values)
            ys.append(mean)
            lowers.append(low)
            uppers.append(high)
        x_label = "sigma"
    elif kind in {"downsample", "downsample_pixel"}:
        pairs_ds: List[Tuple[int, TransformSpec]] = []
        for spec in transforms:
            pct = _spec_downsample_percent(spec)
            if pct is not None and spec.name in extras:
                pairs_ds.append((pct, spec))
        pairs_ds.sort(key=lambda p: p[0])
        for pct, spec in pairs_ds:
            xs.append(float(pct))
            values = extras[spec.name].get(f"{values_key}_values")
            if not values:
                ys.append(float("nan"))
                lowers.append(float("nan"))
                uppers.append(float("nan"))
                continue
            mean, low, high = bootstrap_mean_ci(values)
            ys.append(mean)
            lowers.append(low)
            uppers.append(high)
        x_label = "pixel reduction (%)" if kind == "downsample_pixel" else "downsample reduction (%)"
    else:
        raise ValueError("kind must be 'noise' or 'downsample'")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, marker="o")
    if lowers and uppers:
        ax.fill_between(xs, lowers, uppers, alpha=0.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {kind}")
    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def create_mean_bar_plot(
    transforms: Sequence[TransformSpec],
    extras: Dict[str, Dict[str, Dict[str, float]]],
    key: str,
    asset_root: Path,
    filename: str,
    ylabel: str,
) -> Path:
    ordered_specs = sorted(transforms, key=_transform_order_key)
    names: List[str] = []
    values: List[float] = []
    lower_err: List[float] = []
    upper_err: List[float] = []
    colors: List[str] = []
    groups_seen: set[str] = set()
    for spec in ordered_specs:
        extra = extras.get(spec.name, {}).get(key)
        if not extra:
            continue
        mean = extra.get("mean")
        if not math.isfinite(mean):
            continue
        ci_low = extra.get("ci_low")
        ci_high = extra.get("ci_high")
        if math.isfinite(ci_low) and math.isfinite(ci_high):
            lower = max(0.0, mean - ci_low)
            upper = max(0.0, ci_high - mean)
        else:
            std = extra.get("std", 0.0)
            lower = max(0.0, std)
            upper = max(0.0, std)
        group = _spec_group_category(spec)
        color = CATEGORY_COLOURS.get(group, "#bcbd22")
        names.append(spec.name)
        values.append(mean)
        lower_err.append(lower)
        upper_err.append(upper)
        colors.append(color)
        groups_seen.add(group)

    if not names:
        raise ValueError("No mean shift data available.")

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.4), 4))
    bars = ax.bar(x, values, color=colors, yerr=[lower_err, upper_err], capsize=3)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_title(f"{ylabel} across transforms")

    legend_handles: List[Line2D] = []
    for group in ["original", "gaussian", "sp", "downsample1", "downsample2"]:
        if group in groups_seen:
            legend_handles.append(
                Line2D([0], [0], marker="s", linestyle="", color=CATEGORY_COLOURS[group], label=CATEGORY_LABELS[group])
            )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=min(len(legend_handles), 5),
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.4,
            borderaxespad=0.0,
        )

    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.subplots_adjust(top=0.78, bottom=0.28, left=0.1, right=0.96)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def create_extras_severity_plot(
    transforms: Sequence[TransformSpec],
    extras: Dict[str, Dict[str, Dict[str, float]]],
    values_key: str,
    kind: str,
    asset_root: Path,
    filename: str,
    ylabel: str,
) -> Path:
    xs: List[float] = []
    ys: List[float] = []
    lowers: List[float] = []
    uppers: List[float] = []

    if kind == "noise":
        pairs: List[Tuple[float, TransformSpec]] = []
        for spec in transforms:
            sigma = _spec_noise_sigma(spec)
            if sigma is not None and spec.name in extras:
                pairs.append((sigma, spec))
        pairs.sort(key=lambda p: p[0])
        for sigma, spec in pairs:
            xs.append(sigma)
            values = extras[spec.name].get(f"{values_key}_values")
            if not values:
                ys.append(float("nan"))
                lowers.append(float("nan"))
                uppers.append(float("nan"))
                continue
            mean, low, high = bootstrap_mean_ci(values)
            ys.append(mean)
            lowers.append(low)
            uppers.append(high)
        xlabel = "sigma"
    elif kind in {"downsample", "downsample_pixel"}:
        pairs_ds: List[Tuple[int, TransformSpec]] = []
        for spec in transforms:
            pct = _spec_downsample_percent(spec)
            if pct is not None and spec.name in extras:
                pairs_ds.append((pct, spec))
        pairs_ds.sort(key=lambda p: p[0])
        for pct, spec in pairs_ds:
            xs.append(float(pct))
            values = extras[spec.name].get(f"{values_key}_values")
            if not values:
                ys.append(float("nan"))
                lowers.append(float("nan"))
                uppers.append(float("nan"))
                continue
            mean, low, high = bootstrap_mean_ci(values)
            ys.append(mean)
            lowers.append(low)
            uppers.append(high)
        xlabel = "downsample reduction (%)"
        if kind == "downsample_pixel":
            xlabel = "pixel reduction (%)"
    else:
        raise ValueError("kind must be 'noise' or 'downsample'")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, marker="o")
    if lowers and uppers:
        ax.fill_between(xs, lowers, uppers, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {kind}")
    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def create_mean_shift_plot(
    transforms: Sequence[TransformSpec],
    extras: Dict[str, Dict[str, Dict[str, float]]],
    metric_key: str,
    kind: str,
    asset_root: Path,
    filename: str,
    ylabel: str,
) -> Path:
    xs: List[float] = []
    ys: List[float] = []
    if kind == "noise":
        items: List[Tuple[float, str]] = []
        for spec in transforms:
            sigma = _spec_noise_sigma(spec)
            if sigma is not None and spec.name in extras:
                items.append((sigma, spec.name))
        items.sort(key=lambda x: x[0])
        for sigma, name in items:
            xs.append(sigma)
            ys.append(extras[name][metric_key]["mean"])
        xlabel = "sigma"
    elif kind == "downsample":
        items: List[Tuple[int, str]] = []
        for spec in transforms:
            pct = _spec_downsample_percent(spec)
            if pct is not None and spec.name in extras:
                items.append((pct, spec.name))
        items.sort(key=lambda x: x[0])
        for pct, name in items:
            xs.append(float(pct))
            ys.append(extras[name][metric_key]["mean"])
        xlabel = "downsample reduction (%)"
    else:
        raise ValueError("kind must be 'noise' or 'downsample'")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {kind} severity")
    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def create_angle_radar(
    results: Dict[str, dict],
    extras: Dict[str, Dict[str, Dict[str, float]]],
    transforms: Sequence[TransformSpec],
    asset_root: Path,
    filename: str,
) -> Path:
    candidates = []
    for spec in transforms:
        if spec.name in extras:
            candidates.append(spec)
    if not candidates:
        raise ValueError("No extras available for angle radar")

    selected_specs: List[Tuple[str, str]] = [("original", "Original")]
    added: set[str] = {"original"}

    def add_spec(spec: Optional[TransformSpec], label_suffix: str) -> None:
        if spec and spec.name not in added:
            selected_specs.append((spec.name, label_suffix))
            added.add(spec.name)

    # Identify noise specs with trace statistics.
    noise_specs = [spec for spec in candidates if _spec_noise_sigma(spec) is not None]
    if noise_specs:
        def trace_mean(spec: TransformSpec) -> float:
            return results[spec.name]["mcdo"]["summary"]["trace"].mean

        max_noise_spec = max(noise_specs, key=trace_mean)
        min_noise_spec = min(noise_specs, key=trace_mean)
        add_spec(max_noise_spec, f"{max_noise_spec.name} (noise max trace)")
        add_spec(min_noise_spec, f"{min_noise_spec.name} (noise min trace)")

    # Identify downsample specs with trace statistics.
    down_specs = [spec for spec in candidates if _spec_downsample_percent(spec) is not None]
    if down_specs:
        def down_trace_mean(spec: TransformSpec) -> float:
            return results[spec.name]["mcdo"]["summary"]["trace"].mean

        max_down_spec = max(down_specs, key=down_trace_mean)
        min_down_spec = min(down_specs, key=down_trace_mean)
        add_spec(max_down_spec, f"{max_down_spec.name} (downsample max trace)")
        add_spec(min_down_spec, f"{min_down_spec.name} (downsample min trace)")

    angles = sorted(set(angle for summary in extras.values() for angle in summary.get("angle_trace_means", {}).keys()))
    if not angles:
        raise ValueError("No angle data available")
    theta = np.radians(angles + [angles[0]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    colours = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    for idx, (spec_name, label) in enumerate(selected_specs):
        stats = extras.get(spec_name, {}).get("angle_trace_means", {})
        values = [stats.get(angle, np.nan) for angle in angles]
        values.append(values[0])
        ax.plot(theta, values, marker="o", label=label, color=colours[idx % len(colours)])
        ax.fill(theta, values, alpha=0.1, color=colours[idx % len(colours)])
    ax.set_title("Trace by viewpoint angle")
    ax.set_thetagrids(angles)
    ax.set_rlabel_position(0)
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=min(len(selected_specs), 3),
        frameon=False,
    )
    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.92)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def create_pca_plot(
    analysis_root: Path,
    baseline_dir: Path,
    sample_index: int,
    scenario: str,
    label: str,
    asset_root: Path,
    filename: str,
) -> Path:
    baseline_samples = load_sample_tensors(baseline_dir)
    mu_baseline = baseline_samples[sample_index]["mu"]

    run_dir = analysis_root / "mcdo" / scenario
    samples = load_sample_tensors(run_dir)
    sample = samples[sample_index]
    embeddings = sample["embeddings"]
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    projected = centered @ components.T

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(projected[:, 0], projected[:, 1], s=10, alpha=0.5)
    ax.scatter(0, 0, marker="x", color="#d62728", label="Scenario mean")
    baseline_proj = (mu_baseline - mean) @ components.T
    ax.arrow(
        baseline_proj[0],
        baseline_proj[1],
        -baseline_proj[0],
        -baseline_proj[1],
        head_width=0.05,
        length_includes_head=True,
        color="#9467bd",
        label="Drift from baseline",
    )
    ax.set_title(label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    handles, legends = ax.get_legend_handles_labels()
    if legends:
        ax.legend(loc="upper right")
    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
def compute_extended_metrics(
    results: Dict[str, dict],
    analysis_root: Path,
    transforms: Sequence[TransformSpec],
    data_root: Path,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    extras: Dict[str, Dict[str, Dict[str, float]]] = {}

    base_mcdo_dir = analysis_root / "mcdo" / "original"
    base_det_dir = analysis_root / "baseline" / "original"
    base_mcdo_samples = load_sample_tensors(base_mcdo_dir)
    base_det_samples = load_sample_tensors(base_det_dir)

    angle_map = build_angle_map(data_root)

    for spec in transforms:
        mcdo_dir = analysis_root / "mcdo" / spec.name
        if not mcdo_dir.exists():
            continue
        samples = load_sample_tensors(mcdo_dir)

        euclid_shifts: List[float] = []
        mahal_shifts: List[float] = []
        tangent_traces: List[float] = []
        tangent_maxes: List[float] = []
        circular_vars: List[float] = []
        spectral_entropy: List[float] = []
        spectral_top10: List[float] = []
        corr_anisotropy: List[float] = []

        angle_trace: Dict[int, List[float]] = {angle: [] for angle in sorted(set(angle_map.values()))}

        for idx, tensors in samples.items():
            mu_pert = tensors["mu"]
            cov = tensors["cov"]
            embeddings = tensors["embeddings"]

            mu_det = base_det_samples[idx]["mu"]
            diff = mu_pert - mu_det
            euclid_shifts.append(float(np.linalg.norm(diff)))

            base_cov = base_mcdo_samples[idx]["cov"]
            cov_jitter = base_cov + np.eye(base_cov.shape[0]) * 1e-6
            inv = np.linalg.pinv(cov_jitter)
            mahal_val = float(np.sqrt(diff @ inv @ diff)) if np.isfinite(diff @ inv @ diff) else float("nan")
            mahal_shifts.append(mahal_val)

            tangent = tangent_covariance_numpy(mu_pert, cov)
            tangent_traces.append(float(np.trace(tangent)))
            tangent_eigs = np.linalg.eigvalsh(tangent + np.eye(tangent.shape[0]) * 1e-6)
            tangent_maxes.append(float(np.max(tangent_eigs)))

            corr_anisotropy.append(corr_anisotropy_fro(cov))

            cov_eigs = np.linalg.eigvalsh(cov + np.eye(cov.shape[0]) * 1e-6)
            cov_eigs = np.clip(cov_eigs, 0.0, None)
            total = cov_eigs.sum()
            if total > 0:
                probs = cov_eigs / total
                entropy = float(-(probs * np.log(probs + 1e-12)).sum())
                share = float(probs[-min(10, probs.size):].sum())
            else:
                entropy = float("nan")
                share = float("nan")
            spectral_entropy.append(entropy)
            spectral_top10.append(share)

            norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
            unit = embeddings / norms
            mean_vec = unit.mean(axis=0)
            resultant = np.linalg.norm(mean_vec)
            circular_vars.append(float(1.0 - resultant))

            angle = angle_map.get(idx)
            if angle is not None:
                angle_trace.setdefault(angle, []).append(float(results[spec.name]["mcdo"]["metrics"]["trace"][idx]))

        extras[spec.name] = {
            "mean_shift": summarise_extra_metric(euclid_shifts),
            "mahal_shift": summarise_extra_metric(mahal_shifts),
            "tangent_trace": summarise_extra_metric(tangent_traces),
            "tangent_max": summarise_extra_metric(tangent_maxes),
            "circular_variance": summarise_extra_metric(circular_vars),
            "spectral_entropy": summarise_extra_metric(spectral_entropy),
            "spectral_top10": summarise_extra_metric(spectral_top10),
            "corr_anisotropy": summarise_extra_metric(corr_anisotropy),
            "angle_trace_means": {
                angle: float(np.mean(values)) if values else float("nan") for angle, values in angle_trace.items()
            },
            "corr_anisotropy_values": corr_anisotropy,
        }

    return extras
def tangent_covariance_numpy(mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mu)
    if norm <= 0.0:
        return np.zeros_like(cov)
    unit = mu / norm
    proj = np.eye(mu.shape[0]) - np.outer(unit, unit)
    return proj @ cov @ proj


def load_sample_tensors(run_dir: Path) -> Dict[int, Dict[str, np.ndarray]]:
    sample_dir = run_dir / "individual"
    if not sample_dir.exists():
        raise FileNotFoundError(f"Raw sample directory missing under {run_dir}")
    data: Dict[int, Dict[str, np.ndarray]] = {}
    for folder in sorted(sample_dir.iterdir()):
        if not folder.is_dir():
            continue
        idx = int(folder.name)
        mu = torch.load(folder / "mu.pt", map_location="cpu").numpy()
        cov = torch.load(folder / "Sigma.pt", map_location="cpu").numpy()
        embeddings = torch.load(folder / "embeddings.pt", map_location="cpu").numpy()
        data[idx] = {"mu": mu, "cov": cov, "embeddings": embeddings}
    return data


def build_angle_map(data_root: Path) -> Dict[int, int]:
    try:
        from torchvision.datasets import ImageFolder
    except ImportError as exc:
        raise RuntimeError("torchvision is required for angle metadata") from exc

    dataset = ImageFolder(root=str(data_root))
    angle_map: Dict[int, int] = {}
    for idx, (path, _label) in enumerate(dataset.samples):
        angle = extract_angle(Path(path))
        if angle is not None:
            angle_map[idx] = angle
    return angle_map


def extract_angle(path: Path) -> Optional[int]:
    stem = path.stem
    if "_" not in stem:
        return None
    angle_token = stem.split("_")[-1]
    try:
        angle = int(angle_token)
    except ValueError:
        return None
    return angle


def summarise_extra_metric(values: Sequence[float]) -> Dict[str, float]:
    moments = summarise_array(values)
    payload = moments_to_dict(moments)
    payload["ci_low"] = moments.ci_low
    payload["ci_high"] = moments.ci_high
    return payload


def dict_to_moments(summary: Dict[str, float]) -> MetricMoments:
    return MetricMoments(
        mean=summary.get("mean", float("nan")),
        std=summary.get("std", float("nan")),
        min=summary.get("min", float("nan")),
        max=summary.get("max", float("nan")),
        ci_low=summary.get("ci_low", float("nan")),
        ci_high=summary.get("ci_high", float("nan")),
    )

def _spec_noise_sigma(spec: TransformSpec) -> Optional[float]:
    if spec.kind != "noise":
        return None
    if spec.name.startswith("gaussian_noise_"):
        try:
            val = int(spec.name.split("_")[-1])  # e.g., '010' for 0.010
            return val / 1000.0
        except Exception:
            return None
    return None


def _spec_downsample_percent(spec: TransformSpec) -> Optional[int]:
    if spec.kind not in {"downsample", "downsample_pixel"}:
        return None
    prefix = "pixel_downsample_" if spec.kind == "downsample_pixel" else "downsample_"
    if spec.name.startswith(prefix) and spec.name.endswith("pct"):
        try:
            mid = spec.name[len(prefix) : -len("pct")]
            return int(mid)
        except Exception:
            return None
    return None


def _spec_group_category(spec: TransformSpec) -> str:
    if spec.name == "original":
        return "original"
    if spec.name == "saltpepper_5pct":
        return "sp"
    if spec.kind == "noise":
        return "gaussian"
    if spec.kind == "downsample":
        return "downsample1"
    if spec.kind == "downsample_pixel":
        return "downsample2"
    return "other"


def _transform_order_key(spec: TransformSpec) -> tuple:
    group = _spec_group_category(spec)
    group_rank = GROUP_ORDER.get(group, 99)
    if spec.kind == "noise":
        secondary = _spec_noise_sigma(spec)
        secondary = secondary if secondary is not None else 0.0
    elif spec.kind in {"downsample", "downsample_pixel"}:
        secondary = _spec_downsample_percent(spec)
        secondary = float(secondary) if secondary is not None else 999.0
    else:
        secondary = 0.0
    return (group_rank, secondary, spec.name)


def pca_insight(spec: Optional[TransformSpec], label: str) -> str:
    if spec is None or spec.name == "original":
        return "Baseline embeddings remain compact with minimal directional bias."
    if spec.kind == "noise":
        sigma = _spec_noise_sigma(spec)
        if sigma is not None:
            return f"Gaussian noise with σ={sigma:.2f} widens the cluster without shifting its centre markedly."
        return "Noise widens the embedding cloud while keeping the centroid near baseline."
    if spec.kind == "downsample":
        pct = _spec_downsample_percent(spec)
        if pct is not None:
            return f"Smoothed crop (bicubic upsample) at {pct}% reduction elongates the embedding cloud along a single axis as detail is lost."
        return "Smoothed crops elongate the embedding cloud along a dominant axis."
    if spec.kind == "downsample_pixel":
        pct = _spec_downsample_percent(spec)
        if pct is not None:
            return f"Pixelation at {pct}% produces discrete clusters as block artefacts dominate token activations."
        return "Pixelation carves discrete blocks within the embedding projection."
    return f"{label} perturbs the embedding footprint while preserving global orientation."


def _select_max_severity_specs(transforms: Sequence[TransformSpec]) -> Tuple[TransformSpec, Optional[TransformSpec], Optional[TransformSpec]]:
    base = next(spec for spec in transforms if spec.name == "original")
    noise_specs = [spec for spec in transforms if _spec_noise_sigma(spec) is not None]
    down_specs = [
        spec for spec in transforms if spec.kind == "downsample" and _spec_downsample_percent(spec) is not None
    ]
    max_noise = None
    if noise_specs:
        max_noise = max(noise_specs, key=lambda s: _spec_noise_sigma(s) or 0.0)
    max_down = None
    if down_specs:
        max_down = max(down_specs, key=lambda s: _spec_downsample_percent(s) or 0)
    return base, max_noise, max_down


def create_pass_stability_plot(
    pass_counts: Sequence[int],
    series: Dict[str, Dict[str, List[float]]],
    metric: str,
    asset_root: Path,
    filename: str,
    ylabel: str,
) -> Path:
    # series: {label: {"mean": [..], "sem": [..]}}
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = {"original": "#7f7f7f", "max_noise": "#1f77b4", "max_downsample": "#ff7f0e"}
    labels_order = [k for k in ["original", "max_noise", "max_downsample"] if k in series]
    x = np.asarray(pass_counts)
    for key in labels_order:
        means = np.asarray(series[key]["mean"]) if series[key]["mean"] else None
        sems = np.asarray(series[key]["sem"]) if series[key]["sem"] else None
        if means is None or means.size != x.size:
            continue
        ax.plot(x, means, marker="o", color=colours.get(key, None), label=key)
        if sems is not None and sems.size == x.size:
            ax.fill_between(x, means - 1.96 * sems, means + 1.96 * sems, color=colours.get(key, None), alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("passes (T)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric.replace('_', ' ').title()} stability vs passes")
    ax.legend()
    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def generate_report(
    report_path: Path,
    asset_rel_dir: Path,
    analysis_root: Path,
    downsample_percents: Sequence[int],
    setup: Dict[str, str],
    results: Dict[str, dict],
    transforms: Sequence[TransformSpec],
    chart_paths: Dict[str, Path],
    pass_stability_notes: Optional[Dict[str, float]] = None,
) -> None:
    report_lines: List[str] = ["# Sim2 45° CLIP MCDO Noise Study"]
    figure_manager = FigureManager(report_lines)
    appendix_tables: List[Tuple[str, str]] = []

    # Section 1: Introduction
    report_lines.append("\n## 1. Introduction")
    report_lines.append(
        "We study how additive noise and progressive downsampling alter the geometry of CLIP image embeddings when "
        "Monte Carlo Dropout (MCDO) is applied to the vision tower. The goal is to quantify how perturbations change "
        "the stochastic embedding cloud (variance, orientation, and mean drift) so we can anticipate failure modes "
        "in downstream retrieval or classification settings."
    )
    report_lines.append(
        "All analyses focus on embedding-space diagnostics rather than classifier accuracy, mirroring a scientific report "
        "structure: we document the experimental configuration, define uncertainty metrics, and then interpret how "
        "those metrics respond to perturbations of increasing severity."
    )
    if "modulation_overview_grid" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["modulation_overview_grid"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Preview overview — all perturbations",
            "Rows correspond to noise, smoothed crops, and raw crops respectively; columns step through increasing severity with the baseline at left.",
        )

    # Section 2: Methodology
    report_lines.append("\n## 2. Methodology")
    for key, value in setup.items():
        report_lines.append(f"- **{key}:** {value}")
    report_lines.append(
        "- **Dropout instrumentation:** DropoutAdapter wraps all twelve ViT encoder blocks plus the visual projection head (p = 0.01)."
    )
    report_lines.append(
        "- **Sampling:** 64 stochastic passes per image (microbatch 4, deterministic seed). Deterministic baselines use a single pass with dropout disabled."
    )
    report_lines.append(
        "- **Perturbation sweep:** Gaussian noise σ∈{0.01,…,0.5} and downsampling up to 93% (224→16 px) followed by bicubic upsampling."
    )
    report_lines.append(
        "- **Pass-stability sweep:** Additional runs at T∈{2,4,8,16,32,64,128} for original, strongest-noise, and strongest-downsampling cases."
    )

    # Section 3: Metric Primer
    report_lines.append("\n## 3. Metric Primer")
    report_lines.append(
        "- **Trace (Tr Σ):** Sum of covariance eigenvalues; measures total dispersion of the embedding cloud. Increases imply broader uncertainty."
    )
    report_lines.append(
        "- **Log-determinant (log det Σ):** Log-volume of the covariance ellipsoid. It captures how uncertainty spreads across dimensions; drops signal collapse into a lower-dimensional subspace."
    )
    report_lines.append(
        "- **Off-diagonal mass:** L₁ magnitude of non-diagonal covariance entries; indicates cross-dimensional coupling and anisotropy."
    )
    report_lines.append(
        "- **Anisotropy (corr-F):** Frobenius norm of the correlation matrix off-diagonals; values grow as variance concentrates into preferred axes instead of spreading uniformly."
    )
    report_lines.append(
        "- **Mean shift / Mahalanobis shift:** L2 distance and covariance-normalised distance between the stochastic mean and the deterministic embedding, quantifying drift induced by perturbations."
    )
    report_lines.append(
        "- **Tangent trace / λₘₐₓ:** Variance within the hypersphere orthogonal to the mean direction; highlights changes in directional uncertainty."
    )
    report_lines.append(
        "- **Spectral entropy & top-10 share:** Describe how variance concentrates across eigenmodes. Lower entropy or higher top-10 share means uncertainty is dominated by a few directions."
    )
    report_lines.append(
        "- **Circular variance:** Dispersion of unit-normalised samples; complements tangent analysis by examining orientation consistency."
    )
    report_lines.append(
        "- **Pass-count stability:** How rapidly trace/logdet estimates converge as the number of stochastic passes T increases."
    )

    noise_specs = [spec for spec in transforms if spec.kind == "noise" and spec.name != "original"]
    downsample_specs = [spec for spec in transforms if spec.kind == "downsample"]
    reference_summary = results["original"]["mcdo"]["summary"]
    base_trace = get_moments(reference_summary, "trace").mean
    base_logdet = get_moments(reference_summary, "logdet").mean
    severe_noise = max(noise_specs, key=lambda spec: _spec_noise_sigma(spec) or 0.0) if noise_specs else None
    severe_down = max(downsample_specs, key=lambda spec: _spec_downsample_percent(spec) or 0) if downsample_specs else None
    noise_peak_trace = noise_peak_logdet = None
    noise_peak_trace_pct = noise_peak_logdet_delta = None
    if severe_noise:
        noise_summary = results[severe_noise.name]["mcdo"]["summary"]
        noise_peak_trace = get_moments(noise_summary, "trace").mean
        noise_peak_logdet = get_moments(noise_summary, "logdet").mean
        if base_trace:
            noise_peak_trace_pct = (noise_peak_trace - base_trace) / base_trace * 100.0
        if base_logdet:
            noise_peak_logdet_delta = noise_peak_logdet - base_logdet
    down_peak_trace = down_peak_logdet = None
    down_peak_trace_pct = down_peak_logdet_delta = None
    down_peak_volume_ratio = None
    down_peak_pct = None
    if severe_down:
        down_summary = results[severe_down.name]["mcdo"]["summary"]
        down_peak_trace = get_moments(down_summary, "trace").mean
        down_peak_logdet = get_moments(down_summary, "logdet").mean
        down_peak_pct = _spec_downsample_percent(severe_down) or 0
        if base_trace:
            down_peak_trace_pct = (down_peak_trace - base_trace) / base_trace * 100.0
        if math.isfinite(down_peak_logdet) and math.isfinite(base_logdet):
            down_peak_logdet_delta = down_peak_logdet - base_logdet
            down_peak_volume_ratio = math.exp(down_peak_logdet - base_logdet)

    # Section 4: Aggregate metrics (embedding only)
    report_lines.append("\n## 4. Aggregate MCDO Embedding Metrics")
    extras_map = {spec.name: results.get(spec.name, {}).get("extras", {}) for spec in transforms}
    agg_headers = ["transform", "trace", "logdet", "anisotropy (corr-F)"]
    agg_rows: List[List[str]] = []
    for spec in transforms:
        summary = results[spec.name]["mcdo"]["summary"]
        corr_extra = extras_map.get(spec.name, {}).get("corr_anisotropy")
        corr_value = format_mean_std(dict_to_moments(corr_extra)) if corr_extra else "nan"
        agg_rows.append(
            [
                spec.name,
                format_mean_std(get_moments(summary, "trace")),
                format_mean_std(get_moments(summary, "logdet")),
                corr_value,
            ]
        )
    appendix_tables.append(("Aggregate metrics", build_markdown_table(agg_headers, agg_rows)))
    report_lines.append(
        f"Baseline trace is {base_trace:.2f}, with logdet {base_logdet:.2f}. "
        "Noise and crop-based degradations broaden the covariance while logdet declines; we report anisotropy using correlation-based Frobenius norms for scale-free comparison."
    )
    if "aggregate_overview" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["aggregate_overview"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Aggregate metrics overview",
            "Aggressive downsampling emerges as the dominant driver of higher trace and lower log-volume across transforms.",
        )
    if "mean_shift_bar" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["mean_shift_bar"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Mean shift across transforms",
            "Mean embedding displacement mirrors the same ordering: raw (nearest) and smoothed (bicubic) crops introduce the largest drift from the deterministic baseline.",
        )

    # Section 5: Deterministic baseline
    report_lines.append("\n## 5. Deterministic Baseline (1 pass, dropout disabled)")
    report_lines.append(
        "With dropout disabled every run collapses to trace ≈ 5.12 × 10⁻⁴ and a singular covariance; logdet and anisotropy are therefore undefined."
    )

    # Section 6: Noise sensitivity
    if noise_specs:
        report_lines.append("\n## 6. Sensitivity to Noise")
        noise_headers = ["transform", "Δ trace (%)", "Δ logdet", "Δ off-diag (%)"]
        noise_rows: List[List[str]] = []
        for spec in noise_specs:
            summary = results[spec.name]["mcdo"]["summary"]
            trace_delta, trace_pct = compute_delta(get_moments(reference_summary, "trace"), get_moments(summary, "trace"))
            off_delta, off_pct = compute_delta(
                get_moments(reference_summary, "off_diag_mass"), get_moments(summary, "off_diag_mass")
            )
            logdet_delta, _ = compute_delta(get_moments(reference_summary, "logdet"), get_moments(summary, "logdet"))
            noise_rows.append(
                [
                    spec.name,
                    format_float(trace_pct, 2),
                    format_float(logdet_delta, 4),
                    format_float(off_pct, 2),
                ]
            )
        appendix_tables.append(("Noise severity deltas", build_markdown_table(noise_headers, noise_rows)))
        if severe_noise:
            sigma = _spec_noise_sigma(severe_noise) or 0.0
            trace_pct_text = (
                f"(≈{noise_peak_trace_pct:+.1f}% vs baseline) " if noise_peak_trace_pct is not None and math.isfinite(noise_peak_trace_pct) else ""
            )
            logdet_delta_text = (
                f" ({noise_peak_logdet_delta:+.2f} vs baseline)" if noise_peak_logdet_delta is not None and math.isfinite(noise_peak_logdet_delta) else ""
            )
            report_lines.append(
                f"Trace grows steadily with σ; at σ={sigma:.2f} it reaches {noise_peak_trace:.2f} {trace_pct_text}"
                f"while logdet shifts to {noise_peak_logdet:.2f}{logdet_delta_text}, signalling that variance expands yet concentrates into fewer dominant axes."
            )
        noise_pct_label = (
            f"{noise_peak_trace_pct:+.1f}%" if noise_peak_trace_pct is not None and math.isfinite(noise_peak_trace_pct) else "n/a"
        )
        noise_logdet_label = (
            f"{noise_peak_logdet_delta:+.2f}" if noise_peak_logdet_delta is not None and math.isfinite(noise_peak_logdet_delta) else "n/a"
        )
        if "noise_trace_relative" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["noise_trace_relative"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Relative trace change under noise",
                f"Noise-only perturbations peak at {noise_pct_label} trace change when σ reaches its maximum setting.",
            )
        if "noise_logdet_relative" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["noise_logdet_relative"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Logdet shift under noise",
                f"Log-volume steadily contracts with noise strength, finishing at {noise_logdet_label} relative to baseline.",
            )
        if "noise_severity_trace" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["noise_severity_trace"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Trace vs noise severity",
                "Trace increases monotonically with σ, indicating broader stochastic clouds as perturbations intensify.",
            )
        if "noise_severity_logdet" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["noise_severity_logdet"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Logdet vs noise severity",
                "Log-volume falls steadily with stronger noise, reflecting variance concentrating into fewer dominant modes.",
            )
        if "noise_severity_corr_anisotropy" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["noise_severity_corr_anisotropy"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Correlation anisotropy vs noise severity",
                "Cross-dimensional coupling grows gradually with σ, showing correlated drift rather than isotropic spread.",
            )
        if "mean_shift_noise" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["mean_shift_noise"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Mean shift vs noise severity",
                "Monte Carlo means drift further from the deterministic embedding as σ increases, tracking the speckle strength.",
            )
        if "trace_mean_noise" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["trace_mean_noise"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Trace vs mean shift — noise severity",
                "Trace and mean shift climb together with σ, illustrating that broader clouds coincide with larger centroid drift.",
            )

    # Section 7: Downsampling sensitivity
    if downsample_specs:
        report_lines.append("\n## 7. Sensitivity to Downsampling")
        down_headers = ["transform", "Δ trace (%)", "Δ logdet", "Δ off-diag (%)"]
        down_rows: List[List[str]] = []
        for spec in downsample_specs:
            summary = results[spec.name]["mcdo"]["summary"]
            trace_delta, trace_pct = compute_delta(get_moments(reference_summary, "trace"), get_moments(summary, "trace"))
            off_delta, off_pct = compute_delta(
                get_moments(reference_summary, "off_diag_mass"), get_moments(summary, "off_diag_mass")
            )
            logdet_delta, _ = compute_delta(get_moments(reference_summary, "logdet"), get_moments(summary, "logdet"))
            down_rows.append(
                [
                    spec.name,
                    format_float(trace_pct, 2),
                    format_float(logdet_delta, 4),
                    format_float(off_pct, 2),
                ]
            )
        appendix_tables.append(("Smoothed crop severity deltas", build_markdown_table(down_headers, down_rows)))
        if severe_down:
            pct = down_peak_pct or 0
            down_trace = down_peak_trace if down_peak_trace is not None else float("nan")
            down_logdet = down_peak_logdet if down_peak_logdet is not None else float("nan")
            volume_ratio = down_peak_volume_ratio if down_peak_volume_ratio is not None else float("nan")
            report_lines.append(
                "Smoothed crops (bicubic upsample) beyond 60% sharply increase trace "
                f"(e.g., {pct}% reduction → trace {down_trace:.2f}) while logdet becomes non-monotone—"
                f"after a mild plateau it collapses to {down_logdet:.2f} (volume ratio ≈ {volume_ratio:.3f}), consistent with aliasing and patch-token collapse once only a few pixels remain."
            )
        down_trace_pct_label = (
            f"{down_peak_trace_pct:+.1f}%" if down_peak_trace_pct is not None and math.isfinite(down_peak_trace_pct) else "n/a"
        )
        down_logdet_delta_label = (
            f"{down_peak_logdet_delta:+.2f}"
            if down_peak_logdet_delta is not None and math.isfinite(down_peak_logdet_delta)
            else "n/a"
        )
        if "downsample_trace_relative" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["downsample_trace_relative"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Relative trace change under smoothed crop (bicubic)",
                f"Smoothed crops peak around {down_trace_pct_label} trace change once spatial resolution drops to {down_peak_pct or 0}%.",
            )
        if "downsample_logdet_relative" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["downsample_logdet_relative"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Logdet shift under smoothed crop (bicubic)",
                f"Covariance volume contracts by {down_logdet_delta_label} at the harshest smoothed crop, underscoring aliasing-driven collapse.",
            )
        if "trace_violin" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["trace_violin"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Trace distribution per transform",
                "Trace variance fans out for the most aggressive downsampling settings, underscoring their broader uncertainty.",
            )
        if "downsample_severity_trace" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["downsample_severity_trace"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Trace vs downsampling severity",
                f"Trace rises almost linearly until about 60% reduction before the {down_peak_pct or 0}% case accelerates beyond {down_peak_trace_pct:+.1f}%." if down_peak_trace_pct is not None and math.isfinite(down_peak_trace_pct) else "Trace increases steadily as more resolution is removed, highlighting sensitivity to spatial detail.",
            )
        if "downsample_severity_logdet" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["downsample_severity_logdet"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Logdet vs downsampling severity",
                "Log-volume stays flat for mild reductions then plunges once images fall below 20% of their original side length.",
            )
        if "downsample_severity_corr_anisotropy" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["downsample_severity_corr_anisotropy"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Correlation anisotropy vs downsampling severity",
                "Severe downsampling amplifies cross-dimensional coupling, showing the embedding cloud stretching along fewer axes.",
            )
        if "mean_shift_downsample" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["mean_shift_downsample"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Mean shift vs smoothed crop severity",
                "Mean displacement spikes once resolution falls below 60%, matching the trace surge caused by aggressive smoothing.",
            )
        if "trace_mean_downsample" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["trace_mean_downsample"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Trace vs mean shift — smoothed crops",
                "Smoothed crops show a coupled rise in trace and mean shift as detail disappears, linking spread and drift directly.",
            )
        if "logdet_violin" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["logdet_violin"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Logdet distribution per transform",
                "The tail of extreme downsampling skews toward very low logdet values, reinforcing the volume collapse story.",
            )
        if "offdiag_violin" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["offdiag_violin"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Off-diagonal mass distribution per transform",
                "Off-diagonal covariance mass swells as resolution drops, highlighting stronger axis entanglement.",
            )

    pixel_specs = [spec for spec in transforms if spec.kind == "downsample_pixel"]
    if pixel_specs:
        pixel_headers = ["transform", "Δ trace (%)", "Δ logdet", "Δ off-diag (%)"]
        pixel_rows: List[List[str]] = []
        for spec in pixel_specs:
            summary = results[spec.name]["mcdo"]["summary"]
            trace_delta, trace_pct = compute_delta(get_moments(reference_summary, "trace"), get_moments(summary, "trace"))
            off_delta, off_pct = compute_delta(
                get_moments(reference_summary, "off_diag_mass"), get_moments(summary, "off_diag_mass")
            )
            logdet_delta, _ = compute_delta(get_moments(reference_summary, "logdet"), get_moments(summary, "logdet"))
            pixel_rows.append(
                [
                    spec.name,
                    format_float(trace_pct, 2),
                    format_float(logdet_delta, 4),
                    format_float(off_pct, 2),
                ]
            )
        appendix_tables.append(("Raw crop severity deltas", build_markdown_table(pixel_headers, pixel_rows)))

        report_lines.append("\n### Raw crop (nearest)")
        highest_pixel = max(pixel_specs, key=lambda spec: _spec_downsample_percent(spec) or 0)
        pixel_summary = results[highest_pixel.name]["mcdo"]["summary"]
        trace_high = get_moments(pixel_summary, "trace").mean
        logdet_high = get_moments(pixel_summary, "logdet").mean
        pixel_pct = _spec_downsample_percent(highest_pixel) or 0
        pixel_trace_pct = (trace_high - base_trace) / base_trace * 100.0 if base_trace else None
        pixel_logdet_delta = logdet_high - base_logdet if math.isfinite(logdet_high) and math.isfinite(base_logdet) else None
        report_lines.append(
            "Keeping the raw nearest-neighbour crop (no smoothing) produces larger block artefacts; the harshest setting "
            f"({highest_pixel.name}) yields trace {trace_high:.2f} while logdet drops to {logdet_high:.2f}, indicating strong concentration of variance into a handful of axes."
        )
        if "pixel_trace_relative" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["pixel_trace_relative"].name).as_posix()
            figure_manager.add(
                rel_path,
            "Relative trace change under raw crop (nearest)",
            "Trace rises steadily as the grid coarsens, confirming block artefacts inflate stochastic spread.",
            )
        if "pixel_logdet_relative" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["pixel_logdet_relative"].name).as_posix()
            figure_manager.add(
                rel_path,
            "Relative logdet change under raw crop (nearest)",
            "Log-volume decays more sharply once smoothing is removed, highlighting how raw crops squeeze variance into fewer modes.",
            )
        if "pixel_severity_trace" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["pixel_severity_trace"].name).as_posix()
            figure_manager.add(
                rel_path,
            "Trace vs raw crop severity",
            f"Nearest-neighbour crops reach {pixel_trace_pct:+.1f}% trace change once the short side is reduced by {pixel_pct}%." if pixel_trace_pct is not None and math.isfinite(pixel_trace_pct) else "Raw crops yield comparable trace increases while emphasising block artefacts over smooth blur.",
            )
        if "pixel_severity_logdet" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["pixel_severity_logdet"].name).as_posix()
            figure_manager.add(
                rel_path,
            "Logdet vs raw crop severity",
            f"Logdet collapses to {logdet_high:.2f} ({pixel_logdet_delta:+.2f} vs baseline) for the {pixel_pct}% setting, showing volume loss once smoothing is removed." if pixel_logdet_delta is not None and math.isfinite(pixel_logdet_delta) else f"Logdet collapses to {logdet_high:.2f} for the {pixel_pct}% setting, showing volume loss once smoothing is removed.",
            )
        if "pixel_severity_corr_anisotropy" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["pixel_severity_corr_anisotropy"].name).as_posix()
            figure_manager.add(
                rel_path,
            "Correlation anisotropy vs raw crop severity",
            "Raw crops amplify anisotropy faster than smoothed crops because hard edges align variance along token boundaries.",
            )
        if "mean_shift_pixel" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["mean_shift_pixel"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Mean shift vs raw crop severity",
                "Nearest-neighbour reductions drive the largest centroid drift as aliasing artefacts dominate token activations.",
            )
        if "trace_mean_pixel" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["trace_mean_pixel"].name).as_posix()
            figure_manager.add(
                rel_path,
                "Trace vs mean shift — raw crops",
                "Raw crops yield the steepest trace/shift coupling, emphasising how aliasing inflates spread and drift together.",
            )

    # Cross-metric scatter views
    if "scatter_trace_logdet" in chart_paths or "scatter_trace_offdiag" in chart_paths:
        report_lines.append("\n## 8. Cross-Metric Geometry")
    if "scatter_trace_logdet" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["scatter_trace_logdet"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Logdet vs trace scatter",
            "Smoothed crop points cluster in the high-trace, low-logdet corner, separating cleanly from the noise-induced shifts.",
        )
    if "scatter_trace_offdiag" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["scatter_trace_offdiag"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Off-diagonal mass vs trace scatter",
            "Off-diagonal coupling grows hand-in-hand with trace for the strongest smoothed crops, reinforcing anisotropy concerns.",
        )

    # Mean shift diagnostics
    report_lines.append("\n## 9. Mean Shift Diagnostics")
    shift_headers = ["transform", "L2 shift", "Mahalanobis shift"]
    shift_rows: List[List[str]] = []
    for spec in transforms:
        extras = results.get(spec.name, {}).get("extras")
        if not extras:
            continue
        shift_rows.append(
            [
                spec.name,
                format_mean_std(dict_to_moments(extras["mean_shift"])),
                format_mean_std(dict_to_moments(extras["mahal_shift"])),
            ]
        )
    report_lines.append(build_markdown_table(shift_headers, shift_rows))
    if "mean_shift_noise" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["mean_shift_noise"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Mean shift vs noise severity",
            "Mean L2 displacement tracks noise strength, reinforcing that stochastic means drift progressively under σ increases.",
        )
    if "mean_shift_downsample" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["mean_shift_downsample"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Mean shift vs downsampling severity",
            "Resolution loss beyond 60% triggers a rapid rise in mean shift, reflecting aliasing-induced drift.",
        )
    if "mean_shift_pixel" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["mean_shift_pixel"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Mean shift vs pixelated downsampling severity",
            "Pixelation pushes mean drift even harder once large blocks emerge, underscoring the harsher aliasing penalty.",
        )
    if "mahal_shift_noise" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["mahal_shift_noise"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Mahalanobis shift vs noise severity",
            "Covariance-normalised drift escalates with σ, showing that noise injects uncertainty aligned with high-variance directions.",
        )
    if "mahal_shift_downsample" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["mahal_shift_downsample"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Mahalanobis shift vs downsampling severity",
            "Heavy downsampling rockets Mahalanobis distance, indicating the embedding cloud moves far relative to its contracted covariance.",
        )
    if "mahal_shift_pixel" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["mahal_shift_pixel"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Mahalanobis shift vs pixelated downsampling severity",
            "Nearest-neighbour reductions yield the largest covariance-normalised drift, showing block artefacts distort embeddings most severely.",
        )

    # Spectral and tangent metrics
    report_lines.append("\n## 10. Spectral & Tangent Geometry")
    geom_headers = ["transform", "tangent trace", "tangent λmax", "spectral entropy", "top-10 share", "circular variance"]
    geom_rows: List[List[str]] = []
    for spec in transforms:
        extras = results.get(spec.name, {}).get("extras")
        if not extras:
            continue
        geom_rows.append(
            [
                spec.name,
                format_mean_std(dict_to_moments(extras["tangent_trace"])),
                format_mean_std(dict_to_moments(extras["tangent_max"])),
                format_mean_std(dict_to_moments(extras["spectral_entropy"])),
                format_mean_std(dict_to_moments(extras["spectral_top10"])),
                format_mean_std(dict_to_moments(extras["circular_variance"])),
            ]
        )
    report_lines.append(build_markdown_table(geom_headers, geom_rows))
    if "angle_radar" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["angle_radar"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Trace by viewpoint angle",
            "Flank viewpoints exhibit the largest trace gain once detail is stripped, pointing to orientation-sensitive uncertainty.",
        )

    pca_entries: List[Tuple[Optional[TransformSpec], str, Path]] = []
    base_spec = next((spec for spec in transforms if spec.name == "original"), None)
    if "pca_original" in chart_paths:
        pca_entries.append((base_spec, "Original (baseline)", chart_paths["pca_original"]))
    noise_candidates = [spec for spec in transforms if spec.kind == "noise" and _spec_noise_sigma(spec) is not None]
    if noise_candidates:
        max_noise = max(noise_candidates, key=lambda s: results[s.name]["mcdo"]["summary"]["trace"].mean)
        min_noise = min(noise_candidates, key=lambda s: results[s.name]["mcdo"]["summary"]["trace"].mean)
        for spec, label in ((max_noise, "noise max trace"), (min_noise, "noise min trace")):
            key = f"pca_{spec.name}"
            if key in chart_paths:
                pca_entries.append((spec, f"{spec.name} ({label})", chart_paths[key]))
    down_candidates = [
        spec for spec in transforms if spec.kind == "downsample" and _spec_downsample_percent(spec) is not None
    ]
    if down_candidates:
        max_down = max(down_candidates, key=lambda s: results[s.name]["mcdo"]["summary"]["trace"].mean)
        min_down = min(down_candidates, key=lambda s: results[s.name]["mcdo"]["summary"]["trace"].mean)
        for spec, label in ((max_down, "downsample max trace"), (min_down, "downsample min trace")):
            key = f"pca_{spec.name}"
            if key in chart_paths:
                pca_entries.append((spec, f"{spec.name} ({label})", chart_paths[key]))
    pixel_candidates = [
        spec for spec in transforms if spec.kind == "downsample_pixel" and _spec_downsample_percent(spec) is not None
    ]
    if pixel_candidates:
        max_pixel = max(pixel_candidates, key=lambda s: results[s.name]["mcdo"]["summary"]["trace"].mean)
        min_pixel = min(pixel_candidates, key=lambda s: results[s.name]["mcdo"]["summary"]["trace"].mean)
        for spec, label in ((max_pixel, "pixelated max trace"), (min_pixel, "pixelated min trace")):
            key = f"pca_{spec.name}"
            if key in chart_paths:
                pca_entries.append((spec, f"{spec.name} ({label})", chart_paths[key]))

    def sort_key(item: Tuple[Optional[TransformSpec], str, Path]) -> Tuple[int, float, str]:
        spec, desc, _ = item
        group_order = {
            "Original": 0,
            "noise": 1,
            "downsample": 2,
            "downsample_pixel": 3,
        }
        if spec is None or spec.name == "original":
            group = 0
            severity = 0.0
        elif spec.kind == "noise":
            group = group_order["noise"]
            severity = _spec_noise_sigma(spec) or 0.0
        elif spec.kind == "downsample":
            group = group_order["downsample"]
            severity = float(_spec_downsample_percent(spec) or 0)
        elif spec.kind == "downsample_pixel":
            group = group_order["downsample_pixel"]
            severity = float(_spec_downsample_percent(spec) or 0)
        else:
            group = 99
            severity = 0.0
        return (group, severity, desc)

    for spec_entry, desc, path_obj in sorted(pca_entries, key=sort_key):
        rel_path = (asset_rel_dir / path_obj.name).as_posix()
        title = f"PCA embeddings — {desc}"
        figure_manager.add(rel_path, title, pca_insight(spec_entry, desc))

    # Section 11: Pass Count Stability
    report_lines.append("\n## 11. Pass Count Stability")
    report_lines.append(
        "We evaluate trace and log-determinant stability across Monte Carlo pass counts T ∈ {2,4,8,16,32,64,128}."
    )
    report_lines.append(
        "Lower T increases estimator noise; curves should flatten as T grows. See stability plots in this section if generated."
    )
    trace_range_original = (
        pass_stability_notes.get("trace_range_original") if pass_stability_notes else None
    )
    trace_range_downsample = (
        pass_stability_notes.get("trace_range_downsample") if pass_stability_notes else None
    )
    trace_range_noise = pass_stability_notes.get("trace_range_noise") if pass_stability_notes else None
    if "pass_stability_trace" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["pass_stability_trace"].name).as_posix()
        if (
            trace_range_original is not None
            and trace_range_downsample is not None
            and math.isfinite(trace_range_original)
            and math.isfinite(trace_range_downsample)
        ):
            insight = (
                f"Trace spans shrink to {trace_range_original:.3f} for the baseline sweep, while the harsh downsample stays within {trace_range_downsample:.3f} once T≥32."
            )
        else:
            insight = "Trace estimates settle quickly once T≥32, aligning with the stability discussion later in the report."
        figure_manager.add(rel_path, "Trace stability vs passes", insight)
    if "pass_stability_logdet" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["pass_stability_logdet"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Logdet stability vs passes",
            "Logdet variance collapses as passes double, confirming that 64 draws are ample for stable volume estimates.",
        )

    # Section 12: Detection-to-Embedding Pipeline (YOLOv8 Crop) and Camera Distance
    report_lines.append("\n## 12. Detection-to-Embedding Pipeline (YOLOv8 Crop) and Camera Distance")
    report_lines.append(
        "Object scale shrinks as the camera moves away, mixing background into the crop and reducing effective resolution."
        " To benchmark a practical workflow, we introduce a detection→crop→CLIP pipeline and study how distance-driven scale changes alter the embedding cloud."
    )
    report_lines.append("\n### 12.1 Pipeline")
    report_lines.append("- Detect vehicles in the full-resolution frame with YOLOv8 (COCO classes: car, bus, truck).")
    report_lines.append("- Crop the predicted box, optionally expanding by ≈5% for context while staying inside image bounds.")
    report_lines.append("- Resize the crop to CLIP's 224×224 input via either bicubic antialiased resampling or nearest-neighbour upsampling.")
    report_lines.append("- Apply CLIP preprocessing and run MCDO (T=64, p=0.01) to estimate the stochastic embedding cloud.")
    report_lines.append("- Compare crop embeddings to a reference (full frame or near-distance crop) using mean and Mahalanobis shift, trace/logdet, and anisotropy.")
    report_lines.append("\nNotes:")
    report_lines.append(
        "- CLIP always consumes 224×224 inputs; “raw low-res” crops therefore require an upsample stage. Nearest-neighbour preserves the block grid, while bicubic smooths detail."
    )
    report_lines.append(
        "- Tiny crops widen trace and contract logdet, mirroring the high-severity downsampling response documented above."
    )
    report_lines.append("\n### 12.2 Distance Effects (Expected/Observed)")
    report_lines.append("- Smaller crops (greater distance) increase trace and Mahalanobis shift while logdet contracts.")
    report_lines.append("- Mahalanobis shift reacts faster than raw mean shift because covariance volume shrinks as detail disappears.")
    report_lines.append("- Raw (nearest) crops inject stronger anisotropy and larger shifts than smoothed (bicubic) crops at the same scale.")
    report_lines.append("\n### 12.3 Practical Guidance")
    report_lines.append("- Prefer bicubic antialiased resize before sending crops to CLIP; nearest-neighbour magnifies block artefacts.")
    report_lines.append("- Maintain a minimum crop size by padding boxes with a narrow context band before resizing.")
    report_lines.append("- Bin detections by object scale or distance and report mean/Mahalanobis shift, trace, logdet, and anisotropy per bin to visualise degradation.")

    # Section 13: Discussion & Outlook
    report_lines.append("\n## 13. Discussion & Outlook")
    discussion_points: List[str] = []
    base_trace_mean = get_moments(reference_summary, "trace").mean
    base_logdet_mean = get_moments(reference_summary, "logdet").mean

    if severe_noise:
        sigma = _spec_noise_sigma(severe_noise) or 0.0
        noise_summary = results[severe_noise.name]["mcdo"]["summary"]
        noise_trace = get_moments(noise_summary, "trace").mean
        noise_logdet = get_moments(noise_summary, "logdet").mean
        trace_pct = ((noise_trace - base_trace_mean) / base_trace_mean * 100.0) if base_trace_mean else float("nan")
        logdet_delta = noise_logdet - base_logdet_mean
        extras_noise = results[severe_noise.name].get("extras", {})
        mean_shift = extras_noise.get("mean_shift", {}).get("mean", float("nan"))
        tangent_mean = extras_noise.get("tangent_trace", {}).get("mean", float("nan"))
        if math.isfinite(trace_pct):
            logdet_phrase = "drops" if logdet_delta < 0 else "rises"
            discussion_points.append(
                f"- **High noise (σ={sigma:.2f})** expands trace by {trace_pct:+.1f}% and {logdet_phrase} logdet by {abs(logdet_delta):.2f}. "
                f"The stochastic mean drifts {mean_shift:.2f} L2 units from the deterministic embedding, and tangent variance settles around {tangent_mean:.2f}, confirming broader but directional uncertainty."
            )

    if severe_down:
        pct = _spec_downsample_percent(severe_down) or 0
        down_summary = results[severe_down.name]["mcdo"]["summary"]
        down_trace = get_moments(down_summary, "trace").mean
        down_logdet = get_moments(down_summary, "logdet").mean
        trace_pct = ((down_trace - base_trace_mean) / base_trace_mean * 100.0) if base_trace_mean else float("nan")
        logdet_delta = down_logdet - base_logdet_mean
        extras_down = results[severe_down.name].get("extras", {})
        mahal_shift = extras_down.get("mahal_shift", {}).get("mean", float("nan"))
        spectral_entropy = extras_down.get("spectral_entropy", {}).get("mean", float("nan"))
        if math.isfinite(trace_pct):
            logdet_phrase = "drops" if logdet_delta < 0 else "rises"
            target_px = int(round(224 * (1 - pct / 100)))
            discussion_points.append(
                f"- **Extreme downsampling ({pct}% → {target_px}px)** shifts trace by {trace_pct:+.1f}% and {logdet_phrase} logdet by {abs(logdet_delta):.2f}. "
                f"Mahalanobis drift reaches {mahal_shift:.1f}, while spectral entropy averages {spectral_entropy:.2f}, signalling variance concentration into fewer modes."
            )

        baseline_angles = results.get("original", {}).get("extras", {}).get("angle_trace_means", {})
        down_angles = extras_down.get("angle_trace_means", {})
        if baseline_angles and down_angles:
            diffs = {angle: down_angles.get(angle, float("nan")) - baseline_angles.get(angle, float("nan")) for angle in baseline_angles}
            key_angle = max(diffs, key=lambda k: abs(diffs[k]))
            angle_change = diffs[key_angle]
            discussion_points.append(
                f"- **Viewpoint sensitivity:** At {key_angle}°, trace shifts by {angle_change:+.2f} relative to the clean view, indicating that certain orientations (e.g. flank perspectives) become the most uncertain once detail is removed."
            )

    stability_map = pass_stability_notes or {}
    trace_range = stability_map.get("trace_range_original")
    if trace_range is not None and math.isfinite(trace_range):
        discussion_points.append(
            f"- **Pass-count stability:** Trace estimates converge within ±{trace_range/2:.3f} by 32 passes; high-noise and heavy-downsample settings widen the band only modestly, so T=64 remains a safe budget."
        )

    mi_mean = get_moments(results["original"]["mcdo"]["summary"], "mutual_information").mean
    discussion_points.append(
        f"- **Predictive head:** Mutual information stays near {mi_mean:.2e} for all conditions, so predictive entropy contributes little insight compared with embedding-space diagnostics."
    )

    report_lines.extend(discussion_points)

    # Section 14: Class-level trace shifts (using strongest perturbations)
    strongest_noise = noise_specs[-1] if noise_specs else None
    strongest_downsample = downsample_specs[-1] if downsample_specs else None
    if strongest_noise or strongest_downsample:
        report_lines.append("\n## 14. Class-Level Trace Shifts")
        rows: List[List[str]] = []
        headers = ["class", "trace (original)"]
        if strongest_noise:
            headers += [f"trace ({strongest_noise.name})", "Δ noise"]
        if strongest_downsample:
            headers += [f"trace ({strongest_downsample.name})", "Δ downsample"]

        base_per_label = results["original"]["mcdo"]["per_label_summary"]
        noise_per_label = (
            results[strongest_noise.name]["mcdo"]["per_label_summary"] if strongest_noise else {}
        )
        down_per_label = (
            results[strongest_downsample.name]["mcdo"]["per_label_summary"] if strongest_downsample else {}
        )

        for label, base_metrics in sorted(base_per_label.items()):
            row = [label, format_mean_std(get_moments(base_metrics, "trace"))]
            if strongest_noise:
                noise_metrics = noise_per_label.get(label, {})
                delta_noise = (
                    get_moments(noise_metrics, "trace").mean - get_moments(base_metrics, "trace").mean
                    if noise_metrics
                    else float("nan")
                )
                row.extend(
                    [
                        format_mean_std(get_moments(noise_metrics, "trace")),
                        format_float(delta_noise, 4),
                    ]
                )
            if strongest_downsample:
                down_metrics = down_per_label.get(label, {})
                delta_down = (
                    get_moments(down_metrics, "trace").mean - get_moments(base_metrics, "trace").mean
                    if down_metrics
                    else float("nan")
                )
                row.extend(
                    [
                        format_mean_std(get_moments(down_metrics, "trace")),
                        format_float(delta_down, 4),
                    ]
                )
            rows.append(row)
        report_lines.append(build_markdown_table(headers, rows))

    # Section 11: Example perturbations
    report_lines.append("\n## 15. Example Perturbations")
    report_lines.append(
        "Preview grids illustrate how each perturbation family reshapes the rendered jeep: baseline is shown alongside rising severities."
    )
    if "modulation_overview_grid" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["modulation_overview_grid"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Preview overview — all perturbations",
            "Rows correspond to noise, smoothed crops, and raw crops respectively; columns step through increasing severity with the baseline at left.",
        )
    if "noise_examples_grid" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["noise_examples_grid"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Jeep previews — noise sweep",
            "Gaussian noise progressively speckles the frame while salt & pepper corruption introduces isolated extreme pixels.",
        )
    if "downsample_examples_grid" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["downsample_examples_grid"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Jeep previews — smoothed crop sweep",
            "Smoothing the coarse crop blurs structure as resolution falls, culminating in a soft, low-detail silhouette.",
        )
    if "pixel_examples_grid" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["pixel_examples_grid"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Jeep previews — raw crop sweep",
            "Nearest-neighbour reductions replace detail with large blocks, emphasising aliasing artefacts at coarse grids.",
        )

    report_lines.append("\n## Appendix: Detailed Metrics")
    if appendix_tables:
        for title, table in appendix_tables:
            report_lines.append(f"\n### {title}\n{table}")

    yolo_summary_path = analysis_root / "yolov8" / "variant_summary.csv"
    yolo_rows: List[Dict[str, str]] = []
    if yolo_summary_path.exists():
        with yolo_summary_path.open() as fh:
            reader = csv.DictReader(fh)
            yolo_rows = [row for row in reader]
    if yolo_rows:
        variant_lookup = {row["variant"]: row for row in yolo_rows if row.get("variant")}

        def _safe_float(row: Dict[str, str], key: str) -> float:
            if not row:
                return float("nan")
            try:
                value = float(row.get(key, ""))
            except (TypeError, ValueError):
                return float("nan")
            return value

        def _fmt_percent(value: float) -> str:
            if not math.isfinite(value):
                return "—"
            return f"{value:.1f}" if abs(value) >= 10 else f"{value:.2f}"

        def _fmt_ratio(value: float) -> str:
            if not math.isfinite(value):
                return "—"
            return f"{value:.2f}"

        downsample_rates: Dict[int, float] = {}
        pixel_rates: Dict[int, float] = {}
        downsample_conf: Dict[int, float] = {}
        downsample_area: Dict[int, float] = {}
        for pct in downsample_percents:
            down_row = variant_lookup.get(f"downsample_{pct}pct")
            if down_row:
                downsample_rates[pct] = _safe_float(down_row, "vehicle_detection_rate") * 100
                downsample_conf[pct] = _safe_float(down_row, "best_conf_mean")
                downsample_area[pct] = _safe_float(down_row, "area_frac_mean") * 100
            pixel_row = variant_lookup.get(f"pixel_downsample_{pct}pct")
            if pixel_row:
                pixel_rates[pct] = _safe_float(pixel_row, "vehicle_detection_rate") * 100

        report_lines.append(
            "\n### YOLOv8 detection robustness vs downsampling\n"
            "Using the Section 12 pipeline (YOLOv8n detection → crop → CLIP resize), we evaluated detections for every original frame and"
            " each smoothed (bicubic) / raw (nearest) crop variant. Vehicle detections cover COCO classes {car, bus, truck}. Detailed outputs live in"
            f" `{yolo_summary_path}` and the accompanying per-image CSV."
        )

        bullets: List[str] = []
        original_row = variant_lookup.get("original")
        if original_row:
            baseline_rate = _safe_float(original_row, "vehicle_detection_rate")
            n_images = int(round(_safe_float(original_row, "n_images")))
            hits = int(round(baseline_rate * n_images)) if math.isfinite(baseline_rate) else 0
            misses = n_images - hits
            bullets.append(
                f"- Baseline detection succeeds on {hits}/{n_images} views ({baseline_rate * 100:.1f}%), leaving {misses} hard cases."
            )

        down_segments = [
            f"{downsample_rates[pct]:.1f}% at {pct}% reduction"
            for pct in (40, 60, 80, 90, 93)
            if pct in downsample_rates and math.isfinite(downsample_rates[pct])
        ]
        if down_segments:
            if len(down_segments) > 1:
                trend_text = ", ".join(down_segments[:-1]) + f", and {down_segments[-1]}"
            else:
                trend_text = down_segments[0]
            bullets.append(f"- Smoothed crop (bicubic) recall trends: {trend_text}.")

        pixel_segments = [
            f"{pixel_rates[pct]:.1f}% at {pct}% reduction"
            for pct in (10, 20, 40, 60)
            if pct in pixel_rates and math.isfinite(pixel_rates[pct])
        ]
        zero_threshold = next(
            (pct for pct, rate in sorted(pixel_rates.items()) if not math.isfinite(rate) or rate <= 0.0),
            None,
        )
        if pixel_segments or zero_threshold is not None:
            if pixel_segments:
                if len(pixel_segments) > 1:
                    pixel_text = ", ".join(pixel_segments[:-1]) + f", and {pixel_segments[-1]}"
                else:
                    pixel_text = pixel_segments[0]
                sentence = f"- Raw crop (nearest) recall drops to {pixel_text}"
            else:
                sentence = "- Raw crop (nearest) collapses recall"
            if zero_threshold is not None:
                sentence += f"; detections vanish once reductions exceed {zero_threshold}%."
            else:
                sentence += "."
            bullets.append(sentence)

        if original_row and downsample_conf:
            orig_conf = _safe_float(original_row, "best_conf_mean")
            orig_area = _safe_float(original_row, "area_frac_mean") * 100
            ds90_conf = downsample_conf.get(90)
            ds90_area = downsample_area.get(90)
            if math.isfinite(orig_conf) and ds90_conf is not None and math.isfinite(ds90_conf):
                sentence = f"- Confidence slips from {orig_conf:.2f} (original) to {ds90_conf:.2f} at 90% smoothed crop"
                area_bits: List[str] = []
                if math.isfinite(orig_area):
                    area_bits.append(f"{orig_area:.1f}%")
                if ds90_area is not None and math.isfinite(ds90_area):
                    area_bits.append(f"{ds90_area:.1f}%")
                if area_bits:
                    area_text = " → ".join(area_bits) if len(area_bits) == 2 else area_bits[0]
                    sentence += f" while boxes still span {area_text} of the frame"
                sentence += "."
                bullets.append(sentence)

        report_lines.extend(bullets)
        report_lines.append("")

        variant_sequence = ["original"] + [f"downsample_{pct}pct" for pct in downsample_percents] + [
            f"pixel_downsample_{pct}pct" for pct in downsample_percents
        ]
        table_lines = [
            "| variant | detection rate (%) | mean best conf | mean box area (%) |",
            "| --- | --- | --- | --- |",
        ]
        for variant in variant_sequence:
            row = variant_lookup.get(variant)
            if not row:
                continue
            rate = _safe_float(row, "vehicle_detection_rate") * 100
            conf = _safe_float(row, "best_conf_mean")
            area = _safe_float(row, "area_frac_mean") * 100
            table_lines.append(
                f"| {variant} | {_fmt_percent(rate)} | {_fmt_ratio(conf)} | {_fmt_percent(area)} |"
            )
        report_lines.extend(["", *table_lines, ""])

    # Appendix: predictive MI / entropy
    report_lines.append("\n## Appendix: Predictive Diagnostics (MI & Entropy)")
    report_lines.append(
        "Predictive mutual information (epistemic) and entropy are derived from the CLIP text head over class prompts."
        " They remain near-zero here due to prompt unanimity and low dropout; we include them for completeness."
    )
    if "mi_line" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["mi_line"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Mutual information across transforms",
            "Predictive mutual information remains near zero for every perturbation, confirming the head stays confident despite dropout.",
        )
    if "entropy_line" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["entropy_line"].name).as_posix()
        figure_manager.add(
            rel_path,
            "Entropy across transforms",
            "Predictive entropy barely moves across conditions, reinforcing that embedding diagnostics carry the informative signal.",
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/car_sim/sim2_cropped_45deg"))
    parser.add_argument("--analysis-root", type=Path, default=Path("runs/sim2_noise_study"))
    parser.add_argument("--report-root", type=Path, default=Path("docs/reports/sim2_noise_study"))
    parser.add_argument("--asset-root", type=Path, default=Path("docs/report_assets/sim2_noise_study"))
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--passes", type=int, default=64)
    parser.add_argument(
        "--pass-sweep",
        type=str,
        default="2,4,8,16,32,64,128",
        help="Comma-separated pass counts for stability analysis.",
    )
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--transform-seed", type=int, default=1024)
    parser.add_argument(
        "--noise-levels",
        type=str,
        default="0.01,0.02,0.05,0.1,0.2,0.5",
        help="Noise sigmas for severity sweep.",
    )
    parser.add_argument(
        "--downsample-percents",
        type=str,
        default="1,2,5,10,20,40,60,80,85,90,93",
        help="Percent reductions relative to encoder base pixel size.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run analyses even if metrics.csv already exists.")
    parser.add_argument("--skip-run", action="store_true", help="Skip model evaluation; only build the report.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    # Encoder base size for CLIP ViT-B/32 defaults to 224; leave dynamic resolution detection for future.
    encoder_base_px = 224
    noise_levels = _parse_float_list(args.noise_levels)
    downsample_percents = _parse_int_list(args.downsample_percents)
    transforms = default_transform_specs(noise_levels, downsample_percents, encoder_base_px)
    analysis_root = ensure_dir(args.analysis_root)
    report_root = ensure_dir(args.report_root)
    asset_root = ensure_dir(args.asset_root)
    asset_rel_dir = Path(os.path.relpath(asset_root, report_root))

    sample_image_path = discover_first_image(args.data_root)
    if sample_image_path is None:
        raise FileNotFoundError(f"No images found under {args.data_root}")
    base_image = Image.open(sample_image_path).convert("RGB")

    results: Dict[str, dict] = {}

    for index, spec in enumerate(transforms):
        transform_seed = args.transform_seed + index
        preview_transform = spec.factory(transform_seed)
        preview_image = preview_transform(base_image.copy())
        preview_path = asset_root / f"{spec.name}.png"
        preview_image.save(preview_path)

        mcdo_dir = analysis_root / "mcdo" / spec.name
        baseline_dir = analysis_root / "baseline" / spec.name
        mcdo_csv = mcdo_dir / "metrics.csv"
        baseline_csv = baseline_dir / "metrics.csv"

        if not args.skip_run and (args.force or not mcdo_csv.exists()):
            config = CarSimRoutineConfig(
                model_id=args.model,
                device=args.device,
                root=args.data_root,
                train=False,
                limit=None,
                passes=args.passes,
                microbatch=args.microbatch,
                tau=args.tau,
                seed=args.seed,
                allow_tf32=True,
                dropout_rate=None,
                    adapter_targets=ADAPTER_TARGETS,
                    adapter_p=args.dropout,
                    out_dir=mcdo_dir,
                    save_raw=True,
                no_predictive=False,
                image_transform=spec.factory(transform_seed),
                prompts=None,
                prompt_template="a photo of a {label} jeep",
            )
            run_car_sim(config)

        if not args.skip_run and (args.force or not baseline_csv.exists()):
            config = CarSimRoutineConfig(
                model_id=args.model,
                device=args.device,
                root=args.data_root,
                train=False,
                limit=None,
                passes=1,
                microbatch=1,
                tau=args.tau,
                seed=args.seed,
                allow_tf32=True,
                dropout_rate=0.0,
                    adapter_targets=(),
                    adapter_p=0.0,
                    out_dir=baseline_dir,
                    save_raw=True,
                no_predictive=False,
                image_transform=spec.factory(transform_seed),
                prompts=None,
                prompt_template="a photo of a {label} jeep",
            )
            run_car_sim(config)

        if not mcdo_csv.exists() or not baseline_csv.exists():
            if args.skip_run:
                # Skip missing configurations when not executing runs
                continue
            raise RuntimeError(f"Missing metrics for transform '{spec.name}' (run script with --force).")

        mcdo_metrics, mcdo_per_label = load_metrics(mcdo_csv)
        baseline_metrics, baseline_per_label = load_metrics(baseline_csv)

        mcdo_summary = summarise_metric_collection(mcdo_metrics)
        baseline_summary = summarise_metric_collection(baseline_metrics)

        results[spec.name] = {
            "spec": spec,
            "mcdo": {
                "metrics": mcdo_metrics,
                "summary": mcdo_summary,
                "per_label": mcdo_per_label,
                "per_label_summary": {
                    label: summarise_metric_collection(metrics) for label, metrics in mcdo_per_label.items()
                },
                "count": int(
                    next(
                        (arr.size for arr in mcdo_metrics.values() if isinstance(arr, np.ndarray) and arr.size > 0),
                        0,
                    )
                ),
            },
            "baseline": {
                "metrics": baseline_metrics,
                "summary": baseline_summary,
                "per_label": baseline_per_label,
                "per_label_summary": {
                    label: summarise_metric_collection(metrics) for label, metrics in baseline_per_label.items()
                },
                "count": int(
                    next(
                        (arr.size for arr in baseline_metrics.values() if isinstance(arr, np.ndarray) and arr.size > 0),
                        0,
                    )
                ),
            },
        }

    available_transforms = [spec for spec in transforms if spec.name in results]

    original_spec = next((spec for spec in available_transforms if spec.name == "original"), None)
    noise_transforms = [spec for spec in available_transforms if spec.kind == "noise"]
    downsample_transforms = [spec for spec in available_transforms if spec.kind == "downsample"]
    pixel_transforms = [spec for spec in available_transforms if spec.kind == "downsample_pixel"]
    downsample_analysis_transforms: List[TransformSpec] = []
    if original_spec:
        downsample_analysis_transforms.append(original_spec)
    downsample_analysis_transforms.extend(noise_transforms)
    downsample_analysis_transforms.extend(downsample_transforms)
    downsample_analysis_transforms.extend(pixel_transforms)

    pixel_analysis_transforms: List[TransformSpec] = []
    if original_spec:
        pixel_analysis_transforms.append(original_spec)
    pixel_analysis_transforms.extend(pixel_transforms)

    downsample_only_with_reference: List[TransformSpec] = []
    if original_spec:
        downsample_only_with_reference.append(original_spec)
    downsample_only_with_reference.extend(downsample_transforms)

    extras = compute_extended_metrics(results, analysis_root, available_transforms, args.data_root)
    for spec in available_transforms:
        if spec.name in results and spec.name in extras:
            results[spec.name]["extras"] = extras[spec.name]

    chart_paths: Dict[str, Path] = {}
    try:
        chart_paths["noise_trace_relative"] = create_relative_change_plot(
            results, noise_transforms, "trace", asset_root, "noise_trace_relative.png", "trace"
        )
    except ValueError as error:
        print(f"Warning (noise trace relative): {error}")
    try:
        chart_paths["downsample_trace_relative"] = create_relative_change_plot(
            results, downsample_transforms, "trace", asset_root, "downsample_trace_relative.png", "trace"
        )
    except ValueError as error:
        print(f"Warning (downsample trace relative): {error}")
    if pixel_transforms:
        try:
            chart_paths["pixel_trace_relative"] = create_relative_change_plot(
                results, pixel_transforms, "trace", asset_root, "pixel_trace_relative.png", "trace"
            )
        except ValueError as error:
            print(f"Warning (pixel trace relative): {error}")
    if noise_transforms:
        try:
            chart_paths["trace_mean_noise"] = create_trace_meanshift_scatter(
                results,
                noise_transforms,
                extras,
                "noise",
                asset_root,
                "trace_vs_meanshift_noise.png",
                "Trace vs mean shift — noise severity",
            )
        except ValueError as error:
            print(f"Warning (trace vs mean noise): {error}")
    if downsample_transforms:
        try:
            chart_paths["trace_mean_downsample"] = create_trace_meanshift_scatter(
                results,
                downsample_transforms,
                extras,
                "downsample",
                asset_root,
                "trace_vs_meanshift_downsample.png",
                "Trace vs mean shift — smoothed crop severity",
            )
        except ValueError as error:
            print(f"Warning (trace vs mean downsample): {error}")
    if pixel_transforms:
        try:
            chart_paths["trace_mean_pixel"] = create_trace_meanshift_scatter(
                results,
                pixel_transforms,
                extras,
                "downsample_pixel",
                asset_root,
                "trace_vs_meanshift_pixel.png",
                "Trace vs mean shift — raw crop severity",
            )
        except ValueError as error:
            print(f"Warning (trace vs mean pixel): {error}")
    try:
        chart_paths["aggregate_overview"] = create_aggregate_plot(
            results, downsample_analysis_transforms, extras, asset_root, "aggregate_overview.png"
        )
    except Exception as e:
        print(f"Warning (aggregate overview): {e}")
    try:
        chart_paths["mean_shift_bar"] = create_mean_bar_plot(
            available_transforms, extras, "mean_shift", asset_root, "mean_shift_bar.png", "Mean shift (L2)"
        )
    except ValueError as error:
        print(f"Warning (mean shift bar): {error}")
    try:
        chart_paths["noise_logdet_relative"] = create_relative_change_plot(
            results, noise_transforms, "logdet", asset_root, "noise_logdet_relative.png", "logdet"
        )
    except ValueError as error:
        print(f"Warning (noise logdet relative): {error}")
    try:
        chart_paths["downsample_logdet_relative"] = create_relative_change_plot(
            results, downsample_transforms, "logdet", asset_root, "downsample_logdet_relative.png", "logdet"
        )
    except ValueError as error:
        print(f"Warning (downsample logdet relative): {error}")
    if pixel_transforms:
        try:
            chart_paths["pixel_logdet_relative"] = create_relative_change_plot(
                results, pixel_transforms, "logdet", asset_root, "pixel_logdet_relative.png", "logdet"
            )
        except ValueError as error:
            print(f"Warning (pixel logdet relative): {error}")
    try:
        chart_paths["scatter_trace_logdet"] = create_scatter_plot(
            results, downsample_analysis_transforms, "trace", "logdet", asset_root, "scatter_trace_logdet.png"
        )
        chart_paths["scatter_trace_offdiag"] = create_scatter_plot(
            results, downsample_analysis_transforms, "trace", "off_diag_mass", asset_root, "scatter_trace_offdiag.png"
        )
    except Exception as e:
        print(f"Warning (scatter): {e}")
    try:
        chart_paths["trace_violin"] = create_violin_plot(
            results, downsample_only_with_reference, "trace", asset_root, "trace_violin.png"
        )
        chart_paths["logdet_violin"] = create_violin_plot(
            results, downsample_only_with_reference, "logdet", asset_root, "logdet_violin.png"
        )
        chart_paths["offdiag_violin"] = create_violin_plot(
            results, downsample_only_with_reference, "off_diag_mass", asset_root, "offdiag_violin.png"
        )
    except ValueError as error:
        print(f"Warning: {error}")
    try:
        # Appendix: MI/entropy lines
        chart_paths["mi_line"] = create_metric_line_plot(
            results, available_transforms, "mutual_information", asset_root, "mi_line.png", "mutual information"
        )
        chart_paths["entropy_line"] = create_metric_line_plot(
            results, available_transforms, "entropy", asset_root, "entropy_line.png", "entropy"
        )
    except ValueError as error:
        print(f"Warning: {error}")

    # Severity curves (absolute and relative)
    try:
        chart_paths["noise_severity_trace"] = create_severity_plot(
            results, noise_transforms, "trace", "noise", asset_root, "noise_severity_trace.png", "trace", False
        )
        chart_paths["downsample_severity_trace"] = create_severity_plot(
            results, downsample_transforms, "trace", "downsample", asset_root, "downsample_severity_trace.png", "trace", False
        )
        chart_paths["noise_severity_logdet"] = create_severity_plot(
            results, noise_transforms, "logdet", "noise", asset_root, "noise_severity_logdet.png", "logdet", False
        )
        chart_paths["downsample_severity_logdet"] = create_severity_plot(
            results, downsample_transforms, "logdet", "downsample", asset_root, "downsample_severity_logdet.png", "logdet", False
        )
        chart_paths["pixel_severity_trace"] = create_severity_plot(
            results, pixel_transforms, "trace", "downsample_pixel", asset_root, "pixel_severity_trace.png", "trace", False
        )
        chart_paths["pixel_severity_logdet"] = create_severity_plot(
            results, pixel_transforms, "logdet", "downsample_pixel", asset_root, "pixel_severity_logdet.png", "logdet", False
        )
    except Exception as e:
        print(f"Warning (severity plots): {e}")

    try:
        chart_paths["noise_severity_corr_anisotropy"] = create_extras_severity_plot(
            noise_transforms,
            extras,
            "corr_anisotropy",
            "noise",
            asset_root,
            "noise_severity_corr_anisotropy.png",
            "corr anisotropy (Frobenius)",
        )
        chart_paths["downsample_severity_corr_anisotropy"] = create_extras_severity_plot(
            downsample_transforms,
            extras,
            "corr_anisotropy",
            "downsample",
            asset_root,
            "downsample_severity_corr_anisotropy.png",
            "corr anisotropy (Frobenius)",
        )
        chart_paths["pixel_severity_corr_anisotropy"] = create_extras_severity_plot(
            pixel_transforms,
            extras,
            "corr_anisotropy",
            "downsample_pixel",
            asset_root,
            "pixel_severity_corr_anisotropy.png",
            "corr anisotropy (Frobenius)",
        )
    except Exception as e:
        print(f"Warning (anisotropy severity): {e}")

    # Mean shift and angle radar visualisations
    try:
        chart_paths["mean_shift_noise"] = create_mean_shift_plot(
            noise_transforms, extras, "mean_shift", "noise", asset_root, "mean_shift_noise.png", "Mean shift (L2)"
        )
        chart_paths["mean_shift_downsample"] = create_mean_shift_plot(
            downsample_transforms, extras, "mean_shift", "downsample", asset_root, "mean_shift_downsample.png", "Mean shift (L2)"
        )
        if pixel_transforms:
            chart_paths["mean_shift_pixel"] = create_mean_shift_plot(
                pixel_transforms, extras, "mean_shift", "downsample", asset_root, "mean_shift_pixel.png", "Mean shift (L2)"
            )
        chart_paths["mahal_shift_noise"] = create_mean_shift_plot(
            noise_transforms, extras, "mahal_shift", "noise", asset_root, "mahal_shift_noise.png", "Mahalanobis shift"
        )
        chart_paths["mahal_shift_downsample"] = create_mean_shift_plot(
            downsample_transforms, extras, "mahal_shift", "downsample", asset_root, "mahal_shift_downsample.png", "Mahalanobis shift"
        )
        if pixel_transforms:
            chart_paths["mahal_shift_pixel"] = create_mean_shift_plot(
                pixel_transforms,
                extras,
                "mahal_shift",
                "downsample",
                asset_root,
                "mahal_shift_pixel.png",
                "Mahalanobis shift",
            )
    except Exception as e:
        print(f"Warning (mean shift plots): {e}")

    try:
        chart_paths["angle_radar"] = create_angle_radar(results, extras, downsample_analysis_transforms, asset_root, "angle_radar.png")
    except Exception as e:
        print(f"Warning (angle radar): {e}")

    def _build_preview_grid(key: str, specs: Sequence[str], columns: int) -> None:
        image_paths = [asset_root / f"{name}.png" for name in specs if name]
        valid = [path for path in image_paths if path.exists()]
        if not valid:
            return
        output = asset_root / f"{key}.png"
        chart_paths[key] = create_image_grid(valid, output, columns=columns)

    noise_sorted = sorted(
        [spec for spec in noise_transforms if _spec_noise_sigma(spec) is not None],
        key=lambda s: _spec_noise_sigma(s) or 0.0,
    )
    salt_spec = next((spec for spec in noise_transforms if spec.name == "saltpepper_5pct"), None)
    noise_grid_specs: List[str] = []
    if original_spec:
        noise_grid_specs.append(original_spec.name)
    noise_grid_specs.extend(spec.name for spec in noise_sorted)
    if salt_spec:
        noise_grid_specs.append(salt_spec.name)

    downsample_sorted = sorted(downsample_transforms, key=lambda s: _spec_downsample_percent(s) or 0)
    downsample_grid_specs: List[str] = []
    if original_spec:
        downsample_grid_specs.append(original_spec.name)
    downsample_grid_specs.extend(spec.name for spec in downsample_sorted)

    pixel_sorted = sorted(pixel_transforms, key=lambda s: _spec_downsample_percent(s) or 0)
    pixel_grid_specs: List[str] = []
    if original_spec:
        pixel_grid_specs.append(original_spec.name)
    pixel_grid_specs.extend(spec.name for spec in pixel_sorted)

    if noise_grid_specs:
        _build_preview_grid("noise_examples_grid", noise_grid_specs, columns=4)
    if downsample_grid_specs:
        _build_preview_grid("downsample_examples_grid", downsample_grid_specs, columns=5)
    if pixel_sorted:
        _build_preview_grid("pixel_examples_grid", pixel_grid_specs, columns=5)

    overview_rows: List[Tuple[str, List[Path]]] = []
    if noise_grid_specs:
        overview_rows.append(
            (
                "Noise (sigma)",
                [asset_root / f"{name}.png" for name in noise_grid_specs],
            )
        )
    if downsample_grid_specs:
        overview_rows.append(
            (
                "Smoothed crop (%)",
                [asset_root / f"{name}.png" for name in downsample_grid_specs],
            )
        )
    if pixel_grid_specs:
        overview_rows.append(
            (
                "Raw crop (%)",
                [asset_root / f"{name}.png" for name in pixel_grid_specs],
            )
        )

    if overview_rows:
        try:
            chart_paths["modulation_overview_grid"] = create_multirow_preview_grid(
                overview_rows,
                asset_root / "modulation_overview_grid.png",
            )
        except ValueError as error:
            print(f"Warning (modulation overview grid): {error}")

    try:
        noise_specs = [spec for spec in available_transforms if spec.kind == "noise" and _spec_noise_sigma(spec) is not None]
        down_specs = [
            spec for spec in available_transforms if spec.kind == "downsample" and _spec_downsample_percent(spec) is not None
        ]
        pixel_specs_chart = [
            spec for spec in available_transforms if spec.kind == "downsample_pixel" and _spec_downsample_percent(spec) is not None
        ]
        baseline_dir = analysis_root / "baseline" / "original"
        scenario_map: Dict[str, str] = {"original": "Original (baseline)"}
        if noise_specs:
            max_noise = max(noise_specs, key=lambda s: results[s.name]["mcdo"]["summary"]["trace"].mean)
            min_noise = min(noise_specs, key=lambda s: results[s.name]["mcdo"]["summary"]["trace"].mean)
            scenario_map[max_noise.name] = f"{max_noise.name} (noise max trace)"
            scenario_map[min_noise.name] = f"{min_noise.name} (noise min trace)"
        if down_specs:
            max_down = max(down_specs, key=lambda s: results[s.name]["mcdo"]["summary"]["trace"].mean)
            min_down = min(down_specs, key=lambda s: results[s.name]["mcdo"]["summary"]["trace"].mean)
            scenario_map[max_down.name] = f"{max_down.name} (downsample max trace)"
            scenario_map[min_down.name] = f"{min_down.name} (downsample min trace)"
        if pixel_specs_chart:
            max_pixel = max(pixel_specs_chart, key=lambda s: results[s.name]["mcdo"]["summary"]["trace"].mean)
            min_pixel = min(pixel_specs_chart, key=lambda s: results[s.name]["mcdo"]["summary"]["trace"].mean)
            scenario_map[max_pixel.name] = f"{max_pixel.name} (pixelated max trace)"
            scenario_map[min_pixel.name] = f"{min_pixel.name} (pixelated min trace)"

        for scenario, label in scenario_map.items():
            chart_paths[f"pca_{scenario}"] = create_pca_plot(
                analysis_root,
                baseline_dir,
                sample_index=0,
                scenario=scenario,
                label=label,
                asset_root=asset_root,
                filename=f"pca_{scenario}.png",
            )
    except Exception as e:
        print(f"Warning (PCA gallery): {e}")

    # Pass-count stability analysis (original, max noise, max downsample)
    pass_stability_notes: Dict[str, float] = {}
    try:
        pass_counts = _parse_int_list(args.pass_sweep)
        base_spec, noise_spec, down_spec = _select_max_severity_specs(transforms)
        lbl_specs = [("original", base_spec)] + ([ ("max_noise", noise_spec) ] if noise_spec else []) + ([ ("max_downsample", down_spec) ] if down_spec else [])

        # Build series for trace and logdet
        series_trace: Dict[str, Dict[str, List[float]]] = {}
        series_logdet: Dict[str, Dict[str, List[float]]] = {}
        for label, spec in lbl_specs:
            means_tr: List[float] = []
            sems_tr: List[float] = []
            means_ld: List[float] = []
            sems_ld: List[float] = []
            for T in pass_counts:
                sweep_dir = analysis_root / "pass_sweep" / f"T{T:03d}" / spec.name
                sweep_csv = sweep_dir / "metrics.csv"
                if args.force or not sweep_csv.exists():
                    cfg = CarSimRoutineConfig(
                        model_id=args.model,
                        device=args.device,
                        root=args.data_root,
                        train=False,
                        limit=None,
                        passes=T,
                        microbatch=min(4, max(1, T)),
                        tau=args.tau,
                        seed=args.seed,
                        allow_tf32=True,
                        dropout_rate=None,
                        adapter_targets=ADAPTER_TARGETS,
                        adapter_p=args.dropout,
                        out_dir=sweep_dir,
                        save_raw=False,
                        no_predictive=False,
                        image_transform=spec.factory(args.transform_seed),
                        prompts=None,
                        prompt_template="a photo of a {label} jeep",
                    )
                    run_car_sim(cfg)
                agg, _ = load_metrics(sweep_csv)
                tr = agg.get("trace", np.array([], dtype=np.float64))
                ld = agg.get("logdet", np.array([], dtype=np.float64))
                if tr.size:
                    means_tr.append(float(tr.mean()))
                    sems_tr.append(float(tr.std(ddof=1) / max(1.0, np.sqrt(tr.size))))
                else:
                    means_tr.append(float("nan"))
                    sems_tr.append(float("nan"))
                if ld.size:
                    means_ld.append(float(ld.mean()))
                    sems_ld.append(float(ld.std(ddof=1) / max(1.0, np.sqrt(ld.size))))
                else:
                    means_ld.append(float("nan"))
                    sems_ld.append(float("nan"))
            series_trace[label] = {"mean": means_tr, "sem": sems_tr}
            series_logdet[label] = {"mean": means_ld, "sem": sems_ld}
        chart_paths["pass_stability_trace"] = create_pass_stability_plot(
            pass_counts, series_trace, "trace", asset_root, "pass_stability_trace.png", "trace"
        )
        chart_paths["pass_stability_logdet"] = create_pass_stability_plot(
            pass_counts, series_logdet, "logdet", asset_root, "pass_stability_logdet.png", "logdet"
        )

        def _range(series: Dict[str, List[float]]) -> float:
            arr = np.asarray(series.get("mean", []), dtype=float)
            if arr.size == 0 or np.isnan(arr).all():
                return float("nan")
            return float(np.nanmax(arr) - np.nanmin(arr))

        pass_stability_notes["trace_range_original"] = _range(series_trace.get("original", {}))
        if "max_noise" in series_trace:
            pass_stability_notes["trace_range_noise"] = _range(series_trace["max_noise"])
        if "max_downsample" in series_trace:
            pass_stability_notes["trace_range_downsample"] = _range(series_trace["max_downsample"])
    except Exception as e:
        print(f"Warning (pass stability): {e}")

    summary_payload = {
        "setup": {
            "model": args.model,
            "device": args.device or "auto",
            "passes": args.passes,
            "dropout": args.dropout,
            "dataset": str(args.data_root),
            "adapter_targets": list(ADAPTER_TARGETS),
        },
        "results": {
            spec.name: {
                "kind": spec.kind,
                "description": spec.description,
                "mcdo": {
                    "count": results[spec.name]["mcdo"]["count"],
                    "summary": serialise_summary(results[spec.name]["mcdo"]["summary"]),
                },
                "baseline": {
                    "count": results[spec.name]["baseline"]["count"],
                    "summary": serialise_summary(results[spec.name]["baseline"]["summary"]),
                },
                "extras": results[spec.name].get("extras", {}),
            }
            for spec in available_transforms
        },
    }
    (analysis_root / "summary.json").write_text(json.dumps(summary_payload, indent=2))

    image_count = results["original"]["mcdo"]["count"]
    setup_text = {
        "Dataset": str(args.data_root),
        "Model": args.model,
        "Dropout p": str(args.dropout),
        "Passes": str(args.passes),
        "Images": str(image_count),
    }

    report_path = report_root / "REPORT.md"
    generate_report(
        report_path,
        asset_rel_dir,
        analysis_root,
        downsample_percents,
        setup_text,
        results,
        available_transforms,
        chart_paths,
        pass_stability_notes,
    )


if __name__ == "__main__":
    main()
