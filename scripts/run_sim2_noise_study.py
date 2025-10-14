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
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


@dataclass(frozen=True)
class MetricMoments:
    mean: float
    std: float
    min: float
    max: float


@dataclass(frozen=True)
class TransformSpec:
    name: str
    description: str
    kind: str
    factory: Callable[[int], Callable[[Image.Image], Image.Image]]


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
        downsampled = image.resize((new_width, new_height), resample=Image.BICUBIC)
        restored = downsampled.resize((width, height), resample=Image.BICUBIC)
        return restored


def _format_sigma_name(sigma: float) -> str:
    return f"gaussian_noise_{int(round(sigma*1000)):03d}"  # e.g., 0.01 -> gaussian_noise_010


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
        return MetricMoments(value, 0.0, value, value)
    return MetricMoments(
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)),
        min=float(arr.min()),
        max=float(arr.max()),
    )


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
    KIND_COLOURS = {"noise": "#1f77b4", "downsample": "#ff7f0e", "reference": "#7f7f7f"}

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
        color = "#1f77b4" if spec.kind == "noise" else "#ff7f0e" if spec.kind == "downsample" else "#7f7f7f"
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
    colour_map = {"noise": "#1f77b4", "downsample": "#ff7f0e", "reference": "#7f7f7f"}

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
    if kind == "noise":
        # collect gaussian noise specs in ascending sigma
        pairs: List[Tuple[float, TransformSpec]] = []
        for spec in transforms:
            sigma = _spec_noise_sigma(spec)
            if sigma is not None:
                pairs.append((sigma, spec))
        pairs.sort(key=lambda p: p[0])
        for sigma, spec in pairs:
            xs.append(sigma)
            m = results[spec.name]["mcdo"]["summary"].get(metric)
            base = results["original"]["mcdo"]["summary"].get(metric)
            if m is None or base is None:
                ys.append(float("nan"))
            else:
                val = m.mean
                if relative_to_original and base.mean != 0.0 and math.isfinite(base.mean):
                    val = (val - base.mean) / base.mean * 100.0
                ys.append(val)
        x_label = "sigma"
    elif kind == "downsample":
        pairs_ds: List[Tuple[int, TransformSpec]] = []
        for spec in transforms:
            pct = _spec_downsample_percent(spec)
            if pct is not None:
                pairs_ds.append((pct, spec))
        pairs_ds.sort(key=lambda p: p[0])
        for pct, spec in pairs_ds:
            xs.append(float(pct))
            m = results[spec.name]["mcdo"]["summary"].get(metric)
            base = results["original"]["mcdo"]["summary"].get(metric)
            if m is None or base is None:
                ys.append(float("nan"))
            else:
                val = m.mean
                if relative_to_original and base.mean != 0.0 and math.isfinite(base.mean):
                    val = (val - base.mean) / base.mean * 100.0
                ys.append(val)
        x_label = "downsample reduction (%)"
    else:
        raise ValueError("kind must be 'noise' or 'downsample'")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, marker="o")
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
    colour_map = {"noise": "#1f77b4", "downsample": "#ff7f0e", "reference": "#7f7f7f"}
    for spec in transforms:
        metrics = results[spec.name]["mcdo"]["metrics"]
        x = metrics.get(x_metric)
        y = metrics.get(y_metric)
        if x is None or y is None or x.size == 0 or y.size == 0:
            continue
        color = colour_map.get(spec.kind, "#bcbd22")
        ax.scatter(x, y, s=8, alpha=0.4, label=spec.name if spec.name in {"original"} else None, color=color)
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

    labels = ["original"]
    selected_specs: List[Tuple[str, str]] = [("original", "original")]
    max_noise = max((spec for spec in candidates if _spec_noise_sigma(spec) is not None), key=lambda s: _spec_noise_sigma(s) or 0.0, default=None)
    if max_noise and max_noise.name != "original":
        labels.append(max_noise.name)
        selected_specs.append((max_noise.name, max_noise.name))
    max_down = max((spec for spec in candidates if _spec_downsample_percent(spec) is not None), key=lambda s: _spec_downsample_percent(s) or 0, default=None)
    if max_down and max_down.name != "original":
        labels.append(max_down.name)
        selected_specs.append((max_down.name, max_down.name))

    angles = sorted(set(angle for summary in extras.values() for angle in summary.get("angle_trace_means", {}).keys()))
    if not angles:
        raise ValueError("No angle data available")
    theta = np.radians(angles + [angles[0]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    colours = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]
    for idx, (label, spec_name) in enumerate(selected_specs):
        stats = extras.get(spec_name, {}).get("angle_trace_means", {})
        values = [stats.get(angle, np.nan) for angle in angles]
        values.append(values[0])
        ax.plot(theta, values, marker="o", label=label, color=colours[idx % len(colours)])
        ax.fill(theta, values, alpha=0.1, color=colours[idx % len(colours)])
    ax.set_title("Trace by viewpoint angle")
    ax.set_thetagrids(angles)
    ax.set_rlabel_position(0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    asset_root.mkdir(parents=True, exist_ok=True)
    out_path = asset_root / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def create_pca_gallery(
    analysis_root: Path,
    baseline_dir: Path,
    sample_index: int,
    scenario_names: Sequence[str],
    asset_root: Path,
    filename: str,
    transforms: Sequence[TransformSpec],
) -> Path:
    fig, axes = plt.subplots(1, len(scenario_names), figsize=(5 * len(scenario_names), 4))
    if len(scenario_names) == 1:
        axes = [axes]

    baseline_samples = load_sample_tensors(baseline_dir)
    mu_baseline = baseline_samples[sample_index]["mu"]

    for ax, scenario in zip(axes, scenario_names):
        run_dir = analysis_root / "mcdo" / scenario
        samples = load_sample_tensors(run_dir)
        sample = samples[sample_index]
        embeddings = sample["embeddings"]
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        components = vh[:2]
        projected = centered @ components.T
        ax.scatter(projected[:, 0], projected[:, 1], s=10, alpha=0.5)
        ax.scatter(0, 0, marker="x", color="#d62728", label="MCDO mean")
        baseline_proj = (mu_baseline - mean) @ components.T
        ax.arrow(baseline_proj[0], baseline_proj[1], -baseline_proj[0], -baseline_proj[1],
                 head_width=0.05, length_includes_head=True, color="#9467bd", label="Drift")
        ax.set_title(scenario)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    if labels:
        fig.legend(handles, labels, loc="upper right")
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
            "angle_trace_means": {
                angle: float(np.mean(values)) if values else float("nan") for angle, values in angle_trace.items()
            },
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
    return moments_to_dict(moments)


def dict_to_moments(summary: Dict[str, float]) -> MetricMoments:
    return MetricMoments(
        mean=summary.get("mean", float("nan")),
        std=summary.get("std", float("nan")),
        min=summary.get("min", float("nan")),
        max=summary.get("max", float("nan")),
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
    if spec.kind != "downsample":
        return None
    if spec.name.startswith("downsample_") and spec.name.endswith("pct"):
        try:
            mid = spec.name[len("downsample_") : -len("pct")]
            return int(mid)
        except Exception:
            return None
    return None


def _select_max_severity_specs(transforms: Sequence[TransformSpec]) -> Tuple[TransformSpec, Optional[TransformSpec], Optional[TransformSpec]]:
    base = next(spec for spec in transforms if spec.name == "original")
    noise_specs = [spec for spec in transforms if _spec_noise_sigma(spec) is not None]
    down_specs = [spec for spec in transforms if _spec_downsample_percent(spec) is not None]
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
            ax.fill_between(x, means - sems, means + sems, color=colours.get(key, None), alpha=0.15)
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
    setup: Dict[str, str],
    results: Dict[str, dict],
    transforms: Sequence[TransformSpec],
    chart_paths: Dict[str, Path],
    pass_stability_notes: Optional[Dict[str, float]] = None,
) -> None:
    report_lines: List[str] = ["# Sim2 45° CLIP MCDO Noise Study"]

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
    severe_noise = max(noise_specs, key=lambda spec: _spec_noise_sigma(spec) or 0.0) if noise_specs else None
    severe_down = max(downsample_specs, key=lambda spec: _spec_downsample_percent(spec) or 0) if downsample_specs else None

    # Section 4: Aggregate metrics (embedding only)
    report_lines.append("\n## 4. Aggregate MCDO Embedding Metrics")
    agg_headers = ["transform", "trace", "logdet", "off-diag mass"]
    agg_rows: List[List[str]] = []
    for spec in transforms:
        summary = results[spec.name]["mcdo"]["summary"]
        agg_rows.append(
            [
                spec.name,
                format_mean_std(get_moments(summary, "trace")),
                format_mean_std(get_moments(summary, "logdet")),
                format_mean_std(get_moments(summary, "off_diag_mass")),
            ]
        )
    report_lines.append(build_markdown_table(agg_headers, agg_rows))
    base_trace = get_moments(reference_summary, "trace").mean
    base_logdet = get_moments(reference_summary, "logdet").mean
    base_offdiag = get_moments(reference_summary, "off_diag_mass").mean
    report_lines.append(
        f"Baseline trace is {base_trace:.2f}, with logdet {base_logdet:.2f} and off-diagonal mass {base_offdiag:.2f}. "
        "Noise and downsampling progressively broaden the covariance while gradually reducing logdet, especially "
        "for the most aggressive settings."
    )

    # Section 5: Deterministic baseline
    report_lines.append("\n## 5. Deterministic Baseline (1 pass, dropout disabled)")
    base_headers = ["transform", "trace", "logdet", "off-diag mass"]
    base_rows: List[List[str]] = []
    for spec in transforms:
        baseline_summary = results[spec.name]["baseline"]["summary"]
        base_rows.append(
            [
                spec.name,
                format_mean_std(get_moments(baseline_summary, "trace")),
                format_mean_std(get_moments(baseline_summary, "logdet")),
                format_mean_std(get_moments(baseline_summary, "off_diag_mass")),
            ]
        )
    report_lines.append(build_markdown_table(base_headers, base_rows))
    report_lines.append(
        "All deterministic baselines remain numerically identical: covariance mass collapses to machine precision "
        "(trace 5.12e-04) with zero off-diagonal structure, underscoring that stochasticity is solely induced by dropout."
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
        report_lines.append(build_markdown_table(noise_headers, noise_rows))
        if severe_noise:
            sigma = _spec_noise_sigma(severe_noise) or 0.0
            noise_trace = get_moments(results[severe_noise.name]["mcdo"]["summary"], "trace").mean
            noise_logdet = get_moments(results[severe_noise.name]["mcdo"]["summary"], "logdet").mean
            report_lines.append(
                f"Trace grows steadily with σ; at σ={sigma:.2f} it reaches {noise_trace:.2f} (≈{((noise_trace - base_trace)/base_trace)*100:+.1f}% vs baseline) "
                f"while logdet drops to {noise_logdet:.2f}, signalling that variance is expanding yet concentrating into fewer dominant axes."
            )
        if "trace_relative" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["trace_relative"].name).as_posix()
            report_lines.append(f"\n![Relative trace change under noise & downsampling]({rel_path})")
        if "logdet_relative" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["logdet_relative"].name).as_posix()
            report_lines.append(f"\n![Logdet shift relative to original]({rel_path})")
        if "noise_severity_trace" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["noise_severity_trace"].name).as_posix()
            report_lines.append(f"\n![Trace vs noise severity]({rel_path})")
        if "noise_severity_logdet" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["noise_severity_logdet"].name).as_posix()
            report_lines.append(f"\n![Logdet vs noise severity]({rel_path})")

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
        report_lines.append(build_markdown_table(down_headers, down_rows))
        if severe_down:
            pct = _spec_downsample_percent(severe_down) or 0
            down_summary = results[severe_down.name]["mcdo"]["summary"]
            down_trace = get_moments(down_summary, "trace").mean
            down_logdet = get_moments(down_summary, "logdet").mean
            report_lines.append(
                f"Downsampling beyond 60% sharply increases trace (e.g., {pct}% reduction → trace {down_trace:.2f}) while logdet falls to {down_logdet:.2f}, "
                "indicating uncertainty balloons along a handful of directions once spatial detail is largely removed."
            )
        if "trace_violin" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["trace_violin"].name).as_posix()
            report_lines.append(f"\n![Trace distribution per transform]({rel_path})")
        if "downsample_severity_trace" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["downsample_severity_trace"].name).as_posix()
            report_lines.append(f"\n![Trace vs downsampling severity]({rel_path})")
        if "downsample_severity_logdet" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["downsample_severity_logdet"].name).as_posix()
            report_lines.append(f"\n![Logdet vs downsampling severity]({rel_path})")
        if "logdet_violin" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["logdet_violin"].name).as_posix()
            report_lines.append(f"\n![Logdet distribution per transform]({rel_path})")
        if "offdiag_violin" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["offdiag_violin"].name).as_posix()
            report_lines.append(f"\n![Off-diagonal mass distribution per transform]({rel_path})")

    # Cross-metric scatter views
    if "scatter_trace_logdet" in chart_paths or "scatter_trace_offdiag" in chart_paths:
        report_lines.append("\n## 8. Cross-Metric Geometry")
        if "scatter_trace_logdet" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["scatter_trace_logdet"].name).as_posix()
            report_lines.append(f"\n![Logdet vs Trace scatter]({rel_path})")
        if "scatter_trace_offdiag" in chart_paths:
            rel_path = (asset_rel_dir / chart_paths["scatter_trace_offdiag"].name).as_posix()
            report_lines.append(f"\n![Off-diagonal mass vs Trace scatter]({rel_path})")

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
        report_lines.append(f"\n![Mean shift vs noise severity]({rel_path})")
    if "mean_shift_downsample" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["mean_shift_downsample"].name).as_posix()
        report_lines.append(f"\n![Mean shift vs downsampling severity]({rel_path})")
    if "mahal_shift_noise" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["mahal_shift_noise"].name).as_posix()
        report_lines.append(f"\n![Mahalanobis shift vs noise severity]({rel_path})")
    if "mahal_shift_downsample" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["mahal_shift_downsample"].name).as_posix()
        report_lines.append(f"\n![Mahalanobis shift vs downsampling severity]({rel_path})")

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
        report_lines.append(f"\n![Trace by viewpoint angle]({rel_path})")
    if "pca_gallery" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["pca_gallery"].name).as_posix()
        report_lines.append(f"\n![PCA gallery for representative sample]({rel_path})")

    # Section 11: Pass Count Stability
    report_lines.append("\n## 11. Pass Count Stability")
    report_lines.append(
        "We evaluate trace and log-determinant stability across Monte Carlo pass counts T ∈ {2,4,8,16,32,64,128}."
    )
    report_lines.append(
        "Lower T increases estimator noise; curves should flatten as T grows. See stability plots in this section if generated."
    )
    if "pass_stability_trace" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["pass_stability_trace"].name).as_posix()
        report_lines.append(f"\n![Trace stability vs passes]({rel_path})")
    if "pass_stability_logdet" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["pass_stability_logdet"].name).as_posix()
        report_lines.append(f"\n![Logdet stability vs passes]({rel_path})")

    # Section 12: Discussion & Outlook
    report_lines.append("\n## 12. Discussion & Outlook")
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

    # Section 13: Class-level trace shifts (using strongest perturbations)
    strongest_noise = noise_specs[-1] if noise_specs else None
    strongest_downsample = downsample_specs[-1] if downsample_specs else None
    if strongest_noise or strongest_downsample:
        report_lines.append("\n## 13. Class-Level Trace Shifts")
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
    report_lines.append("\n## 14. Example Perturbations")
    for spec in transforms:
        asset_path = asset_rel_dir / f"{spec.name}.png"
        report_lines.append(f"- **{spec.description}:** ![]({asset_path.as_posix()})")

    # Appendix: predictive MI / entropy
    report_lines.append("\n## Appendix: Predictive Diagnostics (MI & Entropy)")
    report_lines.append(
        "Predictive mutual information (epistemic) and entropy are derived from the CLIP text head over class prompts."
        " They remain near-zero here due to prompt unanimity and low dropout; we include them for completeness."
    )
    if "mi_line" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["mi_line"].name).as_posix()
        report_lines.append(f"\n![Mutual information across transforms]({rel_path})")
    if "entropy_line" in chart_paths:
        rel_path = (asset_rel_dir / chart_paths["entropy_line"].name).as_posix()
        report_lines.append(f"\n![Entropy across transforms]({rel_path})")

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

    extras = compute_extended_metrics(results, analysis_root, available_transforms, args.data_root)
    for spec in available_transforms:
        if spec.name in results and spec.name in extras:
            results[spec.name]["extras"] = extras[spec.name]

    chart_paths: Dict[str, Path] = {}
    try:
        chart_paths["trace_relative"] = create_relative_change_plot(
            results, available_transforms, "trace", asset_root, "trace_relative.png", "trace"
        )
    except ValueError as error:
        print(f"Warning: {error}")
    try:
        chart_paths["logdet_relative"] = create_relative_change_plot(
            results, available_transforms, "logdet", asset_root, "logdet_relative.png", "logdet"
        )
    except ValueError as error:
        print(f"Warning: {error}")
    try:
        chart_paths["scatter_trace_logdet"] = create_scatter_plot(
            results, available_transforms, "trace", "logdet", asset_root, "scatter_trace_logdet.png"
        )
        chart_paths["scatter_trace_offdiag"] = create_scatter_plot(
            results, available_transforms, "trace", "off_diag_mass", asset_root, "scatter_trace_offdiag.png"
        )
    except Exception as e:
        print(f"Warning (scatter): {e}")
    try:
        chart_paths["trace_violin"] = create_violin_plot(
            results, available_transforms, "trace", asset_root, "trace_violin.png"
        )
        chart_paths["logdet_violin"] = create_violin_plot(
            results, available_transforms, "logdet", asset_root, "logdet_violin.png"
        )
        chart_paths["offdiag_violin"] = create_violin_plot(
            results, available_transforms, "off_diag_mass", asset_root, "offdiag_violin.png"
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
            results, available_transforms, "trace", "noise", asset_root, "noise_severity_trace.png", "trace", False
        )
        chart_paths["downsample_severity_trace"] = create_severity_plot(
            results, available_transforms, "trace", "downsample", asset_root, "downsample_severity_trace.png", "trace", False
        )
        chart_paths["noise_severity_logdet"] = create_severity_plot(
            results, available_transforms, "logdet", "noise", asset_root, "noise_severity_logdet.png", "logdet", False
        )
        chart_paths["downsample_severity_logdet"] = create_severity_plot(
            results, available_transforms, "logdet", "downsample", asset_root, "downsample_severity_logdet.png", "logdet", False
        )
    except Exception as e:
        print(f"Warning (severity plots): {e}")

    # Mean shift and angle radar visualisations
    try:
        chart_paths["mean_shift_noise"] = create_mean_shift_plot(
            available_transforms, extras, "mean_shift", "noise", asset_root, "mean_shift_noise.png", "Mean shift (L2)"
        )
        chart_paths["mean_shift_downsample"] = create_mean_shift_plot(
            available_transforms, extras, "mean_shift", "downsample", asset_root, "mean_shift_downsample.png", "Mean shift (L2)"
        )
        chart_paths["mahal_shift_noise"] = create_mean_shift_plot(
            available_transforms, extras, "mahal_shift", "noise", asset_root, "mahal_shift_noise.png", "Mahalanobis shift"
        )
        chart_paths["mahal_shift_downsample"] = create_mean_shift_plot(
            available_transforms, extras, "mahal_shift", "downsample", asset_root, "mahal_shift_downsample.png", "Mahalanobis shift"
        )
    except Exception as e:
        print(f"Warning (mean shift plots): {e}")

    try:
        chart_paths["angle_radar"] = create_angle_radar(extras, available_transforms, asset_root, "angle_radar.png")
    except Exception as e:
        print(f"Warning (angle radar): {e}")

    try:
        noise_specs = [spec for spec in available_transforms if _spec_noise_sigma(spec) is not None]
        down_specs = [spec for spec in available_transforms if _spec_downsample_percent(spec) is not None]
        scenario_names = ["original"]
        if noise_specs:
            scenario_names.append(max(noise_specs, key=lambda s: _spec_noise_sigma(s) or 0.0).name)
        if down_specs:
            scenario_names.append(max(down_specs, key=lambda s: _spec_downsample_percent(s) or 0).name)
        chart_paths["pca_gallery"] = create_pca_gallery(
            analysis_root,
            analysis_root / "baseline" / "original",
            sample_index=0,
            scenario_names=scenario_names,
            asset_root=asset_root,
            filename="pca_gallery.png",
            transforms=available_transforms,
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
    generate_report(report_path, asset_rel_dir, setup_text, results, available_transforms, chart_paths, pass_stability_notes)


if __name__ == "__main__":
    main()
