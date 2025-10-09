#!/usr/bin/env python
"""Project CLIP embeddings with Monte Carlo Dropout onto a 2D PCA plane."""
from __future__ import annotations

import argparse
import re
import sys
import webbrowser
from pathlib import Path
from typing import Dict, List, Sequence

# Allow running without installing the package by injecting repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

for path in (SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch
from PIL import Image

from uclip.core.dropout import dump_dropout_rates, insert_adapters, override_dropout_rate
from uclip.core.sampling import compute_embedding_statistics, sample_embeddings
from uclip.core.utils import load_clip_backbone, set_determinism


COLOR_HEX = {
    "blue": "#1f77b4",
    "green": "#2ca02c",
    "indigo": "#4b0082",
    "orange": "#ff7f0e",
    "red": "#d62728",
    "violet": "#9400d3",
    "purple": "#9467bd",
    "yellow": "#ffd60a",
    "cyan": "#17becf",
    "magenta": "#e377c2",
    "pink": "#e377c2",
    "teal": "#17becf",
    "black": "#222222",
    "white": "#f0f0f0",
    "gray": "#7f7f7f",
    "wolf": "#7f7f7f",
    "moose": "#8c564b",
    "chrome": "#aaaaaa",
}

FALLBACK_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

IMAGE_EXTENSIONS: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/notebook"),
        help="Folder containing RGB images to analyse (recursively scanned).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/clip_mcdo_pca.png"),
        help="Path to save the PCA figure.",
    )
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch32", help="CLIP model checkpoint.")
    parser.add_argument("--device", default=None, help="Force a specific device (e.g. cuda, cpu).")
    parser.add_argument("--passes", type=int, default=32, help="Number of Monte Carlo Dropout forward passes.")
    parser.add_argument("--microbatch", type=int, default=4, help="Microbatch size for stochastic sampling.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for determinism.")
    parser.add_argument(
        "--adapter-target",
        action="append",
        default=[],
        help="Dotted path of a module to wrap with DropoutAdapter (repeatable).",
    )
    parser.add_argument(
        "--adapter-drop",
        type=float,
        default=0.01,
        help="Dropout probability for inserted DropoutAdapters.",
    )
    parser.add_argument(
        "--override-dropout-rate",
        type=float,
        default=None,
        help="Override existing dropout layers with a new probability.",
    )
    parser.add_argument(
        "--mcdo-p",
        type=float,
        default=0.01,
        help="Dropout probability to use for Monte Carlo Dropout sampling when no override is supplied.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most this many images (0 means all).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution (dots per inch) for the saved figure.",
    )
    parser.add_argument(
        "--title",
        default="CLIP embedding geometry under Monte Carlo Dropout",
        help="Title for the PCA plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively in addition to saving it.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path("notebooks/clip_mcdo_pca.html"),
        help="Path to save an interactive Plotly visualisation (set to '' to skip).",
    )
    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Skip saving the static PNG figure (interactive HTML only).",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not automatically open the interactive HTML in a browser.",
    )
    parser.add_argument(
        "--subset-output-dir",
        type=Path,
        default=Path("notebooks/clip_mcdo_pca_subsets"),
        help="Directory for subgroup PCA plots (empty string to disable).",
    )
    return parser.parse_args(argv)


def discover_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Data directory {root} does not exist.")
    images = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise RuntimeError(f"No supported image files found under {root}.")
    return sorted(images)


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGB")


def fit_pca(matrix: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if matrix.ndim != 2:
        raise ValueError("Expected matrix with shape [samples, dim].")
    if matrix.shape[0] < n_components:
        raise ValueError(f"Need at least {n_components} samples to compute PCA.")
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    explained_variance = (singular_values[:n_components] ** 2) / max(1, matrix.shape[0] - 1)
    total_variance = (singular_values**2).sum() / max(1, matrix.shape[0] - 1)
    if total_variance <= 0:
        explained_ratio = np.zeros_like(explained_variance)
    else:
        explained_ratio = explained_variance / total_variance
    return mean, components, explained_variance, explained_ratio


def compute_projected_covariance(samples: np.ndarray) -> np.ndarray:
    if samples.shape[0] <= 1:
        return np.zeros((samples.shape[1], samples.shape[1]), dtype=samples.dtype)
    return np.cov(samples, rowvar=False)


def build_plot(
    records: Sequence[dict],
    explained_ratio: np.ndarray,
    title: str,
    dpi: int,
    output_path: Path,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse, Patch

    fig, ax = plt.subplots(figsize=(8, 6))

    def plot_covariance_ellipse(mean: np.ndarray, cov: np.ndarray, color: str, n_std: float = 1.0, alpha: float = 0.2) -> None:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        width = 2.0 * n_std * np.sqrt(eigvals[0])
        height = 2.0 * n_std * np.sqrt(eigvals[1])
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
        )
        ax.add_patch(ellipse)

    for record in records:
        color = record["color"]
        samples = record["proj_samples_2d"]
        det_point = record["proj_deterministic_2d"]
        mean_point = record["proj_mean_2d"]
        cov = record["proj_cov_2d"]

        ax.scatter(
            samples[:, 0],
            samples[:, 1],
            color=color,
            marker="x",
            alpha=0.45,
            linewidths=1.0,
            s=40,
        )
        ax.scatter(
            det_point[0],
            det_point[1],
            color=color,
            marker="+",
            linewidths=2.5,
            s=120,
            zorder=3,
        )
        ax.scatter(
            mean_point[0],
            mean_point[1],
            color=color,
            marker="x",
            linewidths=3.0,
            s=200,
            zorder=4,
        )
        plot_covariance_ellipse(mean_point, cov, color=color, alpha=0.18)
        ax.text(
            mean_point[0],
            mean_point[1],
            record["label"],
            color=color,
            fontsize=9,
            ha="center",
            va="center",
            zorder=5,
        )

    legend_handles = [
        Line2D([0], [0], marker="x", color="gray", linestyle="", markerfacecolor="none", markeredgewidth=1.5, markersize=6, label="MCDO samples"),
        Line2D([0], [0], marker="+", color="gray", linestyle="", markeredgewidth=2.0, markersize=8, label="Deterministic CLIP"),
        Line2D([0], [0], marker="x", color="gray", linestyle="", markerfacecolor="none", markeredgewidth=3.0, markersize=10, label="MCDO mean"),
        Patch(facecolor="gray", edgecolor="gray", alpha=0.18, label="1σ ellipse"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}% var)")
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)


def ellipse_points(mean: np.ndarray, cov: np.ndarray, n_std: float = 1.0, num: int = 200) -> np.ndarray:
    if cov.shape != (2, 2):
        raise ValueError("Covariance must be 2x2.")
    if not np.any(cov):
        return np.tile(mean, (num, 1))
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    theta = np.linspace(0.0, 2.0 * np.pi, num)
    circle = np.stack((np.cos(theta), np.sin(theta)), axis=0)
    radii = n_std * np.sqrt(eigvals)
    transform = eigvecs @ np.diag(radii)
    points = (transform @ circle).T + mean
    return points


def ellipsoid_mesh(
    mean: np.ndarray,
    cov: np.ndarray,
    n_std: float = 1.0,
    resolution: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cov.shape != (3, 3):
        raise ValueError("Covariance must be 3x3.")
    if not np.any(cov):
        x = np.full((resolution, resolution), mean[0])
        y = np.full((resolution, resolution), mean[1])
        z = np.full((resolution, resolution), mean[2])
        return x, y, z
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    radii = n_std * np.sqrt(eigvals)
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    base = np.stack((x, y, z), axis=-1)
    rotated = base @ eigvecs.T
    x_r = rotated[..., 0] + mean[0]
    y_r = rotated[..., 1] + mean[1]
    z_r = rotated[..., 2] + mean[2]
    return x_r, y_r, z_r


def slugify(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return token or "group"


def build_subset_plot(
    subset_records: Sequence[dict],
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    projected_records = []
    combined = np.concatenate([record["mc_samples"] for record in subset_records], axis=0)
    if combined.shape[0] < 2:
        print(f"Skipping subset '{title}' (needs at least two stochastic samples).")
        return
    try:
        subset_mean, subset_components, _, subset_ratio = fit_pca(combined, n_components=2)
    except ValueError as exc:
        print(f"Skipping subset '{title}' (PCA failed: {exc})")
        return

    for record in subset_records:
        centered_samples = record["mc_samples"] - subset_mean
        projected_samples = centered_samples @ subset_components.T
        projected_mean = (record["mc_mean"] - subset_mean) @ subset_components.T
        projected_det = (record["deterministic"] - subset_mean) @ subset_components.T
        projected_records.append(
            {
                "label": record["label"],
                "color": record["color"],
                "proj_samples_2d": projected_samples,
                "proj_mean_2d": projected_mean,
                "proj_deterministic_2d": projected_det,
                "proj_cov_2d": compute_projected_covariance(projected_samples),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_plot(
        records=projected_records,
        explained_ratio=subset_ratio,
        title=title,
        dpi=dpi,
        output_path=output_path,
        show=False,
    )
    print(f"Saved subset PCA to {output_path}")


def generate_subset_plots(
    records: Sequence[dict],
    output_dir: Path,
    dpi: int,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_category: Dict[str, List[dict]] = {}
    by_angle: Dict[str, List[dict]] = {}

    for record in records:
        category = record.get("category", "unknown")
        by_category.setdefault(category, []).append(record)
        angle = record.get("angle")
        if angle and angle.lower() not in {"", "na"}:
            by_angle.setdefault(angle, []).append(record)

    for category, subset in by_category.items():
        if not subset:
            continue
        title = f"PCA — category {category}"
        path = output_dir / f"category_{slugify(category)}.png"
        build_subset_plot(subset, title, path, dpi=dpi)

    for angle, subset in by_angle.items():
        if not subset:
            continue
        title = f"PCA — angle {angle}"
        path = output_dir / f"angle_{slugify(angle)}.png"
        build_subset_plot(subset, title, path, dpi=dpi)


def build_interactive_plot(
    records: Sequence[dict],
    explained_ratio: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Plotly is required for interactive output. Install via `pip install plotly`.") from exc

    fig = go.Figure()

    for record in records:
        color = record["color"]
        samples = record["proj_samples_3d"]
        det_point = record["proj_deterministic_3d"]
        mean_point = record["proj_mean_3d"]
        cov = record["proj_cov_3d"]

        legend_group = record["label"]
        fig.add_trace(
            go.Scatter3d(
                x=samples[:, 0],
                y=samples[:, 1],
                z=samples[:, 2],
                mode="markers",
                name=f"{record['label']} samples",
                marker=dict(color=color, symbol="circle", size=4, opacity=0.4),
                hovertext=[f"{record['label']} sample {i}" for i in range(samples.shape[0])],
                legendgroup=legend_group,
                showlegend=True,
            )
        )

        try:
            x_mesh, y_mesh, z_mesh = ellipsoid_mesh(mean_point, cov)
            fig.add_trace(
                go.Surface(
                    x=x_mesh,
                    y=y_mesh,
                    z=z_mesh,
                    showscale=False,
                    opacity=0.18,
                    hoverinfo="skip",
                    surfacecolor=np.ones_like(x_mesh),
                    colorscale=[[0, color], [1, color]],
                    legendgroup=legend_group,
                    showlegend=False,
                )
            )
        except ValueError:
            pass

        fig.add_trace(
            go.Scatter3d(
                x=[mean_point[0]],
                y=[mean_point[1]],
                z=[mean_point[2]],
                mode="markers+text",
                name=f"{record['label']} mean",
                marker=dict(
                    color=color,
                    symbol="x",
                    size=10,
                    opacity=0.95,
                    line=dict(width=2, color="#000000"),
                ),
                text=[record["label"]],
                textposition="top center",
                hovertext=[f"{record['label']} mean"],
                legendgroup=legend_group,
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[det_point[0]],
                y=[det_point[1]],
                z=[det_point[2]],
                mode="markers",
                name=f"{record['label']} deterministic",
                marker=dict(
                    color=color,
                    symbol="circle",
                    size=10,
                    opacity=0.95,
                    line=dict(width=2, color="#ffffff"),
                ),
                hovertext=[f"{record['label']} deterministic"],
                legendgroup=legend_group,
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"PC1 ({explained_ratio[0] * 100:.1f}% var)",
            yaxis_title=f"PC2 ({explained_ratio[1] * 100:.1f}% var)",
            zaxis_title=f"PC3 ({explained_ratio[2] * 100:.1f}% var)",
            aspectmode="data",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))


def summarise_dropout_probability(
    effective_rate: float | None,
    dropout_rates: Sequence[float],
) -> str:
    if effective_rate is not None:
        return f"{effective_rate:.3f}"
    if dropout_rates:
        unique = sorted({round(float(rate), 6) for rate in dropout_rates})
        if len(unique) == 1:
            return f"{unique[0]:.3f}"
        return f"{unique[0]:.3f}–{unique[-1]:.3f}"
    return "0.000"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    images = discover_images(args.data_dir)
    if args.limit > 0:
        images = images[: args.limit]

    print(f"Discovered {len(images)} images from {args.data_dir}")
    for path in images:
        print(f" - {path.relative_to(args.data_dir)}")

    set_determinism(args.seed)

    loaded = load_clip_backbone(args.model_id, device=args.device)
    model, processor, device = loaded.model, loaded.processor, loaded.device
    model.eval()

    adapter_targets = list(args.adapter_target) if args.adapter_target else ["visual_projection"]
    auto_inserted_adapter = not args.adapter_target
    insert_adapters(model, adapter_targets, p=args.adapter_drop)
    if auto_inserted_adapter:
        print(f"Inserted DropoutAdapter on visual_projection (p={args.adapter_drop}) to enable MC dropout.")
    dropout_override = args.override_dropout_rate if args.override_dropout_rate is not None else args.mcdo_p
    if dropout_override is not None:
        override_dropout_rate(model, dropout_override)

    dropout_count, dropout_rates = dump_dropout_rates(model)
    effective_dropout = dropout_override
    print(f"Using device: {device}")
    print(f"Detected {dropout_count} dropout layers")
    if dropout_rates:
        print(f"Dropout rates: {dropout_rates}")

    records: List[dict] = []

    for path in images:
        image = load_rgb_image(path)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            deterministic = model.get_image_features(**inputs).squeeze(0)

        samples = sample_embeddings(
            model=model,
            forward_fn=model.get_image_features,
            forward_kwargs=inputs,
            passes=args.passes,
            microbatch=args.microbatch,
        )
        stats = compute_embedding_statistics(samples)

        # Ensure the backbone returns to evaluation mode after dropout sampling.
        model.eval()

        label = path.stem
        relative_parts = path.relative_to(args.data_dir).parts
        if len(relative_parts) > 1:
            category = relative_parts[0]
        else:
            category = label.split("_", 1)[0]
        angle = label.split("_")[-1] if "_" in label else "NA"

        records.append(
            {
                "path": path,
                "label": label,
                "category": category,
                "angle": angle,
                "deterministic": deterministic.detach().cpu().numpy(),
                "mc_samples": stats.embeddings.detach().cpu().numpy(),
                "mc_mean": stats.mean.detach().cpu().numpy(),
            }
        )

    stacked = np.concatenate([record["mc_samples"] for record in records], axis=0)
    if stacked.shape[0] < 3:
        raise RuntimeError(
            "Need at least three samples to compute a 3D PCA projection. Increase --passes or provide more images."
        )
    mean, components, _, explained_ratio = fit_pca(stacked, n_components=3)
    components_2d = components[:2]
    components_3d = components[:3]
    explained_ratio_2d = explained_ratio[:2]
    explained_ratio_3d = explained_ratio[:3]

    for idx, record in enumerate(records):
        centered_samples = record["mc_samples"] - mean
        centered_mean = record["mc_mean"] - mean
        centered_det = record["deterministic"] - mean

        record["proj_samples_2d"] = centered_samples @ components_2d.T
        record["proj_mean_2d"] = centered_mean @ components_2d.T
        record["proj_deterministic_2d"] = centered_det @ components_2d.T
        record["proj_cov_2d"] = compute_projected_covariance(record["proj_samples_2d"])

        record["proj_samples_3d"] = centered_samples @ components_3d.T
        record["proj_mean_3d"] = centered_mean @ components_3d.T
        record["proj_deterministic_3d"] = centered_det @ components_3d.T
        record["proj_cov_3d"] = compute_projected_covariance(record["proj_samples_3d"])

        base_colour = record["label"].split("_", 1)[0].lower()
        base_colour = base_colour.rstrip("0123456789")
        if not base_colour:
            base_colour = record["label"].split("_", 1)[0].lower()
        record["color"] = COLOR_HEX.get(base_colour, FALLBACK_PALETTE[idx % len(FALLBACK_PALETTE)])

    print(f"Explained variance ratios (first three PCs): {explained_ratio_3d}")

    dropout_summary = summarise_dropout_probability(
        effective_rate=effective_dropout,
        dropout_rates=dropout_rates,
    )
    full_title = f"{args.title} (p={dropout_summary}, passes={args.passes}, images={len(records)})"
    print(f"MC dropout rate (p): {dropout_summary}")

    if not args.skip_static:
        print(f"Saving static figure to {args.output}")
        build_plot(
            records=records,
            explained_ratio=explained_ratio_2d,
            title=full_title,
            dpi=args.dpi,
            output_path=args.output,
            show=args.show,
        )

    if args.html_output and str(args.html_output).strip():
        print(f"Saving interactive figure to {args.html_output}")
        build_interactive_plot(
            records=records,
            explained_ratio=explained_ratio_3d,
            title=full_title,
            output_path=args.html_output,
        )
        if not args.no_open:
            try:
                webbrowser.open(args.html_output.resolve().as_uri(), new=2)
            except Exception as exc:  # pragma: no cover - best-effort IE environment
                print(f"Warning: failed to open browser automatically: {exc}")

    if args.subset_output_dir and str(args.subset_output_dir).strip():
        print(f"Saving subset PCA visuals to {args.subset_output_dir}")
        generate_subset_plots(
            records=records,
            output_dir=args.subset_output_dir,
            dpi=args.dpi,
        )

    print("Done.")


if __name__ == "__main__":
    main()
