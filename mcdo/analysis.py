"""Analysis utilities for visualising Monte Carlo Dropout runs."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection registration
from sklearn.decomposition import PCA


LABEL_COLOURS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
    5: "#8c564b",
    6: "#e377c2",
    7: "#7f7f7f",
    8: "#bcbd22",
    9: "#17becf",
}


def load_labels(metrics_csv: Path) -> Dict[int, int]:
    labels: Dict[int, int] = {}
    with metrics_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels[int(row["index"])] = int(row["label"])
    return labels


def load_stats(run_dir: Path, indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    means: List[torch.Tensor] = []
    covariances: List[torch.Tensor] = []
    for idx in indices:
        sample_dir = run_dir / "individual" / f"{idx:05d}"
        mean = torch.load(sample_dir / "mu.pt", map_location="cpu")
        cov = torch.load(sample_dir / "Sigma.pt", map_location="cpu")
        means.append(mean)
        covariances.append(cov)
    return torch.stack(means, dim=0), torch.stack(covariances, dim=0)


def ellipse_from_covariance(center: np.ndarray, cov2d: np.ndarray, colour: str, n_std: float = 2.0) -> Ellipse:
    eigvals, eigvecs = np.linalg.eigh(cov2d)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(eigvals)
    return Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        edgecolor=colour,
        facecolor=colour,
        alpha=0.1,
        linewidth=0.8,
    )

def ellipsoid_mesh(center: np.ndarray, cov3d: np.ndarray, resolution: int = 15, n_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(cov3d)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    radii = n_std * np.sqrt(eigvals + 1e-12)

    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    sphere = np.stack([x, y, z], axis=-1)  # [..., 3]
    scaled = sphere * radii  # broadcasts radii across last dimension
    rotated = scaled @ eigvecs.T

    x_e = rotated[..., 0] + center[0]
    y_e = rotated[..., 1] + center[1]
    z_e = rotated[..., 2] + center[2]
    return x_e, y_e, z_e


def select_indices(
    labels_map: Dict[int, int],
    samples_per_label: int | None,
    allowed_labels: set[int] | None,
) -> List[int]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for index, label in labels_map.items():
        if allowed_labels is not None and label not in allowed_labels:
            continue
        grouped[label].append(index)

    if not grouped:
        raise ValueError("No samples matched the requested label filter.")

    if samples_per_label is None:
        selected = [idx for label in sorted(grouped) for idx in sorted(grouped[label])]
        return selected

    selected: List[int] = []
    for label in sorted(grouped):
        candidates = sorted(grouped[label])
        selected.extend(candidates[:samples_per_label])

    return sorted(selected)


def build_plot(
    means: torch.Tensor,
    covariances: torch.Tensor,
    labels: Sequence[int],
    output_path: Path,
    components: int,
    title: str | None,
) -> None:
    pca = PCA(n_components=components)
    transformed = pca.fit_transform(means.numpy())
    basis = pca.components_.astype(np.float32)

    if components == 2:
        fig, ax = plt.subplots(figsize=(10, 10))
        for coords, cov, label in zip(transformed, covariances, labels):
            colour = LABEL_COLOURS.get(label, "#000000")
            cov_np = cov.numpy().astype(np.float32)
            projected_cov = basis @ cov_np @ basis.T
            ellipse = ellipse_from_covariance(coords, projected_cov, colour)
            ax.add_patch(ellipse)
            ax.scatter(coords[0], coords[1], color=colour, s=15, alpha=0.6)

        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
    else:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        for coords, cov, label in zip(transformed, covariances, labels):
            colour = LABEL_COLOURS.get(label, "#000000")
            cov_np = cov.numpy().astype(np.float32)
            projected_cov = basis @ cov_np @ basis.T
            xs, ys, zs = ellipsoid_mesh(coords, projected_cov)
            ax.plot_surface(xs, ys, zs, color=colour, alpha=0.12, linewidth=0, shade=True)
            ax.scatter(coords[0], coords[1], coords[2], color=colour, s=20, alpha=0.8)

        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("PCA 3")
        ax.view_init(elev=25, azim=35)

    ax.set_title(title or "MC Dropout Mean & Variance PCA")

    if components == 2:
        handles = [
            ax.scatter([], [], color=colour, label=str(digit))
            for digit, colour in LABEL_COLOURS.items()
        ]
    else:
        handles = [
            ax.scatter([], [], [], color=colour, label=str(digit))
            for digit, colour in LABEL_COLOURS.items()
        ]
    ax.legend(handles=handles, title="Digit", loc="upper right", frameon=False, ncol=2)

    if components == 2:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def run(
    run_dir: Path,
    output_path: Path,
    components: int,
    samples_per_label: int | None,
    title: str | None,
    allowed_labels: set[int] | None,
) -> None:
    metrics_csv = run_dir / "metrics.csv"
    labels_map = load_labels(metrics_csv)
    indices = select_indices(labels_map, samples_per_label, allowed_labels)
    labels = [labels_map[idx] for idx in indices]
    means, covariances = load_stats(run_dir, indices)
    build_plot(means, covariances, labels, output_path, components=components, title=title)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PCA of MC Dropout means and variances.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing metrics.csv and individual stats.")
    parser.add_argument("output", type=Path, help="Path to the output image file.")
    parser.add_argument("--components", type=int, default=2, choices=(2, 3), help="Number of PCA components to plot.")
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=None,
        help="Number of samples to include for each label (default: all).",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma separated list of digit labels to include (default: all).",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional plot title override.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    allowed_labels = None
    if args.labels:
        try:
            allowed_labels = {int(token.strip()) for token in args.labels.split(",") if token.strip()}
        except ValueError as exc:
            raise ValueError("Labels must be integers.") from exc
    run(
        run_dir=args.run_dir,
        output_path=args.output,
        components=args.components,
        samples_per_label=args.samples_per_label,
        title=args.title,
        allowed_labels=allowed_labels,
    )


if __name__ == "__main__":
    main()
