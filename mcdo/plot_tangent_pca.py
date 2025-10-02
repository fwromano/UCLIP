"""Generate tangent-space PCA visualisations for MC Dropout embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

from .analysis import LABEL_COLOURS, load_labels, load_stats, select_indices
from .geometry_analysis import tangent_covariance


def _project_cov(cov: torch.Tensor, components: np.ndarray) -> np.ndarray:
    cov_np = cov.numpy().astype(np.float32)
    return components @ cov_np @ components.T


def _plot_subset(
    coords: np.ndarray,
    tangent_covs: Sequence[torch.Tensor],
    labels: Sequence[int],
    indices: Sequence[int],
    mask: Iterable[bool],
    title: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    mask = np.asarray(list(mask))
    subset_coords = coords[mask]
    subset_covs = [tangent_covs[i] for i, keep in enumerate(mask) if keep]
    subset_labels = [labels[i] for i, keep in enumerate(mask) if keep]
    subset_indices = [indices[i] for i, keep in enumerate(mask) if keep]

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    for point, cov, label, idx in zip(subset_coords, subset_covs, subset_labels, subset_indices):
        colour = LABEL_COLOURS.get(label, "#000000")

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        radii = 2.0 * np.sqrt(eigvals + 1e-12)
        ellipsoid = (np.stack([sphere_x, sphere_y, sphere_z], axis=-1) * radii) @ eigvecs.T
        ex, ey, ez = ellipsoid[..., 0] + point[0], ellipsoid[..., 1] + point[1], ellipsoid[..., 2] + point[2]

        ax.plot_surface(ex, ey, ez, color=colour, alpha=0.12, linewidth=0)
        ax.scatter(point[0], point[1], point[2], color=colour, s=10, alpha=0.8)
        ax.text(point[0], point[1], point[2], f"{idx}", fontsize=6, color=colour)

    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_plots(
    run_dir: Path,
    output_dir: Path,
    samples_per_label: int | None = None,
) -> None:
    metrics_csv = run_dir / "metrics.csv"
    labels_map = load_labels(metrics_csv)
    # use same ordering as in metrics
    all_indices = select_indices(labels_map, samples_per_label, None)
    labels = [labels_map[idx] for idx in all_indices]
    means, covariances = load_stats(run_dir, all_indices)

    means_list = [mu for mu in means]
    cov_list = [cov for cov in covariances]

    tangent_covs = [tangent_covariance(mu, cov) for mu, cov in zip(means_list, cov_list)]

    pca = PCA(n_components=3)
    coords = pca.fit_transform(torch.stack(means_list).numpy())
    components = pca.components_.astype(np.float32)

    projected_covs = [torch.tensor(_project_cov(cov, components)) for cov in tangent_covs]

    coords_all = coords
    _plot_subset(
        coords_all,
        projected_covs,
        labels,
        all_indices,
        mask=np.ones(len(all_indices), dtype=bool),
        title="Tangent PCA (all digits)",
        out_path=output_dir / "pca_tangent_all.png",
    )

    labels_arr = np.array(labels)
    mask_1_7 = np.isin(labels_arr, [1, 7])
    _plot_subset(
        coords_all,
        projected_covs,
        labels,
        all_indices,
        mask=mask_1_7,
        title="Tangent PCA (digits 1 vs 7)",
        out_path=output_dir / "pca_tangent_1_7.png",
    )

    mask_3_5 = np.isin(labels_arr, [3, 5])
    _plot_subset(
        coords_all,
        projected_covs,
        labels,
        all_indices,
        mask=mask_3_5,
        title="Tangent PCA (digits 3 vs 5)",
        out_path=output_dir / "pca_tangent_3_5.png",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tangent-space PCA views for MC Dropout embeddings")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--samples-per-label", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    generate_plots(args.run_dir, args.output_dir, args.samples_per_label)


if __name__ == "__main__":
    main()
