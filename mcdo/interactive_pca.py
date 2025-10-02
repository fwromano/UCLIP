"""Generate an interactive 3D PCA visualisation of MC Dropout embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Set, Optional

import numpy as np
import plotly.graph_objects as go
from plotly import offline as pyo
from sklearn.decomposition import PCA
import torch

from .analysis import LABEL_COLOURS, load_labels, load_stats, select_indices
from .geometry_analysis import tangent_covariance

try:  # torchvision is optional at runtime
    from torchvision.datasets import MNIST
except ImportError:  # pragma: no cover
    MNIST = None  # type: ignore

import base64
import io
from PIL import Image


def _load_thumbnails(indices, root: Path, size: int = 64):
    if MNIST is None:
        raise RuntimeError('torchvision is required to embed MNIST thumbnails.')

    dataset = MNIST(root=str(root), train=False, download=True)
    thumbnails = []
    for idx in indices:
        image, _ = dataset[idx]
        if hasattr(image, 'resize'):
            pil_image = image
        else:
            pil_image = Image.fromarray(np.array(image))
        resized = pil_image.resize((size, size))
        buffer = io.BytesIO()
        resized.save(buffer, format='PNG')
        encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
        thumbnails.append(encoded)
    return thumbnails


def ellipsoid_mesh(
    center: np.ndarray,
    cov: np.ndarray,
    resolution: int = 20,
    n_std: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(cov)
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

    sphere = np.stack([x, y, z], axis=-1)
    scaled = sphere * radii
    rotated = scaled @ eigvecs.T

    xs = rotated[..., 0] + center[0]
    ys = rotated[..., 1] + center[1]
    zs = rotated[..., 2] + center[2]
    return xs, ys, zs


def build_interactive_plot(
    means: torch.Tensor,
    covariances: torch.Tensor,
    labels: Sequence[int],
    indices: Sequence[int],
    output: Path,
    title: str | None,
    auto_open: bool,
    image_data: Optional[Sequence[Optional[str]]] = None,
    mode_label: str = "Ambient",
) -> None:
    pca = PCA(n_components=3)
    transformed = pca.fit_transform(means.numpy())
    basis = pca.components_.astype(np.float32)

    fig = go.Figure()
    legend_seen: dict[int, bool] = {}

    for i, (coords, cov, label, idx) in enumerate(zip(transformed, covariances, labels, indices)):
        cov_np = cov.numpy().astype(np.float32)
        projected_cov = basis @ cov_np @ basis.T
        xs, ys, zs = ellipsoid_mesh(coords, projected_cov)

        trace_val = float(np.trace(cov_np))
        offdiag_val = float(np.abs(cov_np - np.diag(np.diag(cov_np))).sum())

        colour = LABEL_COLOURS.get(label, "#000000")
        hover = (
            f"index: {idx}<br>digit: {label}<br>trace: {trace_val:.2f}<br>"
            f"off-diagonal mass: {offdiag_val:.2f}"
        )

        img_html = ""
        if image_data is not None and i < len(image_data) and image_data[i]:
            img_html = (
                "<br><img src='data:image/png;base64,"
                + image_data[i]
                + "' width='64' height='64'>"
            )

        fig.add_trace(
            go.Mesh3d(
                x=xs.ravel(),
                y=ys.ravel(),
                z=zs.ravel(),
                alphahull=0,
                opacity=0.15,
                color=colour,
                hoverinfo="skip",
                showscale=False,
                legendgroup=str(label),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[coords[0]],
                y=[coords[1]],
                z=[coords[2]],
                mode="markers+text",
                marker=dict(color=colour, size=6),
                text=[str(idx)],
                textposition="top center",
                hovertext=hover + img_html,
                hoverinfo="text",
                legendgroup=str(label),
                name=f"digit {label}",
                showlegend=not legend_seen.get(label, False),
            )
        )

        legend_seen[label] = True

    fig.update_layout(
        title=title or f"MC Dropout Embeddings ({mode_label})",
        scene=dict(
            xaxis_title="PCA 1",
            yaxis_title="PCA 2",
            zaxis_title="PCA 3",
        ),
        legend=dict(itemsizing="constant"),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    if auto_open:
        pyo.plot(fig, filename=str(output), auto_open=True, include_plotlyjs="cdn")
    else:
        fig.write_html(output, include_plotlyjs="cdn")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive PCA plot for MC Dropout embeddings.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing metrics.csv and individual stats.")
    parser.add_argument("output", type=Path, help="Path to the interactive HTML file to write.")
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
    parser.add_argument("--open", action="store_true", help="Open the plot in a browser after generation.")
    parser.add_argument("--show-images", action="store_true", help="Embed MNIST thumbnails in hover tooltips.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data"), help="Path to MNIST root (used with --show-images).")
    parser.add_argument("--tangent", action="store_true", help="Use tangent-plane covariance instead of ambient covariance.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    allowed_labels: Set[int] | None = None
    if args.labels:
        try:
            allowed_labels = {int(token.strip()) for token in args.labels.split(",") if token.strip()}
        except ValueError as exc:  # pragma: no cover - CLI guard
            raise ValueError("Labels must be integers separated by commas.") from exc

    labels_map = load_labels(args.run_dir / "metrics.csv")
    indices = select_indices(labels_map, args.samples_per_label, allowed_labels)
    labels = [labels_map[idx] for idx in indices]

    means_tensor, cov_tensor = load_stats(args.run_dir, indices)
    means_list = [mu for mu in means_tensor]
    cov_list = [cov for cov in cov_tensor]

    mode_label = "Ambient"
    if args.tangent:
        cov_list = [tangent_covariance(mu, cov) for mu, cov in zip(means_list, cov_list)]
        mode_label = "Tangent"

    means = torch.stack(means_list)
    covariances = torch.stack(cov_list)

    images = None
    if args.show_images:
        images = _load_thumbnails(indices, args.dataset_root)

    build_interactive_plot(
        means,
        covariances,
        labels,
        indices,
        args.output,
        args.title,
        auto_open=args.open,
        image_data=images,
        mode_label=mode_label,
    )


if __name__ == "__main__":
    main()
