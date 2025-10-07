"""Tangent-space uncertainty diagnostics for MC Dropout embeddings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch


def load_indices(metrics_csv: Path, limit: int | None = None) -> Sequence[int]:
    import csv

    indices: list[int] = []
    with metrics_csv.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            indices.append(int(row["index"]))
            if limit is not None and len(indices) >= limit:
                break
    return indices


def tangent_covariance(mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    mean = mean / (mean.norm() + 1e-12)
    dim = mean.numel()
    P = torch.eye(dim, device=mean.device, dtype=mean.dtype) - torch.outer(mean, mean)
    return P @ cov @ P


def circular_stats(embeddings: torch.Tensor) -> Dict[str, float]:
    norms = embeddings.norm(dim=1, keepdim=True).clamp_min(1e-12)
    unit = embeddings / norms
    mean_vec = unit.mean(dim=0)
    resultant = mean_vec.norm().item()
    return {
        "resultant_length": resultant,
        "circular_variance": 1.0 - resultant,
    }


def analyse_sample(sample_dir: Path) -> Dict[str, float]:
    mu = torch.load(sample_dir / "mu.pt")
    sigma = torch.load(sample_dir / "Sigma.pt")
    embeddings = torch.load(sample_dir / "embeddings.pt")

    ambient_trace = float(torch.trace(sigma).item())
    ambient_logdet = float(torch.logdet(sigma + 1e-6 * torch.eye(sigma.shape[0], device=sigma.device)).item())
    ambient_offdiag = float((sigma - torch.diag(torch.diag(sigma))).abs().sum().item())

    tangent = tangent_covariance(mu, sigma)
    tangent_trace = float(torch.trace(tangent).item())
    tangent_eigs = torch.linalg.eigvalsh(tangent).cpu().numpy()
    tangent_max = float(tangent_eigs.max())

    circ = circular_stats(embeddings)

    return {
        "ambient_trace": ambient_trace,
        "ambient_logdet": ambient_logdet,
        "ambient_offdiag": ambient_offdiag,
        "tangent_trace": tangent_trace,
        "tangent_lambda_max": tangent_max,
        "resultant_length": circ["resultant_length"],
        "circular_variance": circ["circular_variance"],
    }


def aggregate(records: Iterable[Dict[str, float]]) -> Dict[str, float]:
    collected: Dict[str, list[float]] = {}
    for record in records:
        for key, value in record.items():
            collected.setdefault(key, []).append(value)

    summary: Dict[str, float] = {}
    for key, values in collected.items():
        arr = np.asarray(values)
        summary[f"{key}_mean"] = float(arr.mean())
        summary[f"{key}_std"] = float(arr.std(ddof=0))
        summary[f"{key}_min"] = float(arr.min())
        summary[f"{key}_max"] = float(arr.max())
    return summary


def run(run_dir: Path, limit: int | None = None) -> Dict[str, object]:
    metrics_csv = run_dir / "metrics.csv"
    indices = load_indices(metrics_csv, limit)
    records = {}
    per_index: Dict[int, Dict[str, float]] = {}

    for idx in indices:
        sample_dir = run_dir / "individual" / f"{idx:05d}"
        if not sample_dir.exists():
            continue
        stats = analyse_sample(sample_dir)
        per_index[idx] = stats

    summary = aggregate(per_index.values())
    return {
        "indices": list(per_index.keys()),
        "per_index": per_index,
        "summary": summary,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geometry-aware MC Dropout diagnostics")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run(args.run_dir, args.limit)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
