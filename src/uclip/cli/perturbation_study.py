"""MNIST perturbation study combining noise and resolution degradations."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

from .mnist_driver import MNISTRoutineConfig, run as run_mnist
from ..core.utils import ensure_dir, save_json


MNIST_CANONICAL_SIZE = 28


@dataclass
class Scenario:
    name: str
    transform: Callable[[Image.Image], Image.Image]
    metadata: Dict[str, object]


def parse_float_list(payload: str) -> List[float]:
    if not payload:
        return []
    values: List[float] = []
    for chunk in payload.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    return values


def parse_int_list(payload: str) -> List[int]:
    if not payload:
        return []
    values: List[int] = []
    for chunk in payload.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values


def identity_transform(image: Image.Image) -> Image.Image:
    return image


def make_noise_transform(std: float) -> Callable[[Image.Image], Image.Image]:
    def transform(image: Image.Image) -> Image.Image:
        if std <= 0.0:
            return image
        array = np.asarray(image).astype(np.float32) / 255.0
        noise = np.random.normal(loc=0.0, scale=std, size=array.shape).astype(np.float32)
        perturbed = np.clip(array + noise, 0.0, 1.0)
        return Image.fromarray((perturbed * 255.0).astype(np.uint8))

    return transform


def make_downsample_transform(target_size: int) -> Callable[[Image.Image], Image.Image]:
    def transform(image: Image.Image) -> Image.Image:
        width, height = image.size
        if target_size >= min(width, height):
            return image
        downsampled = image.resize((target_size, target_size), Image.BILINEAR)
        return downsampled.resize((width, height), Image.BILINEAR)

    return transform


def build_scenarios(noise_stds: Iterable[float], downsample_sizes: Iterable[int]) -> List[Scenario]:
    scenarios: List[Scenario] = [
        Scenario(
            name="baseline",
            transform=identity_transform,
            metadata={"type": "baseline", "noise_std": None, "downsample_size": None},
        )
    ]

    for std in noise_stds:
        if std <= 0.0:
            continue
        scenarios.append(
            Scenario(
                name=f"noise_std_{std:.2f}".replace('.', 'p'),
                transform=make_noise_transform(std),
                metadata={
                    "type": "noise",
                    "noise_std": std,
                    "downsample_size": None,
                },
            )
        )

    for size in downsample_sizes:
        if size <= 0 or size >= MNIST_CANONICAL_SIZE:
            continue
        scenarios.append(
            Scenario(
                name=f"downsample_{size}",
                transform=make_downsample_transform(size),
                metadata={
                    "type": "downsample",
                    "noise_std": None,
                    "downsample_size": size,
                },
            )
        )

    return scenarios


def summarise_overall(summary: dict) -> Dict[str, Optional[float]]:
    label_entries = summary.get("label_summaries", {})
    total_count = sum(entry.get("count", 0) for entry in label_entries.values())

    def weighted_average(key: str) -> Optional[float]:
        numerator = 0.0
        denominator = 0
        for entry in label_entries.values():
            if key not in entry:
                continue
            count = entry.get("count", 0)
            numerator += count * float(entry[key])
            denominator += count
        if denominator == 0:
            return None
        return numerator / denominator

    metrics: Dict[str, Optional[float]] = {
        "count": float(total_count),
        "trace_mean": weighted_average("trace_mean"),
        "logdet_mean": weighted_average("logdet_mean"),
        "off_diag_mean": weighted_average("off_diag_mean"),
    }

    if label_entries and any("accuracy" in entry for entry in label_entries.values()):
        metrics["accuracy_mean"] = weighted_average("accuracy")
        metrics["entropy_mean"] = weighted_average("entropy_mean")
        metrics["mi_mean"] = weighted_average("mi_mean")
        metrics["confidence_mean"] = weighted_average("confidence_mean")
    else:
        metrics["accuracy_mean"] = None
        metrics["entropy_mean"] = None
        metrics["mi_mean"] = None
        metrics["confidence_mean"] = None

    if "overall_accuracy" in summary:
        metrics["overall_accuracy"] = float(summary["overall_accuracy"])
    else:
        metrics["overall_accuracy"] = None

    return metrics


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run noise and resolution perturbation study on MNIST")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--passes", type=int, default=32)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--adapter-target", action="append", default=[])
    parser.add_argument("--adapter-drop", type=float, default=0.1)
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--no-predictive", action="store_true")
    parser.add_argument("--out-root", type=str, default="runs/mnist_perturbations")
    parser.add_argument("--noise-stds", type=str, default="0.0,0.1,0.2,0.3")
    parser.add_argument("--downsample-sizes", type=str, default="28,20,14,10")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)

    out_root = ensure_dir(Path(args.out_root))
    noise_levels = parse_float_list(args.noise_stds)
    downsample_sizes = parse_int_list(args.downsample_sizes)
    scenarios = build_scenarios(noise_levels, downsample_sizes)

    aggregated_rows = []

    for scenario in scenarios:
        scenario_out = ensure_dir(out_root / scenario.name)

        config = MNISTRoutineConfig(
            model_id=args.model,
            device=args.device,
            root=Path(args.root),
            train=args.train,
            limit=None if args.limit <= 0 else args.limit,
            passes=args.passes,
            microbatch=args.microbatch,
            tau=args.tau,
            seed=args.seed,
            allow_tf32=not args.disable_tf32,
            dropout_rate=args.dropout_rate,
            adapter_targets=args.adapter_target,
            adapter_p=args.adapter_drop,
            out_dir=scenario_out,
            save_raw=args.save_raw,
            no_predictive=args.no_predictive,
            image_transform=scenario.transform,
        )

        summary = run_mnist(config)

        metrics = summarise_overall(summary)
        row = {
            "scenario": scenario.name,
            "type": scenario.metadata.get("type"),
            "noise_std": scenario.metadata.get("noise_std"),
            "downsample_size": scenario.metadata.get("downsample_size"),
            "overall_accuracy": metrics.get("overall_accuracy"),
            "accuracy_mean": metrics.get("accuracy_mean"),
            "trace_mean": metrics.get("trace_mean"),
            "logdet_mean": metrics.get("logdet_mean"),
            "off_diag_mean": metrics.get("off_diag_mean"),
            "entropy_mean": metrics.get("entropy_mean"),
            "mi_mean": metrics.get("mi_mean"),
            "confidence_mean": metrics.get("confidence_mean"),
            "count": metrics.get("count"),
        }
        aggregated_rows.append(row)

        save_json(scenario_out / "perturbation_summary.json", {
            "scenario": scenario.metadata,
            "mnist_summary": summary,
            "overall_metrics": metrics,
        })

    csv_path = out_root / "aggregated_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = [
            "scenario",
            "type",
            "noise_std",
            "downsample_size",
            "count",
            "overall_accuracy",
            "accuracy_mean",
            "trace_mean",
            "logdet_mean",
            "off_diag_mean",
            "entropy_mean",
            "mi_mean",
            "confidence_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated_rows:
            writer.writerow(row)

    save_json(out_root / "aggregated_metrics.json", {"runs": aggregated_rows})


if __name__ == "__main__":
    main()
