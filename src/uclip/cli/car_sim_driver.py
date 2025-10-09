"""Run Monte Carlo Dropout sampling across the car_sim dataset."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from ..core.dropout import dump_dropout_rates, insert_adapters, override_dropout_rate
from ..core.sampling import (
    compute_embedding_statistics,
    compute_predictive_distribution,
    diagnostics,
    sample_embeddings,
)
from ..core.utils import (
    ensure_dir,
    load_clip_backbone,
    save_json,
    save_numpy,
    save_tensor,
    set_determinism,
)
from .mnist_driver import compute_text_embeddings, ensure_rgb


def normalise_label(label: str) -> str:
    """Best-effort conversion of folder names to human-friendly phrases."""
    tokens = label.replace("-", " ").replace("_", " ").split()
    filtered = [token for token in tokens if token.lower() not in {"jeep", "cropped"}]
    if not filtered:
        filtered = tokens
    return " ".join(filtered).strip()


@dataclass
class CarSimRoutineConfig:
    model_id: str
    device: Optional[str]
    root: Path
    train: bool
    limit: Optional[int]
    passes: int
    microbatch: int
    tau: float
    seed: int
    allow_tf32: bool
    dropout_rate: Optional[float]
    adapter_targets: Sequence[str]
    adapter_p: float
    out_dir: Path
    save_raw: bool
    no_predictive: bool
    image_transform: Optional[Callable[[Image.Image], Image.Image]] = None
    prompts: Optional[Sequence[str]] = None
    prompt_template: str = "a photo of a {label} jeep"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCDO uncertainty on car_sim with CLIP")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--passes", "-T", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--adapter-target", action="append", default=[])
    parser.add_argument("--adapter-drop", type=float, default=0.1)
    parser.add_argument("--root", type=str, default="data/car_sim")
    parser.add_argument("--train", action="store_true", help="Use the training split if present (default: test/full)")
    parser.add_argument("--limit", type=int, default=256, help="Number of samples to process (None for full split)")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--save-raw", action="store_true", help="Persist per-sample tensors under out/individual/")
    parser.add_argument("--no-predictive", action="store_true", help="Skip predictive head computations; report only embedding statistics")
    parser.add_argument("--prompt", action="append", default=[], help="Custom prompts (one per class, overrides template)")
    parser.add_argument("--prompt-template", type=str, default="a photo of a {label} jeep", help="Template used to convert class names to prompts")
    return parser.parse_args(argv)


def resolve_dataset_root(root: Path, train: bool) -> Path:
    train_dir = root / "train"
    test_dir = root / "test"
    if train and train_dir.is_dir():
        return train_dir
    if not train and test_dir.is_dir():
        return test_dir
    return root


def load_dataset(root: Path, train: bool) -> ImageFolder:
    dataset_root = resolve_dataset_root(root, train=train)
    dataset = ImageFolder(root=str(dataset_root))
    return dataset


def build_prompts(class_names: Sequence[str], template: str) -> Sequence[str]:
    prompts = []
    for name in class_names:
        friendly = normalise_label(name)
        prompt = template.format(label=friendly, raw_label=name)
        prompts.append(prompt)
    return prompts


def run(config: CarSimRoutineConfig) -> dict:
    set_determinism(seed=config.seed, allow_tf32=config.allow_tf32)

    loaded = load_clip_backbone(config.model_id, device=config.device)
    model, processor, device = loaded.model, loaded.processor, loaded.device

    if config.adapter_targets:
        insert_adapters(model, config.adapter_targets, p=config.adapter_p)
    if config.dropout_rate is not None:
        override_dropout_rate(model, config.dropout_rate)

    dropout_count, dropout_rates = dump_dropout_rates(model)

    dataset = load_dataset(config.root, config.train)
    class_names = list(dataset.classes)

    if config.prompts:
        if len(config.prompts) != len(class_names):
            raise ValueError(
                f"Expected {len(class_names)} prompts, received {len(config.prompts)}"
            )
        prompts = list(config.prompts)
    else:
        prompts = list(build_prompts(class_names, config.prompt_template))

    text_embeddings = None
    if not config.no_predictive:
        text_embeddings = compute_text_embeddings(model, processor, prompts, device).to(device)

    out_dir = ensure_dir(config.out_dir)
    per_item_dir = ensure_dir(out_dir / "individual") if config.save_raw else None
    csv_path = out_dir / "metrics.csv"

    if config.no_predictive:
        fields = [
            "index",
            "label",
            "label_name",
            "trace",
            "logdet",
            "off_diag_mass",
        ]
    else:
        fields = [
            "index",
            "label",
            "label_name",
            "predicted",
            "predicted_name",
            "correct",
            "trace",
            "logdet",
            "off_diag_mass",
            "entropy",
            "mutual_information",
            "confidence",
        ]

    summary = defaultdict(list)

    limit = config.limit if config.limit and config.limit > 0 else len(dataset)

    with csv_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

        for idx in range(min(limit, len(dataset))):
            image, label = dataset[idx]
            image = ensure_rgb(image)
            if config.image_transform is not None:
                image = config.image_transform(image)
            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            embeddings = sample_embeddings(
                model=model,
                forward_fn=model.get_image_features,
                forward_kwargs=inputs,
                passes=config.passes,
                microbatch=config.microbatch,
            )
            stats = compute_embedding_statistics(embeddings)
            diag = diagnostics(stats.covariance)
            predictive = None
            if text_embeddings is not None:
                predictive = compute_predictive_distribution(stats.embeddings, text_embeddings, tau=config.tau)

            label_name = class_names[label]
            row = {
                "index": idx,
                "label": int(label),
                "label_name": label_name,
                "trace": float(diag.trace),
                "logdet": float(diag.logdet),
                "off_diag_mass": float(diag.off_diag_mass),
            }
            if predictive is not None:
                probs = predictive["predictive_mean"].detach()
                predicted_idx = int(torch.argmax(probs).item())
                confidence = float(probs.max().item())
                row.update(
                    {
                        "predicted": predicted_idx,
                        "predicted_name": class_names[predicted_idx],
                        "correct": bool(predicted_idx == label),
                        "entropy": float(predictive["mean_entropy"].item()),
                        "mutual_information": float(predictive["mutual_information"].item()),
                        "confidence": confidence,
                    }
                )
            writer.writerow(row)

            summary[label].append(row)

            if per_item_dir is not None:
                sample_dir = ensure_dir(per_item_dir / f"{idx:05d}")
                save_tensor(sample_dir / "mu.pt", stats.mean)
                save_tensor(sample_dir / "Sigma.pt", stats.covariance)
                save_tensor(sample_dir / "embeddings.pt", stats.embeddings)
                if predictive is not None:
                    probs = predictive["predictive_mean"].detach()
                    save_numpy(sample_dir / "probs_mean.npy", probs.cpu().numpy())
                    save_numpy(
                        sample_dir / "entropy_mean.npy",
                        predictive["mean_entropy"].detach().cpu().item(),
                    )
                    save_numpy(
                        sample_dir / "MI.npy",
                        predictive["mutual_information"].detach().cpu().item(),
                    )

    label_summaries = {}
    for label, rows in summary.items():
        if not rows:
            continue
        count = len(rows)
        entry = {
            "name": class_names[int(label)],
            "count": count,
            "trace_mean": sum(row["trace"] for row in rows) / count,
            "logdet_mean": sum(row["logdet"] for row in rows) / count,
            "off_diag_mean": sum(row["off_diag_mass"] for row in rows) / count,
        }
        if not config.no_predictive:
            correct_count = sum(1 for row in rows if row.get("correct"))
            entry.update(
                {
                    "accuracy": correct_count / count,
                    "entropy_mean": sum(row["entropy"] for row in rows) / count,
                    "mi_mean": sum(row["mutual_information"] for row in rows) / count,
                    "confidence_mean": sum(row["confidence"] for row in rows) / count,
                }
            )
        label_summaries[int(label)] = entry

    aggregate = {
        "dropout_modules": dropout_count,
        "dropout_rates": dropout_rates,
        "limit": min(limit, len(dataset)),
        "passes": config.passes,
        "microbatch": config.microbatch,
        "temperature": config.tau,
        "label_summaries": label_summaries,
        "class_names": class_names,
        "prompts": prompts,
    }
    if not config.no_predictive:
        total = max(1, sum(len(rows) for rows in summary.values()))
        aggregate["overall_accuracy"] = sum(row["correct"] for rows in summary.values() for row in rows) / total

    save_json(out_dir / "summary.json", aggregate)
    return aggregate


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = CarSimRoutineConfig(
        model_id=args.model,
        device=args.device,
        root=Path(args.root),
        train=args.train,
        limit=args.limit if args.limit > 0 else None,
        passes=args.passes,
        microbatch=args.microbatch,
        tau=args.tau,
        seed=args.seed,
        allow_tf32=not args.disable_tf32,
        dropout_rate=args.dropout_rate,
        adapter_targets=args.adapter_target,
        adapter_p=args.adapter_drop,
        out_dir=Path(args.out),
        save_raw=args.save_raw,
        no_predictive=args.no_predictive,
        prompts=tuple(args.prompt) if args.prompt else None,
        prompt_template=args.prompt_template,
    )
    summary = run(config)
    print("Finished car_sim sweep", summary)


if __name__ == "__main__":
    main()
