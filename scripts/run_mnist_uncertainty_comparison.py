#!/usr/bin/env python
"""Compare MNIST uncertainty methods: (A) encoder-side MCDO + logistic head, (B) Laplace logistic head."""
from __future__ import annotations

import argparse
import json
import os
import sys
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
os.environ["NETWORKX_GRAPH_BACKEND"] = "networkx"
os.environ["NETWORKX_BACKEND"] = "networkx"

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tqdm import tqdm

from uclip.core.dropout import enable_mc_dropout, insert_adapters
from uclip.core.utils import load_clip_backbone, set_determinism

from PIL import Image
# ----------------------------
# Dataset utilities
# ----------------------------


class ArrayMNIST(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = images.astype(np.uint8)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        image = Image.fromarray(self.images[idx], mode="L")
        label = int(self.labels[idx])
        return image, label


def collate_pil(batch: Sequence[Tuple[Image.Image, int]]) -> Tuple[List[Image.Image], torch.Tensor]:
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


def read_idx_images(path: Path) -> np.ndarray:
    with open(path, "rb") as handle:
        magic, num, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected magic number {magic} in {path}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with open(path, "rb") as handle:
        magic, num = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected magic number {magic} in {path}")
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data


def split_mnist(root: Path) -> Tuple[Dataset, Dataset, Dataset]:
    root.mkdir(parents=True, exist_ok=True)
    raw_dir = root / "MNIST" / "raw"
    train_images = read_idx_images(raw_dir / "train-images-idx3-ubyte")
    train_labels = read_idx_labels(raw_dir / "train-labels-idx1-ubyte")
    test_images = read_idx_images(raw_dir / "t10k-images-idx3-ubyte")
    test_labels = read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte")

    train_full = ArrayMNIST(train_images, train_labels)
    test_set = ArrayMNIST(test_images, test_labels)

    train_indices = list(range(0, 50_000))
    val_indices = list(range(50_000, 60_000))

    train_set = Subset(train_full, train_indices)
    val_set = Subset(train_full, val_indices)
    return train_set, val_set, test_set


class NoiseDataset(Dataset):
    def __init__(self, base: Dataset, sigma: float, seed: int = 0) -> None:
        self.base = base
        self.sigma = sigma
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple["Image.Image", int]:
        image, label = self.base[idx]
        arr = np.array(image).astype(np.float32) / 255.0
        rng = np.random.default_rng(self.seed + idx)
        noise = rng.normal(0.0, self.sigma, size=arr.shape).astype(np.float32)
        perturbed = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((perturbed * 255.0).astype(np.uint8)), label


class DownsampleDataset(Dataset):
    def __init__(self, base: Dataset, target_size: int) -> None:
        self.base = base
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple["Image.Image", int]:
        image, label = self.base[idx]
        width, height = image.size
        new_img = image.resize((self.target_size, self.target_size), Image.BILINEAR)
        restored = new_img.resize((width, height), Image.BILINEAR)
        return restored, label


# ----------------------------
# Feature extraction
# ----------------------------


def compute_features(
    model,
    processor,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pil)
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Extracting features", leave=False, disable=True):
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            emb = model.get_image_features(**inputs).detach().cpu()
            feats.append(emb)
            labels.append(targets)
    return torch.cat(feats), torch.cat(labels)


# ----------------------------
# Logistic head & training
# ----------------------------


class LogisticHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 10) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_head(
    head: LogisticHead,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> LogisticHead:
    train_ds = TensorDataset(train_feats, train_labels)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    head = head.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val = float("inf")

    for epoch in range(epochs):
        head.train()
        running_loss = 0.0
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * feats.size(0)

        val_loss = evaluate_ce(head, val_feats, val_labels, criterion, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = head.state_dict()

        print(f"Epoch {epoch+1:02d}: train loss {running_loss / len(train_ds):.4f} | val CE {val_loss:.4f}")

    if best_state is not None:
        head.load_state_dict(best_state)
    return head.to(torch.device("cpu"))


def evaluate_ce(
    head: LogisticHead,
    feats: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    head.eval()
    with torch.no_grad():
        logits = head(feats.to(device))
        loss = criterion(logits, labels.to(device))
    return float(loss.item())


# ----------------------------
# Temperature scaling
# ----------------------------


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200, lr: float = 5e-3) -> float:
    device = logits.device
    log_t = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([log_t], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(max_iter):
        optimizer.zero_grad()
        temperature = torch.exp(log_t)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        optimizer.step()

    temperature = float(torch.exp(log_t).item())
    return max(temperature, 1e-3)


# ----------------------------
# Metrics
# ----------------------------


def compute_classification_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    log_probs = torch.log(probs.clamp(min=1e-12))
    nll = -log_probs[torch.arange(len(labels)), labels].mean().item()

    one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
    brier = ((probs - one_hot) ** 2).sum(dim=1).mean().item()

    ece = expected_calibration_error(probs, labels)
    acc = (probs.argmax(dim=1) == labels).float().mean().item()
    return {"accuracy": acc, "nll": nll, "brier": brier, "ece": ece}


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1)
    ece = torch.zeros(1)
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        accuracy = accuracies[mask].float().mean()
        confidence = confidences[mask].mean()
        ece += (mask.float().mean()) * torch.abs(accuracy - confidence)
    return float(ece.item())


def predictive_entropies(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)


def mutual_information(sample_probs: torch.Tensor, mean_probs: torch.Tensor) -> torch.Tensor:
    mean_entropy = -(sample_probs * torch.log(sample_probs.clamp_min(1e-12))).sum(dim=-1).mean(dim=0)
    predictive_entropy = predictive_entropies(mean_probs)
    return predictive_entropy - mean_entropy


def variation_ratio(sample_probs: torch.Tensor) -> torch.Tensor:
    preds = sample_probs.argmax(dim=-1)  # [T, N]
    one_hot = F.one_hot(preds, num_classes=sample_probs.size(-1))
    counts = one_hot.sum(dim=0)
    max_counts = counts.max(dim=1).values
    return 1.0 - max_counts.float() / sample_probs.size(0)


# ----------------------------
# Method A evaluation
# ----------------------------


def evaluate_mcdo(
    model,
    head: LogisticHead,
    processor,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    temperature: float,
    passes: int = 50,
) -> Dict[str, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pil)
    all_mean_probs: List[torch.Tensor] = []
    all_sample_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    head = head.to(device)
    head.eval()
    model.eval()

    for images, labels in tqdm(loader, desc="MCDO inference", leave=False, disable=True):
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        labels = labels.to(device)

        sample_logits: List[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(passes):
                model.apply(enable_mc_dropout)
                feats = model.get_image_features(**inputs)
                logits = head(feats)
                sample_logits.append((logits / temperature))

        stacked_logits = torch.stack(sample_logits)  # [T, B, C]
        sample_probs = stacked_logits.softmax(dim=-1)
        mean_probs = sample_probs.mean(dim=0)

        all_sample_probs.append(sample_probs.cpu())
        all_mean_probs.append(mean_probs.cpu())
        all_labels.append(labels.cpu())

    mean_probs = torch.cat(all_mean_probs)
    sample_probs = torch.cat(all_sample_probs, dim=1)
    labels = torch.cat(all_labels)

    return {"mean_probs": mean_probs, "sample_probs": sample_probs, "labels": labels}


# ----------------------------
# Method B (Laplace)
# ----------------------------


@dataclass
class LaplacePosterior:
    weight_mean: torch.Tensor
    bias_mean: torch.Tensor
    weight_var: torch.Tensor
    bias_var: torch.Tensor


def estimate_laplace(
    head: LogisticHead,
    features: torch.Tensor,
    labels: torch.Tensor,
    prior_precision: float = 1.0,
) -> LaplacePosterior:
    head.eval()
    with torch.no_grad():
        logits = head(features)
        probs = logits.softmax(dim=1)

    x_sq = features.pow(2)  # [N, D]
    weight_var = torch.empty_like(head.linear.weight)
    bias_var = torch.empty_like(head.linear.bias)

    for c in range(probs.size(1)):
        weights = probs[:, c] * (1.0 - probs[:, c])
        hessian_diag = weights.unsqueeze(1) * x_sq
        diag_sum = hessian_diag.sum(dim=0) + prior_precision
        bias_diag = weights.sum() + prior_precision
        weight_var[c] = 1.0 / diag_sum
        bias_var[c] = 1.0 / bias_diag

    return LaplacePosterior(
        weight_mean=head.linear.weight.detach().clone(),
        bias_mean=head.linear.bias.detach().clone(),
        weight_var=weight_var,
        bias_var=bias_var,
    )


def evaluate_laplace(
    posterior: LaplacePosterior,
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    samples: int = 50,
) -> Dict[str, torch.Tensor]:
    weight_std = posterior.weight_var.sqrt()
    bias_std = posterior.bias_var.sqrt()

    weight_samples = posterior.weight_mean.unsqueeze(0) + torch.randn(samples, *posterior.weight_mean.shape) * weight_std.unsqueeze(0)
    bias_samples = posterior.bias_mean.unsqueeze(0) + torch.randn(samples, *posterior.bias_mean.shape) * bias_std.unsqueeze(0)

    logits = torch.matmul(features, weight_samples.transpose(1, 2)) + bias_samples.unsqueeze(1)
    logits = logits / temperature
    sample_probs = logits.softmax(dim=-1)
    mean_probs = sample_probs.mean(dim=0)

    return {"mean_probs": mean_probs, "sample_probs": sample_probs, "labels": labels}


# ----------------------------
# OOD helpers
# ----------------------------


def ood_metrics(id_unc: torch.Tensor, ood_unc: torch.Tensor) -> Dict[str, float]:
    y_true = np.concatenate([np.zeros(len(id_unc)), np.ones(len(ood_unc))])
    scores = np.concatenate([id_unc.numpy(), ood_unc.numpy()])
    auroc = roc_auc_score(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    aupr = auc(recall, precision)
    return {"auroc": float(auroc), "aupr": float(aupr)}


# ----------------------------
# Main
# ----------------------------


def main(args: argparse.Namespace) -> None:
    set_determinism(0)
    torch.set_grad_enabled(True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    train_set, val_set, test_set = split_mnist(Path(args.data_root))

    if args.max_train > 0:
        train_set = Subset(train_set, range(min(args.max_train, len(train_set))))
    if args.max_eval > 0:
        val_set = Subset(val_set, range(min(args.max_eval, len(val_set))))
        eval_indices = range(min(args.max_eval, len(test_set)))
        test_set = Subset(test_set, eval_indices)

    loaded = load_clip_backbone(args.model_id, device=args.device)
    clip_model, processor = loaded.model, loaded.processor

    train_feats, train_labels = compute_features(clip_model, processor, train_set, args.batch_size, device)
    val_feats, val_labels = compute_features(clip_model, processor, val_set, args.batch_size, device)
    test_feats, test_labels = compute_features(clip_model, processor, test_set, args.batch_size, device)

    head = LogisticHead(train_feats.shape[1], num_classes=10)
    head = train_head(head, train_feats, train_labels, val_feats, val_labels, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay)

    val_logits = head(val_feats).detach()
    temperature = fit_temperature(val_logits, val_labels)
    print(f"Fitted temperature: {temperature:.4f}")

    prior_precision = args.laplace_prior
    laplace_posterior = estimate_laplace(head, train_feats, train_labels, prior_precision=prior_precision)

    noise_levels = [0.05, 0.10, 0.20, 0.35]
    downsample_sizes = [24, 20, 16, 12]
    noise_datasets = {f"noise_{sigma:.2f}": NoiseDataset(test_set, sigma=sigma, seed=42) for sigma in noise_levels}
    down_datasets = {f"down_{size}": DownsampleDataset(test_set, target_size=size) for size in downsample_sizes}

    # Method B evaluation (deterministic features suffice)
    print("Evaluating Laplace head...")
    method_b_results: Dict[str, Dict[str, float]] = {}
    laplace_test = evaluate_laplace(laplace_posterior, test_feats, test_labels, temperature, samples=args.samples)
    metrics_b = compute_classification_metrics(laplace_test["mean_probs"], test_labels)
    entropy_b = predictive_entropies(laplace_test["mean_probs"])
    mi_b = mutual_information(laplace_test["sample_probs"], laplace_test["mean_probs"])
    method_b_results["test"] = {**metrics_b, "entropy_mean": float(entropy_b.mean()), "mi_mean": float(mi_b.mean())}

    ood_scores_b: Dict[str, Dict[str, float]] = {}
    for name, dataset in {**noise_datasets, **down_datasets}.items():
        feats, labels = compute_features(clip_model, processor, dataset, args.batch_size, device)
        eval_res = evaluate_laplace(laplace_posterior, feats, labels, temperature, samples=args.samples)
        entropy_ood = predictive_entropies(eval_res["mean_probs"])
        ood_scores_b[name] = ood_metrics(entropy_b, entropy_ood)

    # Method A evaluation (requires dropout instrumentation)
    adapter_targets = [
        "vision_model.encoder.layers.8",
        "vision_model.encoder.layers.9",
        "vision_model.encoder.layers.10",
        "vision_model.encoder.layers.11",
        "visual_projection",
    ]
    insert_adapters(clip_model, adapter_targets, p=args.dropout_p)
    print("Evaluating MCDO...")
    method_a_results: Dict[str, Dict[str, float]] = {}
    mcdo_test = evaluate_mcdo(clip_model, head, processor, test_set, args.batch_size, device, temperature, passes=args.passes)
    metrics_a = compute_classification_metrics(mcdo_test["mean_probs"], mcdo_test["labels"])
    entropy_a = predictive_entropies(mcdo_test["mean_probs"])
    mi_a = mutual_information(mcdo_test["sample_probs"], mcdo_test["mean_probs"])
    var_ratio_a = variation_ratio(mcdo_test["sample_probs"])
    method_a_results["test"] = {
        **metrics_a,
        "entropy_mean": float(entropy_a.mean()),
        "mi_mean": float(mi_a.mean()),
        "variation_ratio_mean": float(var_ratio_a.mean()),
    }

    ood_scores_a: Dict[str, Dict[str, float]] = {}
    for name, dataset in {**noise_datasets, **down_datasets}.items():
        mcdo_ood = evaluate_mcdo(clip_model, head, processor, dataset, args.batch_size, device, temperature, passes=args.passes)
        entropy_ood = predictive_entropies(mcdo_ood["mean_probs"])
        ood_scores_a[name] = ood_metrics(entropy_a, entropy_ood)

    output = {
        "temperature": temperature,
        "method_a": {"test": method_a_results["test"], "ood": ood_scores_a},
        "method_b": {"test": method_b_results["test"], "ood": ood_scores_b},
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mnist_uncertainty_comparison.json"
    json_path.write_text(json.dumps(output, indent=2))
    print(f"Saved summary to {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MNIST CLIP uncertainty methods (MCDO vs Laplace).")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--model-id", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout-p", type=float, default=0.10)
    parser.add_argument("--passes", type=int, default=50)
    parser.add_argument("--samples", type=int, default=50, help="Weight samples for Laplace head.")
    parser.add_argument("--laplace-prior", type=float, default=1.0)
    parser.add_argument("--max-train", type=int, default=0, help="Limit training samples (0 = full).")
    parser.add_argument("--max-eval", type=int, default=0, help="Limit evaluation samples (0 = full).")
    parser.add_argument("--output-dir", type=str, default="docs/report_assets/mnist_uncertainty")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
