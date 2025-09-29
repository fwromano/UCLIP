"""Command-line interface for Monte Carlo Dropout sampling."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Set

import torch

from .dropout import dump_dropout_rates, insert_adapters, override_dropout_rate
from .sampling import (
    compute_embedding_statistics,
    compute_predictive_distribution,
    diagnostics,
    sample_embeddings,
)
from .utils import (
    ensure_dir,
    load_clip_backbone,
    load_prompts,
    load_text_embeddings,
    pil_from_path,
    save_json,
    save_tensor,
    save_numpy,
    set_determinism,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo Dropout sampling for CLIP models")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--img", required=True, help="Path to an input image.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--passes", "-T", type=int, default=128, help="Number of stochastic forward passes.")
    parser.add_argument("--microbatch", type=int, default=1, help="Replicate inputs to amortise kernels.")
    parser.add_argument("--dropout-rate", type=float, default=None, help="Override dropout probability across modules.")
    parser.add_argument(
        "--adapter-target",
        action="append",
        default=[],
        help="Dotted module paths to wrap with dropout adapters.",
    )
    parser.add_argument("--adapter-drop", type=float, default=0.1, help="Dropout probability for adapters.")
    parser.add_argument("--tau", type=float, default=1.0, help="Temperature for predictive softmax.")
    parser.add_argument("--labels", type=str, default=None, help="Path to newline-delimited text prompts.")
    parser.add_argument("--text-emb", type=str, default=None, help="Optional pre-computed text embedding tensor.")
    parser.add_argument("--out", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--save",
        type=str,
        default="mu,Sigma,embeddings,pbar,entropies",
        help="Comma separated artifacts to persist.",
    )
    parser.add_argument(
        "--eig-topk",
        type=int,
        default=10,
        help="Number of covariance eigenvalues to persist in diagnostics.",
    )
    parser.add_argument(
        "--allow-missing-dropout",
        action="store_true",
        help="Skip error if backbone exposes no dropout modules.",
    )
    parser.add_argument(
        "--no-predictive",
        action="store_true",
        help="Skip predictive distribution computation even if labels are provided.",
    )
    return parser.parse_args(argv)


def parse_save_targets(option: str) -> Set[str]:
    return {token.strip().lower() for token in option.split(",") if token.strip()}


def compute_text_embeddings(
    model: torch.nn.Module,
    processor,
    prompts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)
    return text_emb.detach()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    set_determinism(seed=args.seed, allow_tf32=not args.disable_tf32)

    loaded = load_clip_backbone(args.model, device=args.device)
    model, processor, device = loaded.model, loaded.processor, loaded.device

    if args.adapter_target:
        insert_adapters(model, args.adapter_target, p=args.adapter_drop)

    if args.dropout_rate is not None:
        override_dropout_rate(model, args.dropout_rate)

    dropout_count, dropout_rates = dump_dropout_rates(model)
    if dropout_count == 0 and not args.allow_missing_dropout:
        raise RuntimeError(
            "Backbone exposes no dropout modules. Provide --adapter-target or --allow-missing-dropout."
        )

    image = pil_from_path(args.img)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    embeddings = sample_embeddings(
        model=model,
        forward_fn=model.get_image_features,
        forward_kwargs=inputs,
        passes=args.passes,
        microbatch=args.microbatch,
    )

    stats = compute_embedding_statistics(embeddings)
    diag = diagnostics(stats.covariance)

    predictive = None
    if not args.no_predictive:
        text_embeddings = None
        if args.text_emb:
            text_embeddings = load_text_embeddings(args.text_emb)
            if text_embeddings is not None:
                text_embeddings = text_embeddings.to(device)
        elif args.labels:
            prompts = load_prompts(args.labels)
            if prompts:
                text_embeddings = compute_text_embeddings(model, processor, prompts, device)
            else:
                raise ValueError("Provided labels file is empty.")
        if text_embeddings is not None:
            text_embeddings = text_embeddings.to(embeddings.device)
            predictive = compute_predictive_distribution(embeddings, text_embeddings, tau=args.tau)

    out_dir = ensure_dir(args.out)
    save_targets = parse_save_targets(args.save)

    if "mu" in save_targets:
        save_tensor(Path(out_dir) / "mu.pt", stats.mean)
    if "sigma" in save_targets or "cov" in save_targets:
        save_tensor(Path(out_dir) / "Sigma.pt", stats.covariance)
    if "embeddings" in save_targets:
        save_tensor(Path(out_dir) / "embeddings.pt", stats.embeddings)

    eigs_cpu = diag.eigenvalues.detach().cpu()
    metrics_payload = {
        "trace": diag.trace,
        "logdet": diag.logdet,
        "off_diag_mass": diag.off_diag_mass,
        "eigenvalues": eigs_cpu.tolist(),
        "dropout_count": dropout_count,
        "dropout_rates": dropout_rates,
        "passes": args.passes,
        "microbatch": args.microbatch,
        "temperature": args.tau,
    }

    save_json(Path(out_dir) / "trace.json", {"trace": diag.trace})
    save_json(Path(out_dir) / "logdet.json", {"logdet": diag.logdet})
    save_json(Path(out_dir) / "offdiag.json", {"off_diag_mass": diag.off_diag_mass})
    topk = min(args.eig_topk, eigs_cpu.numel())
    top_values = eigs_cpu[-topk:].flip(0).tolist() if topk > 0 else []
    save_json(Path(out_dir) / "eig_topk.json", {"topk": topk, "eigenvalues": top_values})
    save_json(Path(out_dir) / "metrics.json", metrics_payload)

    if predictive is not None:
        if "pbar" in save_targets:
            save_numpy(Path(out_dir) / "probs_mean.npy", predictive["predictive_mean"].detach().cpu().numpy())
        if "entropies" in save_targets:
            save_numpy(
                Path(out_dir) / "entropy_mean.npy",
                predictive["mean_entropy"].detach().cpu().item(),
            )
            save_numpy(
                Path(out_dir) / "MI.npy",
                predictive["mutual_information"].detach().cpu().item(),
            )
        if "probs" in save_targets:
            save_numpy(
                Path(out_dir) / "probs_per_pass.npy",
                predictive["probs_per_pass"].detach().cpu().numpy(),
            )

    print(
        "Saved Monte Carlo Dropout artifacts to",
        str(out_dir),
        "| T=",
        args.passes,
        "| dropout modules=",
        dropout_count,
    )


if __name__ == "__main__":
    main()
