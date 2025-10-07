"""Sampling and summary routines for Monte Carlo Dropout."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
from torch import nn

from .dropout import enable_mc_dropout
from .utils import cosine_similarity, normalise_embeddings, softmax_with_temperature


@dataclass
class MCDOSamplingResult:
    embeddings: torch.Tensor
    mean: torch.Tensor
    covariance: torch.Tensor


@dataclass
class MCSDiagnosticMetrics:
    trace: float
    logdet: float
    off_diag_mass: float
    eigenvalues: torch.Tensor


@torch.no_grad()
def sample_embeddings(
    model: nn.Module,
    forward_fn: Callable[..., torch.Tensor],
    forward_kwargs: dict,
    passes: int,
    microbatch: int = 1,
) -> torch.Tensor:
    """Collect T stochastic embeddings with dropout enabled."""

    if passes <= 0:
        raise ValueError("passes must be > 0")

    chunks = []
    remaining = passes
    inputs = forward_kwargs

    while remaining > 0:
        current = min(microbatch, remaining)
        batched_inputs = {
            key: value.repeat(current, *([1] * (value.ndim - 1)))
            if isinstance(value, torch.Tensor) and value.shape[0] == 1
            else value
            for key, value in inputs.items()
        }
        model.apply(enable_mc_dropout)
        outputs = forward_fn(**batched_inputs)
        chunks.append(outputs.detach())
        remaining -= current

    stacked = torch.cat(chunks, dim=0)
    return stacked


def compute_embedding_statistics(samples: torch.Tensor) -> MCDOSamplingResult:
    if samples.ndim != 2:
        raise ValueError("samples must be 2D [T, D]")

    samples = samples.float()
    mean = samples.mean(dim=0)
    if samples.shape[0] < 2:
        dim = samples.shape[1]
        covariance = torch.zeros(dim, dim, dtype=samples.dtype, device=samples.device)
    else:
        covariance = torch.cov(samples.T)
    return MCDOSamplingResult(embeddings=samples, mean=mean, covariance=covariance)


def compute_predictive_distribution(
    embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    tau: float = 1.0,
) -> Dict[str, torch.Tensor]:
    sims = cosine_similarity(embeddings, text_embeddings)
    probs = softmax_with_temperature(sims, tau)
    predictive_mean = probs.mean(dim=0)
    per_pass_entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)
    epistemic = entropy(predictive_mean) - per_pass_entropy.mean()
    return {
        "probs_per_pass": probs,
        "predictive_mean": predictive_mean,
        "mean_entropy": per_pass_entropy.mean(),
        "per_pass_entropy": per_pass_entropy,
        "mutual_information": epistemic,
    }


def entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-12)
    return -(p * p.log()).sum()


def diagnostics(covariance: torch.Tensor, eps: float = 1e-6) -> MCSDiagnosticMetrics:
    dim = covariance.shape[0]
    jitter = eps * torch.eye(dim, device=covariance.device, dtype=covariance.dtype)
    stable_cov = covariance + jitter
    trace = torch.trace(stable_cov).item()
    logdet = torch.logdet(stable_cov).item()
    diag = torch.diag(stable_cov)
    off_diag = (stable_cov - torch.diag(diag)).abs().sum().item()
    eig = torch.linalg.eigvalsh(stable_cov)
    return MCSDiagnosticMetrics(trace=trace, logdet=logdet, off_diag_mass=off_diag, eigenvalues=eig)
