"""General utilities for Monte Carlo Dropout workflows."""
from __future__ import annotations

import importlib
import json
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn

def _ensure_flex_attention_stub() -> None:
    """Avoid importing torch flex attention when optional deps are missing."""

    module_name = "transformers.integrations.flex_attention"
    if module_name in sys.modules:
        return

    try:
        importlib.import_module(module_name)
    except Exception:  # pragma: no cover - exercised only when deps are missing
        # Ensure parent package is initialised before injecting the stub.
        try:
            importlib.import_module("transformers.integrations")
        except Exception:
            pass

        stub = types.ModuleType(module_name)

        def _unavailable(*_args, **_kwargs):
            raise RuntimeError(
                "Flex attention support is unavailable in this environment."
            )

        stub.flex_attention_forward = _unavailable  # type: ignore[attr-defined]
        stub.make_flex_block_causal_mask = _unavailable  # type: ignore[attr-defined]
        stub.WrappedFlexAttention = None  # type: ignore[attr-defined]
        stub.repeat_kv = _unavailable  # type: ignore[attr-defined]
        sys.modules[module_name] = stub


_ensure_flex_attention_stub()


try:  # transformers is optional until runtime
    from transformers import CLIPModel, CLIPProcessor
except ImportError:  # pragma: no cover - handled at CLI runtime
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore


@dataclass
class LoadedModel:
    model: nn.Module
    processor: "CLIPProcessor"
    device: torch.device


def set_determinism(seed: int = 0, allow_tf32: bool = True) -> None:
    """Set global seeds and toggle TF32 to stabilise stochastic runs."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_clip_backbone(model_id: str, device: Optional[str] = None) -> LoadedModel:
    if CLIPModel is None or CLIPProcessor is None:
        raise RuntimeError("transformers is required to load CLIP backbones.")

    resolved_device = resolve_device(device)
    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)

    model.to(resolved_device)
    model.eval()

    return LoadedModel(model=model, processor=processor, device=resolved_device)


def pil_from_path(path: str | Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_prompts(path: Optional[str]) -> Optional[Sequence[str]]:
    if path is None:
        return None
    payload = Path(path).read_text().strip().splitlines()
    return [line.strip() for line in payload if line.strip()]


def load_text_embeddings(path: Optional[str]) -> Optional[torch.Tensor]:
    if path is None:
        return None
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Text embedding checkpoint did not contain a tensor.")
    return tensor


def normalise_embeddings(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = normalise_embeddings(a)
    b_n = normalise_embeddings(b)
    return a_n @ b_n.T


def softmax_with_temperature(logits: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.softmax(logits / max(tau, 1e-6), dim=-1)


def save_numpy(path: Path, array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array))
