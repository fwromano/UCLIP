"""Dropout instrumentation helpers."""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn


_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
)


def is_dropout(module: nn.Module) -> bool:
    return isinstance(module, _DROPOUT_TYPES)


def enable_mc_dropout(module: nn.Module) -> None:
    """Toggle dropout layers to training mode without disturbing other modules."""

    if is_dropout(module):
        module.train()


def collect_dropout_modules(model: nn.Module) -> List[nn.Module]:
    return [m for m in model.modules() if is_dropout(m)]


class DropoutAdapter(nn.Module):
    """Wrap a module with an additional dropout stage."""

    def __init__(self, base: nn.Module, p: float = 0.1) -> None:
        super().__init__()
        self.base = base
        self.do = nn.Dropout(p)

    def forward(self, *args, **kwargs):  # type: ignore[override]
        output = self.base(*args, **kwargs)
        if isinstance(output, torch.Tensor):
            return self.do(output)
        if isinstance(output, tuple):
            if not output:
                return output
            first, *rest = output
            return (self.do(first), *rest)
        if isinstance(output, list) and output:
            first, *rest = output
            return [self.do(first), *rest]
        return output


def wrap_with_dropout(parent: nn.Module, attribute: str, p: float = 0.1) -> None:
    """Insert a dropout adapter around `getattr(parent, attribute)`."""

    target = getattr(parent, attribute)
    adapter = DropoutAdapter(target, p=p)
    setattr(parent, attribute, adapter)


def dump_dropout_rates(model: nn.Module) -> Tuple[int, List[float]]:
    drops = collect_dropout_modules(model)
    return len(drops), [module.p for module in drops]  # type: ignore[attr-defined]


def insert_adapters(model: nn.Module, target_paths: List[str], p: float = 0.1) -> None:
    """Wrap target submodules defined by dotted paths with dropout adapters."""

    for path in target_paths:
        if not path:
            continue
        if "." in path:
            *parents, attribute = path.split(".")
            parent = model
            for name in parents:
                parent = getattr(parent, name)
        else:
            parent = model
            attribute = path
        wrap_with_dropout(parent, attribute, p=p)


def override_dropout_rate(model: nn.Module, p: float) -> None:
    for module in collect_dropout_modules(model):
        module.p = p
