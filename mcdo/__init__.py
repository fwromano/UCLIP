"""Monte Carlo Dropout utilities for CLIP-style models."""

from .dropout import (
    DropoutAdapter,
    collect_dropout_modules,
    dump_dropout_rates,
    enable_mc_dropout,
    insert_adapters,
    override_dropout_rate,
    wrap_with_dropout,
)
from .sampling import (
    MCDOSamplingResult,
    MCSDiagnosticMetrics,
    compute_embedding_statistics,
    compute_predictive_distribution,
    diagnostics,
    sample_embeddings,
)
from .utils import set_determinism

__all__ = [
    "DropoutAdapter",
    "collect_dropout_modules",
    "dump_dropout_rates",
    "enable_mc_dropout",
    "insert_adapters",
    "override_dropout_rate",
    "wrap_with_dropout",
    "MCDOSamplingResult",
    "MCSDiagnosticMetrics",
    "compute_embedding_statistics",
    "compute_predictive_distribution",
    "diagnostics",
    "sample_embeddings",
    "set_determinism",
]
