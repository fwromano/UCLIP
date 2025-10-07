# Monte Carlo Dropout Toolkit

Utilities for running Monte Carlo Dropout (MCDO) uncertainty with CLIP-style backbones. The toolkit follows the battle-tested checklist for toggling `nn.Dropout` at inference, sampling stochastic embeddings, and logging scalar diagnostics and predictive metrics.

## Installation

Install the toolkit in editable mode so the `src/` layout is discoverable:

```bash
pip install -e .[analysis]
```

If you prefer requirements files, `requirements/base.txt` mirrors the runtime dependencies and optional plotting extras can be found in the `analysis` extra above.

## Quick start

```bash
uclip-sample \
  --model openai/clip-vit-base-patch32 \
  --img path/to/image.jpg \
  --out runs/example \
  --passes 128 \
  --labels prompts.txt
```

The legacy invocation (`python -m mcdo.sample`) remains available via lightweight compatibility shims.

Key flags:

- `--dropout-rate` to override existing dropout probabilities globally.
- `--adapter-target` to wrap specific modules with dropout adapters when the backbone defaults to zero dropout.
- `--tau` to set the softmax temperature for predictive probabilities.
- `--save` controls which artifacts are persisted (defaults to `mu,Sigma,embeddings,pbar,entropies`).

Outputs under `--out` include:

- `mu.pt` and `Sigma.pt` (embedding mean and covariance).
- `embeddings.pt` containing all Monte Carlo samples.
- Diagnostic scalars: `trace.json`, `logdet.json`, `offdiag.json`, `eig_topk.json`, and a consolidated `metrics.json`.
- Optional predictive tensors (`probs_mean.npy`, `entropy_mean.npy`, `MI.npy`, `probs_per_pass.npy`).

## Predictive uncertainty

Provide either `--labels prompts.txt` (newline-separated prompts) or `--text-emb some.pt`. The CLI computes deterministic text embeddings once, then averages softmaxed cosine similarities across stochastic vision embeddings, reporting predictive mean probabilities, BALD-style mutual information, and the mean per-pass entropy.

## Determinism safeguards

- Seeds default to zero; override via `--seed`.
- TF32 is enabled unless `--disable-tf32` is supplied.
- The CLI reapplies selective dropout enabling before every pass so LayerNorm and other modules stay in evaluation mode.

The scripts expect the Hugging Face `transformers` cache to be accessible or network downloads to be allowed the first time a backbone is requested.
