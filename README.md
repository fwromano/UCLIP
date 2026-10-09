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

## Sim2 relative heading dataset

The [labeled Sim2 dataset](data/car_sim/heading_labels/README.md) contains
1,949 cropped vehicle views, including 1,404 Jeep views, with recovered camera-relative
headings: front = 0 degrees and rear = 180 degrees. Use the CSV labels rather than
the original filename angles. Four source-counter artifacts have one-degree uncertainty.

- [Interactive viewer](https://fwromano.github.io/datasets/sim2-headings/)
- [CSV labels](data/car_sim/heading_labels/heading_labels.csv)
- [JSON labels and provenance](data/car_sim/heading_labels/heading_labels.json)

Every crop was matched exactly to its source video; independent verification
replayed 110 source frames and cross-checked all 46 images in the 45-degree subset.
See the dataset README for reproduction commands and the angle convention.

## Sim2 cached image embeddings

The [CLIP embedding cache](data/car_sim/clip_embeddings/README.md) adds one
512-dimensional float32 vector for every labeled crop using
`openai/clip-vit-base-patch16`. Unit vectors are stored in `embeddings.npy`;
`index.json` links each row to the original image hash, corrected heading and
source provenance. The model revision and preprocessing are pinned in the
manifest. The cache covers all 1,949 crops, including the 1,404 Jeep images.

```bash
python3 scripts/embed_sim2.py --output /tmp/sim2-clip-regenerated
python3 scripts/embed_sim2.py --output /tmp/sim2-clip-regenerated --verify
```

This supplies cached appearance features for simulation. WIRE appearance ingest
and association scoring remain separate integration work.

Explore the [heading-linked embedding orbit](https://fwromano.github.io/datasets/sim2-headings/embedding.html):
rotate a Jeep view and watch its cached B/16 vector move through a fixed PCA
space shared by all four colors. The standalone [HTML](data/car_sim/clip_embeddings/heading_pca.html)
includes all previews and Plotly for offline use; the [projection JSON](data/car_sim/clip_embeddings/heading_pca.json)
retains the basis, variance and embedding-row associations.

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/build_sim2_embedding_view.py
```
