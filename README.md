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
space shared by all four colors. The [live HTML](data/car_sim/clip_embeddings/heading_pca.html) loads preview
sheets on demand; the [offline HTML](data/car_sim/clip_embeddings/heading_pca_offline.html)
embeds them for disconnected use. Both use a lightweight canvas renderer; the [projection JSON](data/car_sim/clip_embeddings/heading_pca.json)
retains the basis, variance and embedding-row associations.

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/build_sim2_embedding_view.py
```


## MAAS vehicle sphere scans and camera-matched appearance study

The SECTR-C `MAAS_Vehicle-Scans_Dataset.zip` extends the Sim2 study with
2,184 rendered frames: ten truck paint variants and original Jeep, Wolf and
Moose views, each at 24 azimuths and seven elevations (−5 through 90 degrees).
The [sphere study](https://fwromano.github.io/datasets/vehicle-scans/) compares
full-frame/cropped CLIP features and held-out angles, with a
[shared embedding explorer](https://fwromano.github.io/datasets/vehicle-scans/embedding.html)
that includes the existing Sim2 Jeep features in one fixed PCA basis.

The source archive stays in SECTR-C. These commands read it in place and write
explicit output directories; they never publish ROS or alter tracker scoring:

```bash
python3 scripts/prepare_vehicle_scans.py \
  --archive /path/to/MAAS_Vehicle-Scans_Dataset.zip \
  --yolo-weights /path/to/yolov8n.pt \
  --crop-overrides docs/vehicle_scans/crop-overrides.json \
  --output /path/to/scans-cropped

python3 scripts/build_vehicle_scan_study.py \
  --crops /path/to/scans-cropped --output /path/to/scans-study \
  --sim2-cache data/car_sim/clip_embeddings --model-cache /path/to/offline-hf-cache

python3 scripts/evaluate_vehicle_scan_viewpoints.py \
  --crops /path/to/scans-cropped --study /path/to/scans-study \
  --output /path/to/scans-study/elevation_holdout.json

python3 scripts/sample_vehicle_scan_camera.py \
  --crops /path/to/scans-cropped --study /path/to/scans-study \
  --truth /path/to/composer.truth.json --sensors /path/to/sensors.json \
  --wire-root /path/to/afcagc_wire --model-cache /path/to/offline-hf-cache \
  --target-map truck01=truck/yellow --output /path/to/camera-crops

python3 scripts/build_vehicle_scan_view.py \
  --crops /path/to/scans-cropped --study /path/to/scans-study \
  --camera /path/to/camera-crops --sim2-root . --output /path/to/viewer

python3 -m unittest discover -s tests -p test_vehicle_scans.py
```

Use the recorded NumPy/Pillow/Torch/Transformers/Matplotlib environment, plus
OpenCV and Ultralytics (crop preparation) and PyProj (camera replay). The study
requires the offline cache of the same immutable CLIP B/16 revision as Sim2.
`--resume` resumes hash-bound embedding stages; existing outputs otherwise need
an explicit new directory. `build_vehicle_scan_view.py --refresh` regenerates
its own presentation outputs without recomputing embeddings.

Folder labels define the study categories. YOLO classes are crop-audit metadata,
with a padded foreground fallback for missed views. Truck colors share crop
bounds at a viewpoint; `crop-overrides.json` is an exported snapshot of the
SECTR-C dataset review. The visually inferred heading convention is
`(270 - filename azimuth) % 360`; first filename-parameter distance units remain
unconfirmed. Camera replay uses WIRE's actual camera/profile/cuboid helpers,
aligns the source image to the camera axes, then isotropically resamples and
independently encodes the resulting pixels. Exact trajectory samples and asset
bindings are required. Unsupported people and clipped bodies are explicit.

The evidence covers rendered model/paint categories and deterministic camera
replay, with authored physical dimensions. It does not calibrate optical effects,
classify independent same-color vehicle identities, or add an MHT appearance
likelihood. Existing Sim2 caches and studies remain intact.
