# Sim2 cached CLIP image embeddings

Each of the 1,949 labeled Sim2 crops has one 512-dimensional image embedding
from `openai/clip-vit-base-patch16`, including all 1,404 Jeep crops. These are
deterministic image features in evaluation mode, with no Monte Carlo dropout.
Original crops and heading labels are preserved.

## Files and row contract

- `embeddings.npy`: float32 matrix, shape `(1949, 512)`, L2-normalized rows.
- `raw_norms.npy`: float32 projection norms, shape `(1949,)`. Recover the
  unnormalized projected feature with `embeddings[row] * raw_norms[row]`.
- `index.json`: existing image/heading/provenance metadata plus `embedding_row`.
  Image paths remain relative to `data/car_sim`; use the corrected
  `relative_heading_deg`, not the old filename angle.
- `manifest.json`: immutable model revision, complete image-processor settings,
  input-label hash, output hashes, runtime versions and generation settings.
- `verification.json`: all-row integrity checks and independent single-image
  encodes for the nearest four cardinal views in every category.

The model is pinned to Hugging Face commit
`57c216476eefef5ab752ec549e440a49ae4ae5f3`. Images are converted to RGB and passed
through that revision's `CLIPImageProcessor` resize, center-crop, rescale and
normalization. `CLIPModel.get_image_features` returns the projected image
features; the cache stores unit vectors for cosine comparisons. Consumers must
use the same model revision, preprocessing and normalization for comparable
online features. The generator exposes `--model-name` with the requested
default; another model requires its own explicit immutable `--revision`.

## Load a view's vector

```python
import json
from pathlib import Path
import numpy as np

cache = Path('data/car_sim/clip_embeddings')
index = json.loads((cache / 'index.json').read_text())
vectors = np.load(cache / 'embeddings.npy', mmap_mode='r', allow_pickle=False)

category, requested_angle = 'Indigo', 45.0
view = min(
    (row for row in index['images'] if row['category'] == category),
    key=lambda row: abs((row['relative_heading_deg'] - requested_angle + 180) % 360 - 180),
)
embedding = vectors[view['embedding_row']]
print(view['image_path'], view['relative_heading_deg'], embedding.shape)
```

## Generate and verify

Run from UCLIP with its documented dependencies installed. First execution
downloads the pinned model. CPU is the default; CUDA is an explicit option.
Choose a new output directory when regenerating: existing outputs are never
overwritten by generation. Verification writes only its verification report.

```bash
python3 scripts/embed_sim2.py --output /tmp/sim2-clip-regenerated --device cpu
python3 scripts/embed_sim2.py --output /tmp/sim2-clip-regenerated --device cpu --verify
python3 scripts/embed_sim2.py --verify
```

`--batch-size`, `--threads`, and `--cache-dir` control generation resources.
Verification checks every image hash, row-to-label association, artifact hash,
matrix shape/type, finite values and unit norms, then independently re-encodes
the selected views one image at a time. A hash mismatch or numeric disagreement
causes a nonzero exit. The report records the maximum observed disagreement.

## Heading-linked PCA explorer

Open [the interactive embedding orbit](heading_pca.html), or use the
[public viewer](https://fwromano.github.io/datasets/sim2-headings/embedding.html).
The heading slider and autoplay select actual cropped Jeep images and their
cached vectors together. The plot supports interactive 3D, three 2D axis pairs,
click-to-select measured points, all four colors, and a full-vector cosine
comparison to each color's nearest front view. At 360 degrees it selects the
same crop/vector as at zero. The live HTML loads small WebP preview sheets on
demand. [The separate offline viewer](heading_pca_offline.html) embeds all sheets
for disconnected use. Original PNGs remain linked at full resolution.

The shared, centered PCA uses all 1,404 normalized Jeep embeddings. It reuses
`scripts/clip_mcdo_pca.py:fit_pca`; the top three axes capture 38.87 percent of
their variation. Changing color or axes does not refit the basis. One measured
crop represents each available heading; duplicates remain visible in the sample
cloud and can be selected directly. Solid path segments cover angular gaps up
to five degrees. Larger gaps and the wraparound seam are dashed guides, with no
synthetic image, vector interpolation, loop constraint or uncertainty ellipsoid.

`heading_pca.json` exports the mean, three PCA components, variance ratios and
all projected coordinates keyed by embedding row. `heading_pca_verification.json`
records input/output hashes and numerical checks. The browser verification
record and screenshots accompany the generated explorer.

Rebuild from UCLIP with its analysis dependencies installed:

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/build_sim2_embedding_view.py
```

The browser performs presentation and control updates only; Python fits the PCA,
checks image/vector associations and prepares preview images. The template and canvas renderer are
`src/uclip/viz/sim2_embedding_orbit.html` and `sim2_embedding_orbit.js`. The full 512-dimensional cache remains
the authoritative appearance feature; projected distances omit other axes.

## Viewer performance verification

The canvas renderer retains the static point cloud between heading updates and
redraws only the selected marker. Image/marker selection commits together after
its preview is ready. Rapid scrubbing coalesces to the latest request; playback
updates at most 30 times per second. Heading lookup tables and coordinate DOM
nodes are prepared once. Preview sheets contain at most 64 crops at 256 × 176;
the browser retains at most three decoded sheets and loads at most two at once.
The plot renders at display resolution and supports drag rotation and zoom.

`heading_pca_performance.json` records two equivalent old/new trials in local
Chromium with network caching disabled and 4× CPU throttling. This measures
browser presentation, not deployment-network latency or field behavior. The
exported PCA JSON is byte-identical to the previous viewer. The browser report
checks headings, colors, projections, scrubbing, point selection, autoplay,
mobile layout and disconnected use.

Repeat the benchmark using Playwright CLI's `run-code` command after opening
any generated viewer URL in a browser session:

```bash
python3 -m http.server 18974 --directory data/car_sim/clip_embeddings
# In another terminal, with playwright-cli installed:
playwright-cli -s=orbit-bench open http://127.0.0.1:18974/heading_pca.html
playwright-cli -s=orbit-bench run-code "$(cat scripts/benchmark_sim2_embedding_view.js)"
```

The benchmark requires Chromium's DevTools protocol. It records load-to-ready,
requestAnimationFrame percentiles, main-thread long tasks, heap use, HTML bytes
and marker association during a 60°/second playback. Use the same browser,
viewport and machine for before/after comparisons. The older build is retained
at UCLIP commit `373d4ab3b31c5d22fafe2d088b89f0e0fbf8d987`.

## Simulation evidence boundary

This cache supplies appearance features for a selected crop. It does not add
appearance evidence to WIRE messages or implement an appearance likelihood.
Those remain separate SECTR-C integration work. The cache also does not measure
embedding uncertainty, camera elevation, image degradation or instance identity.
Repeated lookup of a cached view is not an independent appearance observation.

For simulation, choose a view after the camera's detection checks; retain the
actual selected azimuth and angular mismatch in diagnostics. Keep simulator
truth IDs out of tracker inputs. Compare appearance through an explicit scoring
model alongside the existing position/class evidence. Heading calibration and
the four one-degree counter uncertainties remain as documented in
`../heading_labels/README.md`.

References: [model](https://huggingface.co/openai/clip-vit-base-patch16),
[CLIP API](https://huggingface.co/docs/transformers/model_doc/clip).
