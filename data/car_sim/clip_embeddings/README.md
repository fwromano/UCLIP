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
