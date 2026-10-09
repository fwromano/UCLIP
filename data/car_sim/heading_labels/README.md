# Sim2 relative heading labels

Generated and checked on 2026-10-09. These manifests label all 1,949 existing
`sim2_cropped` PNGs, including 1,404 Jeep images and 545 Moose/Wolf images.
The empty `Chrome` directory has no images to label.

## Convention

| Relative heading | View |
| --- | --- |
| 0 degrees | Front faces the camera |
| 90 degrees | Side view, vehicle nose points toward image right |
| 180 degrees | Rear faces the camera |
| 270 degrees | Side view, vehicle nose points toward image left |

Headings are in `[0, 360)`. They describe relative viewing azimuth, not geographic
compass heading. Jeep front/rear/side views were checked visually. Moose/Wolf
labels retain their renderer's angle convention; their mechanical front/back
orientation was not independently calibrated. The counter is integer-valued;
these labels do not measure continuous 3D pose or camera elevation.

## Files

- `heading_labels.csv`: image-to-heading lookup. `image_path` is relative to
  `UCLIP/data/car_sim/`. Filter `vehicle_type=jeep` for the four Jeep colors.
- `heading_labels.json`: the same labels plus crop rectangles, source-video
  provenance, video/image SHA-256 hashes, source-counter audits, and the complete
  45-degree reference crosswalk.
- `heading_review.html`: offline image browser with category selection and a
  heading slider; its image paths resolve against the adjacent dataset.
- `reviewed_labels.json` and `counter_review.png`: hash-bound reviews of the six
  counters rejected by automated recognition.
- `summary.json` and `verification.json`: coverage and verification results.

Use the manifest rather than parsing the image filename. Original PNG names and
pixels are unchanged. Some headings have multiple crops; some headings have no
available image. The browser shows the nearest available view and its actual
label.

## Recovery method and evidence

Every full-set crop was matched pixel-for-pixel to a rectangle in its original
Sim2 MP4. The old extraction decoder identifies candidate source frames; a new
glyph recognizer reads the source video's displayed angle. Label recovery does
not depend on CLIP similarity or assumed filenames.

The labels correct 378 old filename angles. Of 1,949 labels, 1,943 were read
automatically from source counters and two were read manually from matched
source frames. Four original video overlays display the corrupt value `990`
around the 99-to-100-degree transition. Those four labels are recovered as
99 degrees with one-degree label uncertainty and explicit review provenance.
For direct/manual counter reads, zero label uncertainty means the displayed
integer was read; it does not remove renderer rounding or camera/model error.

All 46 images in `sim2_cropped_45deg` match a full-set crop by SHA-256. Those
filenames are target angles, not guaranteed exact headings: nine differ from
the recovered source angle. In particular, `Wolf2_180.png` in the subset
actually matches a 352-degree full-set view, and its `Wolf2_315.png` matches
299 degrees. Preserve the reference crosswalk rather than using those target
filenames as ground truth. Jeep references differ by at most seven degrees.

Verification checked full-set coverage and every image hash, independently
replayed 110 source frames, checked all 46 reference hashes, and checked the
Jeep cardinal-view convention. All source-video counter audits have zero
unexplained decreases or jumps above three degrees. This is dataset-label
verification, not a trained heading-estimator benchmark.

## Cached appearance features

The adjacent [CLIP cache](../clip_embeddings/README.md) adds a
512-dimensional `openai/clip-vit-base-patch16` image embedding for every crop.
Its `index.json` preserves these heading/provenance fields and adds
`embedding_row`, pointing into the normalized float32 `embeddings.npy` matrix.
The model revision, preprocessing and verification evidence are retained there.

Source repository revisions at recovery:

- UCLIP: `aafa1b3fc00873a0d7b7f2f9385c814f9fae35e6`
- VectorManifold: `3280f04a9ff0918aa817bdacf4dc2a2c8fabec76`

## Reproduce

Run from the UCLIP repository. Requires Python, NumPy and OpenCV; no model
downloads are needed.

```bash
python3 scripts/label_sim2_headings.py \
  --data-root data/car_sim \
  --video-root ../VectorManifold/data/raw/video/sim2 \
  --legacy-extractor ../VectorManifold/scripts/process_sim2.py \
  --reviewed-labels data/car_sim/heading_labels/reviewed_labels.json \
  --output data/car_sim/heading_labels

python3 scripts/verify_sim2_heading_labels.py \
  data/car_sim/heading_labels/heading_labels.json
```

The generator exits unsuccessfully if images remain unmatched/unlabeled or
the recovered counter timeline needs review. Human reviews apply only to
the recorded image hashes and exact matched source-frame indices.
