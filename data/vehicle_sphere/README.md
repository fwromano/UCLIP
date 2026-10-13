# MAAS vehicle sphere-view dataset — v1

[Download the full dataset](https://github.com/fwromano/UCLIP/releases/tag/vehicle-sphere-v1)

The two original source archives contain the truck, Jeep, Wolf and Moose rendered
views. The combined index covers **3,864 views in 23 vehicle/paint series**, with
24 source azimuths × seven elevations per series. This is a hemisphere plus a
−5° ring, not full lower-sphere coverage. The second archive repeats the 168
original Jeep views; the prepared index retains them once.

| Release asset | Contents |
|---|---|
| `MAAS_Vehicle-Scans_Dataset.zip` | 2,184 original 640×640 PNGs: ten truck paints and original Jeep/Wolf/Moose |
| `MAAS_Vehicle-Scans_Dataset_Jeep.zip` | 1,848 Jeep PNGs: original plus ten paint variants |
| `vehicle-sphere-prepared-v1.zip` | 3,864 reviewed padded crops, row index and pinned 512D CLIP B/16 embeddings |
| `SHA256SUMS.txt` | SHA256 for the three data archives |

The repository commits the combined `index.json`, embeddings, model/evaluation
metadata and download manifest. Full-resolution source PNGs and prepared crops
are release assets. The index's `embedding_row` addresses the NumPy cache; each
`crop_path` resolves within the extracted `vehicle-sphere-v1/` directory.
Historical absolute source/preparation paths are retained for exact provenance;
they are not required to load the crops or cached features.

```bash
gh release download vehicle-sphere-v1 --repo fwromano/UCLIP --pattern '*.zip' --pattern SHA256SUMS.txt
sha256sum -c SHA256SUMS.txt
unzip vehicle-sphere-prepared-v1.zip
```

All crop hashes, finite unit embeddings and 23×168 coverage were checked at
publication. The original 2,184 vectors remain byte-identical. Folder labels
define vehicle/paint categories; YOLO labels are crop-audit metadata. Relative
heading is `(270 - source_azimuth_deg) % 360`; it is visually inferred. Filename
distance units remain unconfirmed. Truck/Jeep body sizes used by SECTR-C are
simulation assumptions, not calibrated camera measurements.

The original [13-series study](../../docs/vehicle_scans/REPORT.md),
[embedding explorer](../../docs/vehicle_scans/embedding_offline.html) and
[single-object PCA](../../docs/vehicle_scans/single-object-pca/index.html) are
included. Original study accuracy results remain scoped to the first 13 series;
the ten added Jeep paint series use the same pinned model and fixed historical
PCA basis. Publication does not change the live SECTR-C tracker or camera stack.
