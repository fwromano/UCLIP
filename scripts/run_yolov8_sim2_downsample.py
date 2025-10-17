#!/usr/bin/env python
"""Run YOLOv8 detections on the Sim2 noise study dataset across downsample variants."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Configure Ultralytics to store settings within the repository (avoid $HOME writes).
def _prepare_ultralytics_settings(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ULTRALYTICS_SETTINGS", str(path))


def _import_yolo():  # noqa: D401 - lightweight wrapper for deferred import
    """Import YOLO after configuring environment."""
    from ultralytics import YOLO  # type: ignore

    return YOLO


# Reuse the same transform implementations as the main noise study script.
try:
    from scripts.run_sim2_noise_study import _DownsampleTransform, _PixelateTransform  # type: ignore
except ImportError as exc:  # pragma: no cover - fallback shouldn't trigger in repo
    raise SystemExit(
        "Unable to import downsampling transforms. Ensure the script is executed from the repository root."
    ) from exc


DOWN_SAMPLE_PERCENTS: Tuple[int, ...] = (1, 2, 5, 10, 20, 40, 60, 80, 85, 90, 93)
VEHICLE_CLASS_IDS = {2, 5, 7}  # COCO: car, bus, truck


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    kind: str


@dataclass
class DetectionRecord:
    image_relpath: str
    variant: str
    kind: str
    image_width: int
    image_height: int
    detection_count_all: int
    detection_count_vehicle: int
    best_vehicle_conf: float
    best_vehicle_area_frac: float
    best_vehicle_area_px: float
    best_vehicle_box_xyxy: Tuple[float, float, float, float]
    best_vehicle_cls: int


def iter_image_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            yield path


def apply_variant(image: Image.Image, spec: VariantSpec, transforms: Dict[str, object]) -> Image.Image:
    if spec.name == "original":
        return image
    transform = transforms.get(spec.name)
    if transform is None:
        raise KeyError(f"No transform configured for variant '{spec.name}'")
    return transform(image.copy())


def summarise_boxes(result, image_area: float) -> Tuple[int, int, float, float, float, Tuple[float, float, float, float], int]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or boxes.xyxy.shape[0] == 0:
        return 0, 0, math.nan, math.nan, math.nan, (math.nan,) * 4, -1

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    vehicle_mask = np.isin(cls, list(VEHICLE_CLASS_IDS))
    vehicle_indices = np.where(vehicle_mask)[0]
    if vehicle_indices.size == 0:
        return xyxy.shape[0], 0, math.nan, math.nan, math.nan, (math.nan,) * 4, -1

    # Pick the highest confidence vehicle detection.
    best_idx = vehicle_indices[np.argmax(conf[vehicle_indices])]
    x1, y1, x2, y2 = xyxy[best_idx].tolist()
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    area_frac = area / image_area if image_area > 0 else math.nan
    return (
        xyxy.shape[0],
        vehicle_indices.size,
        float(conf[best_idx]),
        area_frac,
        area,
        (float(x1), float(y1), float(x2), float(y2)),
        int(cls[best_idx]),
    )


def build_variants(downsample_percents: Sequence[int]) -> List[VariantSpec]:
    variants: List[VariantSpec] = [VariantSpec("original", "Original", "original")]
    for pct in downsample_percents:
        variants.append(
            VariantSpec(
                name=f"downsample_{pct}pct",
                label=f"Downsample {pct}%",
                kind="downsample",
            )
        )
        variants.append(
            VariantSpec(
                name=f"pixel_downsample_{pct}pct",
                label=f"Pixelated {pct}%",
                kind="pixel_downsample",
            )
        )
    return variants


def build_transforms(downsample_percents: Sequence[int], base_resolution: int = 224) -> Dict[str, object]:
    transforms: Dict[str, object] = {}
    for pct in downsample_percents:
        target = max(1, int(round(base_resolution * (1.0 - pct / 100.0))))
        transforms[f"downsample_{pct}pct"] = _DownsampleTransform(target_max_dim=target)
        transforms[f"pixel_downsample_{pct}pct"] = _PixelateTransform(target_max_dim=target)
    return transforms


def summarise_records(records: Sequence[DetectionRecord]) -> Dict[str, float]:
    n = len(records)
    vehicle_hits = sum(1 for r in records if r.detection_count_vehicle > 0)

    def _mean(values: Iterable[float]) -> float:
        vals = [v for v in values if not math.isnan(v)]
        return float(sum(vals) / len(vals)) if vals else math.nan

    def _std(values: Iterable[float]) -> float:
        vals = [v for v in values if not math.isnan(v)]
        if len(vals) < 2:
            return math.nan
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        return float(math.sqrt(var))

    best_conf_values = [r.best_vehicle_conf for r in records]
    area_frac_values = [r.best_vehicle_area_frac for r in records]

    return {
        "n_images": n,
        "vehicle_detection_rate": vehicle_hits / n if n > 0 else math.nan,
        "best_conf_mean": _mean(best_conf_values),
        "best_conf_std": _std(best_conf_values),
        "area_frac_mean": _mean(area_frac_values),
        "area_frac_std": _std(area_frac_values),
    }


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    settings_path = output_root / "ultralytics_settings.yaml"
    _prepare_ultralytics_settings(settings_path)

    YOLO = _import_yolo()
    model = YOLO(args.model)

    image_paths = list(iter_image_paths(data_root))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {data_root}")

    variants = build_variants(args.downsample_percents)
    transforms = build_transforms(args.downsample_percents, base_resolution=args.base_resolution)

    per_image_records: List[DetectionRecord] = []

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        relpath = str(image_path.relative_to(data_root))
        width, height = image.size
        image_area = float(width * height)

        for spec in variants:
            transformed = apply_variant(image, spec, transforms)
            results = model.predict(
                source=transformed,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
                save=False,
                device=args.device,
            )
            result = results[0]
            det_all, det_vehicle, best_conf, area_frac, area_px, box_xyxy, best_cls = summarise_boxes(
                result, image_area
            )
            per_image_records.append(
                DetectionRecord(
                    image_relpath=relpath,
                    variant=spec.name,
                    kind=spec.kind,
                    image_width=width,
                    image_height=height,
                    detection_count_all=det_all,
                    detection_count_vehicle=det_vehicle,
                    best_vehicle_conf=best_conf,
                    best_vehicle_area_frac=area_frac,
                    best_vehicle_area_px=area_px,
                    best_vehicle_box_xyxy=box_xyxy,
                    best_vehicle_cls=best_cls,
                )
            )

    # Aggregate per variant.
    summary_rows: List[Dict[str, object]] = []
    variant_groups: Dict[str, List[DetectionRecord]] = {}
    for record in per_image_records:
        variant_groups.setdefault(record.variant, []).append(record)

    for spec in variants:
        records = variant_groups.get(spec.name, [])
        metrics = summarise_records(records)
        summary_rows.append(
            {
                "variant": spec.name,
                "kind": spec.kind,
                "label": spec.label,
                **metrics,
            }
        )

    # Write outputs.
    per_image_csv = output_root / "per_image_results.csv"
    summary_csv = output_root / "variant_summary.csv"
    summary_json = output_root / "variant_summary.json"

    write_csv(
        per_image_csv,
        [
            "image_relpath",
            "variant",
            "kind",
            "image_width",
            "image_height",
            "detection_count_all",
            "detection_count_vehicle",
            "best_vehicle_conf",
            "best_vehicle_area_frac",
            "best_vehicle_area_px",
            "best_vehicle_box_xyxy",
            "best_vehicle_cls",
        ],
        [
            {
                **record.__dict__,
                "best_vehicle_box_xyxy": " ".join(f"{v:.2f}" for v in record.best_vehicle_box_xyxy)
                if not math.isnan(record.best_vehicle_box_xyxy[0])
                else "",
            }
            for record in per_image_records
        ],
    )

    write_csv(summary_csv, ["variant", "kind", "label", "n_images", "vehicle_detection_rate",
                             "best_conf_mean", "best_conf_std", "area_frac_mean", "area_frac_std"], summary_rows)

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w") as fh:
        json.dump(summary_rows, fh, indent=2)

    print(f"Wrote per-image results to {per_image_csv}")
    print(f"Wrote variant summary to {summary_csv}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/car_sim/sim2_cropped_45deg", help="Root folder with original images.")
    parser.add_argument("--output-root", default="runs/sim2_noise_study/yolov8", help="Directory for detections and summaries.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights path or model identifier.")
    parser.add_argument("--device", default="cpu", help="Torch device for inference (e.g. 'cpu' or 'cuda:0').")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for YOLO inference.")
    parser.add_argument(
        "--downsample-percents",
        type=int,
        nargs="*",
        default=list(DOWN_SAMPLE_PERCENTS),
        help="Downsample percentages to evaluate.",
    )
    parser.add_argument("--base-resolution", type=int, default=224, help="CLIP base resolution used for downsampling.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":  # pragma: no cover
    main()
