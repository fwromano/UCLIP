"""Summarise predictive uncertainty metrics from an MC Dropout run."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        if value.lower() == "nan":
            return float("nan")
        raise


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _pstdev(values: Sequence[float], mean: float | None = None) -> float:
    if not values:
        return 0.0
    if mean is None:
        mean = _mean(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or not x:
        return float("nan")
    mean_x = _mean(x)
    mean_y = _mean(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    return num / (den_x * den_y)


def summarise(run_dir: Path, top_k: int = 5) -> Dict:
    metrics_path = run_dir / "metrics.csv"
    rows: List[dict] = []
    with metrics_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_entropy = _safe_float(row.get("entropy", "nan"))
            mi = _safe_float(row.get("mutual_information", "nan"))
            aleatoric = total_entropy - mi
            trace = _safe_float(row.get("trace", "nan"))
            logdet = _safe_float(row.get("logdet", "nan"))
            offdiag = _safe_float(row.get("off_diag_mass", "nan"))
            confidence = _safe_float(row.get("confidence", "nan"))
            rows.append(
                {
                    "index": int(row["index"]),
                    "label": int(row["label"]),
                    "predicted": int(row["predicted"]) if row.get("predicted") else None,
                    "correct": row.get("correct") in {"True", "true", "1"},
                    "total_entropy": total_entropy,
                    "aleatoric_entropy": aleatoric,
                    "mutual_information": mi,
                    "trace": trace,
                    "logdet": logdet,
                    "offdiag": offdiag,
                    "confidence": confidence,
                }
            )

    totals = {
        "total_entropy": [row["total_entropy"] for row in rows],
        "aleatoric_entropy": [row["aleatoric_entropy"] for row in rows],
        "mutual_information": [row["mutual_information"] for row in rows],
        "trace": [row["trace"] for row in rows],
        "logdet": [row["logdet"] for row in rows],
        "offdiag": [row["offdiag"] for row in rows],
        "confidence": [row["confidence"] for row in rows],
    }

    summary = {
        "count": len(rows),
        "overall": {
            "total_entropy_mean": _mean(totals["total_entropy"]),
            "total_entropy_std": _pstdev(totals["total_entropy"]),
            "aleatoric_entropy_mean": _mean(totals["aleatoric_entropy"]),
            "aleatoric_entropy_std": _pstdev(totals["aleatoric_entropy"]),
            "mutual_information_mean": _mean(totals["mutual_information"]),
            "mutual_information_std": _pstdev(totals["mutual_information"]),
            "trace_mean": _mean(totals["trace"]),
            "trace_std": _pstdev(totals["trace"]),
            "logdet_mean": _mean(totals["logdet"]),
            "logdet_std": _pstdev(totals["logdet"]),
            "offdiag_mean": _mean(totals["offdiag"]),
            "offdiag_std": _pstdev(totals["offdiag"]),
            "confidence_mean": _mean(totals["confidence"]),
        },
        "correlations": {
            "trace_mutual_information": _pearson(totals["trace"], totals["mutual_information"]),
            "trace_total_entropy": _pearson(totals["trace"], totals["total_entropy"]),
            "confidence_total_entropy": _pearson(totals["confidence"], totals["total_entropy"]),
        },
    }

    per_label: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        label_stats = per_label[row["label"]]
        for key in ("total_entropy", "aleatoric_entropy", "mutual_information", "trace", "offdiag", "logdet"):
            label_stats[key].append(row[key])
        label_stats["correct"].append(1.0 if row["correct"] else 0.0)

    summary["per_label"] = {
        label: {
            "count": len(stats_dict["total_entropy"]),
            "total_entropy_mean": _mean(stats_dict["total_entropy"]),
            "mutual_information_mean": _mean(stats_dict["mutual_information"]),
            "aleatoric_entropy_mean": _mean(stats_dict["aleatoric_entropy"]),
            "trace_mean": _mean(stats_dict["trace"]),
            "offdiag_mean": _mean(stats_dict["offdiag"]),
            "logdet_mean": _mean(stats_dict["logdet"]),
            "accuracy": _mean(stats_dict["correct"]),
        }
        for label, stats_dict in per_label.items()
    }

    sorted_mi = sorted(rows, key=lambda row: row["mutual_information"], reverse=True)
    sorted_entropy = sorted(rows, key=lambda row: row["total_entropy"], reverse=True)

    def _strip(row: dict) -> dict:
        return {
            "index": row["index"],
            "label": row["label"],
            "correct": row["correct"],
            "predicted": row["predicted"],
            "total_entropy": row["total_entropy"],
            "aleatoric_entropy": row["aleatoric_entropy"],
            "mutual_information": row["mutual_information"],
            "trace": row["trace"],
            "offdiag": row["offdiag"],
            "confidence": row["confidence"],
        }

    summary["top_mutual_information"] = [_strip(row) for row in sorted_mi[:top_k]]
    summary["top_entropy_low_mi"] = [_strip(row) for row in sorted_entropy if row["mutual_information"] < 1e-6][:top_k]

    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise MC Dropout uncertainty metrics.")
    parser.add_argument("run_dir", type=Path, help="Directory containing metrics.csv")
    parser.add_argument("--out", type=Path, default=None, help="Optional path to write JSON summary.")
    parser.add_argument("--top", type=int, default=5, help="Number of examples to include in rankings.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = summarise(args.run_dir, top_k=args.top)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
