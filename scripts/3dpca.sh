#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="runs/mnist_encoder_only_dropout_p001_T10_L1000"
OUT_DIR="reports/mnist_mcdo"

options=(
  "Digits 1 vs 7 (ambient, 4 samples each)"
  "Digits 3 vs 5 (ambient, 4 samples each)"
  "Top 10 ambient trace samples"
  "Digits 1 vs 7 (tangent, 4 samples each)"
  "Digits 3 vs 5 (tangent, 4 samples each)"
  "All digits (ambient)"
  "All digits (tangent)"
)

echo "Select interactive PCA variant:" >&2
PS3="Choice (1-${#options[@]}): "

select opt in "${options[@]}"; do
  case "$REPLY" in
    1)
      python -m mcdo.interactive_pca "$RUN_DIR" "$OUT_DIR/pca_digits1_7_interactive.html" \
        --samples-per-label 4 --labels 1,7 --title "Digits 1 vs 7 (interactive)" --open \
        --show-images --dataset-root data
      echo "Opened $OUT_DIR/pca_digits1_7_interactive.html"
      break
      ;;
    2)
      python -m mcdo.interactive_pca "$RUN_DIR" "$OUT_DIR/pca_digits3_5_interactive.html" \
        --samples-per-label 4 --labels 3,5 --title "Digits 3 vs 5 (interactive)" --open \
        --show-images --dataset-root data
      echo "Opened $OUT_DIR/pca_digits3_5_interactive.html"
      break
      ;;
    3)
      python - <<'PY'
import csv
from pathlib import Path
from mcdo.analysis import load_stats
from mcdo.interactive_pca import build_interactive_plot, _load_thumbnails

run_dir = Path("runs/mnist_encoder_only_dropout_p001_T10_L1000")
metrics_csv = run_dir / "metrics.csv"

rows = []
with metrics_csv.open() as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        rows.append(
            (
                float(row["trace"]),
                int(row["index"]),
                int(row["label"]),
            )
        )

rows.sort(reverse=True)
top = rows[:10]
indices = [idx for _, idx, _ in top]
labels = [label for _, _, label in top]

means, covariances = load_stats(run_dir, indices)
images = _load_thumbnails(indices, Path('data'))
build_interactive_plot(
    means,
    covariances,
    labels,
    indices,
    Path("reports/mnist_mcdo/pca_top10_trace_interactive.html"),
    "Top 10 trace samples (interactive)",
    auto_open=True,
    image_data=images,
)
print("Opened reports/mnist_mcdo/pca_top10_trace_interactive.html")
PY
      break
      ;;
    4)
      python -m mcdo.interactive_pca "$RUN_DIR" "$OUT_DIR/pca_digits1_7_tangent_interactive.html" \
        --samples-per-label 4 --labels 1,7 --title "Digits 1 vs 7 (tangent interactive)" \
        --open --show-images --dataset-root data --tangent
      echo "Opened $OUT_DIR/pca_digits1_7_tangent_interactive.html"
      break
      ;;
    5)
      python -m mcdo.interactive_pca "$RUN_DIR" "$OUT_DIR/pca_digits3_5_tangent_interactive.html" \
        --samples-per-label 4 --labels 3,5 --title "Digits 3 vs 5 (tangent interactive)" \
        --open --show-images --dataset-root data --tangent
      echo "Opened $OUT_DIR/pca_digits3_5_tangent_interactive.html"
      break
      ;;
    6)
      python -m mcdo.interactive_pca "$RUN_DIR" "$OUT_DIR/pca_all_ambient_interactive.html" \
        --title "All digits (ambient interactive)" --open
      echo "Opened $OUT_DIR/pca_all_ambient_interactive.html"
      break
      ;;
    7)
      python -m mcdo.interactive_pca "$RUN_DIR" "$OUT_DIR/pca_all_tangent_interactive.html" \
        --title "All digits (tangent interactive)" --open --tangent
      echo "Opened $OUT_DIR/pca_all_tangent_interactive.html"
      break
      ;;
    *)
      echo "Invalid selection" >&2
      ;;
  esac
done
