## Car Sim Perturbation Study – Execution Plan

### Goal
Re-run the MNIST noise + downsampling robustness experiment on the `car_sim` dataset using the new CLIs, and collect artifacts for downstream analysis/reporting.

### 1. Run the perturbation sweep
- Command (assumes repo root, CPU execution):
  ```
  PYTHONPATH=src \
  python -m uclip.cli.car_sim_perturbation_study \
    --root data/car_sim \
    --out-root runs/car_sim_perturbations_noise_res_v1 \
    --passes 32 \
    --microbatch 4 \
    --limit 0 \
    --noise-stds 0.0,0.05,0.1,0.2,0.35 \
    --downsample-sizes 144,104,72
  ```
- Artefacts per scenario (baseline + corruptions) land in `runs/car_sim_perturbations_noise_res_v1/<scenario>/`.
- Each scenario directory contains:
  - `metrics.csv`: per-sample statistics (trace/logdet/off-diag, predictive metrics when enabled).
  - `summary.json`: aggregate stats per class and overall.
  - `individual/` (optional if `--save-raw`): tensors for deeper inspection.

### 2. Inspect aggregated outputs
- Primary table: `runs/car_sim_perturbations_noise_res_v1/aggregated_metrics.csv`.
- JSON mirror: `runs/car_sim_perturbations_noise_res_v1/aggregated_metrics.json`.
- Confirm coverage of all scenarios (baseline, 4× noise, 3× downsampling).
- Note: `car_sim` uses ImageFolder class ordering; check `summary.json["class_names"]` for mapping.

### 3. Sanity checks
- Verify runtime logs for warnings (especially CUDA fallback); adjust `--device` if GPU available.
- If sweep was interrupted mid-run, delete the target scenario directory and rerun to avoid mixing partial results.
- Optional: run a short smoke test before the full sweep, e.g. `--limit 32 --passes 8`, targeting a temporary output folder.

### 4. Reporting parity with MNIST study
- Mirror structure from `docs/reports/mnist_mcdo/REPORT_perturbation_noise_res.md`:
  - Describe dataset and perturbation settings (noise σ, downsample short-side sizes).
  - Summarize aggregate trends (trace/logdet/off-diagonal, predictive metrics if enabled).
  - Highlight class-level sensitivity using `summary.json` entries.
  - Reference plots or tables stored under a new `docs/report_assets/car_sim/...` directory.
- Capture open questions / next steps (e.g., extend to other corruptions, compare against MNIST behaviour).

### 5. Optional enhancements
- Enable `--save-raw` for deeper embedding diagnostics (beware of storage footprint).
- Supply custom prompts via `--prompt` flags if template-generated text underperforms.
- If GPU available, pass `--device cuda` for faster sweeps.

### Deliverables
1. Completed run directory `runs/car_sim_perturbations_noise_res_v1/` with full scenario coverage.
2. Derived figures/tables under `docs/report_assets/car_sim/`.
3. Written report (Markdown) summarising findings, analogous to the MNIST counterpart.

Document owner: Codex agent. Update this plan after major changes to workflow or tooling.
