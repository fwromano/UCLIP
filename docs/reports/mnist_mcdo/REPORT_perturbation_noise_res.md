# MNIST Perturbation Study — Noise & Resolution Effects on CLIP MCDO

## 1. Motivation & Context
- Builds directly on the dropout-instrumented MNIST study (`REPORT.md`), reusing `openai/clip-vit-base-patch32` with adapters on all 12 vision blocks plus the projection head (dropout `p = 0.1`).
- Objective: quantify how simple corruptions (additive Gaussian noise and loss of spatial resolution) reshape the Monte Carlo Dropout embedding cloud and predictive diagnostics.
- Scope: first 128 MNIST test images, 32 stochastic passes per image, temperature `τ = 1.0`. Predictive head enabled to monitor entropy, mutual information (MI), and top-1 confidence.

Command:
```bash
python -m mcdo.perturbation_study \
  --limit 128 \
  --passes 32 \
  --microbatch 4 \
  --noise-stds 0.0,0.05,0.1,0.2,0.35 \
  --downsample-sizes 24,20,16,12 \
  --out-root runs/mnist_perturbations_noise_res_dropout_v1 \
  --adapter-target visual_projection \
  --adapter-target vision_model.encoder.layers.0 \
  --adapter-target vision_model.encoder.layers.1 \
  --adapter-target vision_model.encoder.layers.2 \
  --adapter-target vision_model.encoder.layers.3 \
  --adapter-target vision_model.encoder.layers.4 \
  --adapter-target vision_model.encoder.layers.5 \
  --adapter-target vision_model.encoder.layers.6 \
  --adapter-target vision_model.encoder.layers.7 \
  --adapter-target vision_model.encoder.layers.8 \
  --adapter-target vision_model.encoder.layers.9 \
  --adapter-target vision_model.encoder.layers.10 \
  --adapter-target vision_model.encoder.layers.11
```
Outputs: scenario-specific metrics under `runs/mnist_perturbations_noise_res_dropout_v1/<scenario>/` and consolidated tables in the same root (`aggregated_metrics.{csv,json}`).

## 2. Perturbation Menu
- **Baseline**: raw MNIST digit, RGB converted by the CLIP processor.
- **Noise**: additive Gaussian noise injected in pixel space (σ ∈ {0.05, 0.10, 0.20, 0.35}) before CLIP preprocessing; values clipped to `[0,1]`.
- **Downsampling**: image shrunken to {24, 20, 16, 12} px and bilinearly upsampled back to 28 px to mimic resolution loss while keeping aspect ratio.

All scenarios reuse the same dropout instrumentation and random seed to isolate the perturbation effect.

## 3. Aggregate Metrics
*Baselines:* trace = 39.12, logdet = −6656.75, off-diagonal mass = 3426.30, MI ≈ 9.78e−6, top-1 confidence = 0.10091. Predictive accuracy remains ≈11.7% across perturbations (CLIP prompts are weak for MNIST), so emphasis is on covariance geometry.

### 3.1 Gaussian Noise Levels
| Noise σ | Trace | Δ Trace (%) | Logdet | Δ Logdet | Off-diag | Δ Off (%) | Confidence | Δ Conf (×1e-4) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 39.50 | 0.96% | -6656.9 | -0.15 | 3416.6 | -0.28% | 0.101002 | 0.92 |
| 0.10 | 39.68 | 1.42% | -6656.9 | -0.17 | 3435.1 | 0.26% | 0.101033 | 1.23 |
| 0.20 | 39.85 | 1.87% | -6657.0 | -0.21 | 3451.3 | 0.73% | 0.101043 | 1.34 |
| 0.35 | 39.92 | 2.04% | -6657.1 | -0.31 | 3456.5 | 0.88% | 0.101038 | 1.28 |

![Relative change in covariance metrics under Gaussian noise](../report_assets/mnist_mcdo/assets/noise_vs_variance.png)

### 3.2 Spatial Downsampling
| Downsample size | Trace | Δ Trace (%) | Logdet | Δ Logdet | Off-diag | Δ Off (%) | Confidence | Δ Conf (×1e-4) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 39.02 | -0.26% | -6656.9 | -0.18 | 3419.8 | -0.19% | 0.100958 | 0.48 |
| 20 | 38.99 | -0.33% | -6657.0 | -0.25 | 3422.6 | -0.11% | 0.100962 | 0.53 |
| 16 | 38.97 | -0.38% | -6657.1 | -0.32 | 3425.8 | -0.01% | 0.100966 | 0.56 |
| 12 | 38.96 | -0.42% | -6657.1 | -0.34 | 3432.0 | 0.16% | 0.100958 | 0.48 |

![Relative change in covariance metrics under spatial downsampling](../report_assets/mnist_mcdo/assets/downsample_vs_variance.png)

## 4. Key Findings
1. **Noise inflates the embedding cloud:** trace grows almost linearly with σ (≈+2% at σ=0.35), mirrored by rising off-diagonal mass (+0.9%), signalling broader and more correlated uncertainty ellipsoids. Log-volume (logdet) shrinks slightly, indicating variance concentrates along fewer axes as noise increases.
2. **Resolution loss mildly contracts variance:** downsampling suppresses trace by up to 0.42% and marginally lowers off-diagonal mass, suggesting the dropout-induced spread collapses when high-frequency detail is removed. The smallest 12 px case subtly re-inflates cross-dimension coupling (+0.16%), hinting that extreme blur introduces aliasing-like correlations. This aligns with the intuition that blurrier inputs present fewer stochastic paths for dropout to explore, so the induced embedding variance shrinks.
3. **Predictive signals are insensitive:** MI stays ∼1×10⁻⁵ and entropy ~2.3026 for all scenarios; top-1 confidence drifts by <1.4×10⁻⁴. Consistent with earlier work, CLIP text prompts for MNIST lack discriminative power, keeping accuracy ~random (11–12%) regardless of corruption.
4. **Digit-specific behaviour:** per-digit summaries (see `perturbation_summary.json`) show that noisy inputs inflate trace most for digits with loops (6,9), while aggressive downsampling suppresses variance for simpler shapes (1,7). These shifts align with the intuition that dropout emphasises spatial ambiguity amplified by noise and mitigated when fine detail vanishes.

## 5. Next Steps
- Inject other corruptions (e.g., elastic distortions, contrast shifts) to map a broader robustness surface.
- Couple the perturbation study with prompt engineering or linear probes to revive accuracy and examine whether uncertainty metrics become informative.
- Swap sigma schedules for depth-wise adapters to test whether higher dropout rates amplify corruption sensitivity.
