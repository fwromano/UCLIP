# Sim2 45° Noise Study — Critique & Recommendations

## Scope & Framing
- Goal is embedding uncertainty under MCDO; the report now avoids accuracy/confidence which were orthogonal. Keep MI only if we explicitly connect it to text prompts; otherwise frame MI as optional and focus on embedding-only metrics.
- State a single, crisp question upfront (e.g., “How do additive noise and resolution loss change the MCDO covariance of CLIP image embeddings across 8 viewpoints?”) and mirror it in the conclusions.

## Methodology Gaps
- Adapter scope: We enabled dropout on all 12 encoder blocks plus the projection head. Add a comparative grid (encoder-only vs encoder+projection) at p=0.01 to show how scope changes trace/logdet, similar to the MNIST grid.
- Dropout strength: Only p=0.01 was tested. Add p∈{0.02, 0.05} to assess sensitivity and whether MI rises with stronger stochasticity.
- Pass count stability: We used 64 passes. Include a brief T-sweep (e.g., T∈{32, 64, 128}) with error bars on trace/logdet to show estimator stability.
- Randomness control: Use 3–5 seeds for transforms and model sampling; include mean ± 95% CI bars for key metrics to show robustness.

## Perturbation Design
- Severity sweeps: Noise only includes σ∈{0.05,0.10} and downsampling only {128,64} px. Add several intermediate severities (e.g., σ 0.02–0.20, downsample 192,160,128,96,80,64) and plot metrics vs severity, not just category bars.
- Additional corruptions: Add Gaussian blur, JPEG compression, contrast/exposure shifts, small occlusions; these are common in downstream settings and often more realistic than salt/pepper.
- Rotation sensitivity: Dataset has 8 camera angles. Add per-angle summaries to test if uncertainty is periodic or orientation-dependent (0° vs 180°, etc.).

## Metrics & Analyses to Add
- Mean shift: Quantify drift of the stochastic mean under perturbations vs original deterministic embedding: ||μ_pert − μ_orig||₂ and Mahalanobis distance in original covariance. Report per-class and aggregated.
- Spectral shape: Plot eigenvalue spectra and spectral entropy; report top-k variance share (e.g., Σλ₁..λ₁₀ / trace). This shows whether variance concentrates into few modes.
- Cross-metric scatter: Add trace vs logdet and trace vs off-diagonal mass scatter per transform to visualise anisotropy changes.
- Tangent vs ambient: Save per-sample tensors (`--save-raw`) and run `uclip.analysis.geometry_analysis` to report tangent covariance trace, max eigenvalue, and circular variance. Include plots comparing ambient vs tangent trace (as in the MNIST geometry notes).
- Per-class distributions: Add violins/histograms for logdet and off-diagonal mass (not just trace) to mirror the MNIST “trace violin” breadth.

## Mutual Information (MI)
- Interpretation: MI remains ~2–3×10⁻⁵ across scenarios, i.e., extremely small. This likely reflects prompt unanimity and low predictive spread at p=0.01.
- Prompts: Current template yields awkward phrases (e.g., “a photo of a Indigo jeep” and non-color labels like “Moose3”, “Wolf2”). Improve prompts: “a photo of an indigo jeep”, “a photo of a white jeep”, or use neutral vehicle prompts when labels aren’t true attributes. Consider synonyms and multi-prompt averaging.
- Temperature: Sweep τ to test whether softer distributions surface MI structure (τ∈{0.7,1.0,1.3}).
- If embeddings-only: If classification isn’t in scope, move MI/entropy to an appendix and focus the main text on embedding dispersion metrics; optionally add embedding-only dispersion indices.

## Visualisations
- Severity curves: Replace categorical bar charts with line plots against severity (σ, target px). Add error bars (95% CI) from seed repeats.
- PCA views: Include 2D PCA of MCDO samples for a few representative images with covariance ellipses; overlay arrows from baseline μ to perturbed μ to show directional drift.
- Per-angle radar: A simple 8-slice radar or polar plot of mean trace by angle highlights orientation effects in one view.
- Legend & units: Ensure all plots have consistent legends, units, and scales; annotate percentage changes directly on plots where helpful.

## Reproducibility & Reporting Hygiene
- Include the exact CLI used (model id, adapter targets, p, T, seeds) and commit hash; note CPU/GPU and runtime. This makes the study self-contained.
- Persist `summary.json` keys explaining the adapter scope and p; the current summary includes adapter targets—good; keep that visible in the report text too.
- Save raw tensors for a subset (`--save-raw`) to enable geometry diagnostics and future drill-down without re-running everything.

## Actionable Next Steps
1. Add encoder-only vs encoder+projection comparison at p=0.01 and p=0.05; plot their deltas side-by-side.
2. Introduce severity sweeps and convert bar charts to severity curves with CIs.
3. Save raw per-sample tensors and add tangent-space metrics + a small PCA gallery with covariance ellipses.
4. Improve prompts (grammar + semantics) or move predictive MI to appendix if classification isn’t a goal.
5. Add per-angle analysis to determine viewpoint sensitivity.

These changes would bring the sim2 report to parity with, and in places beyond, the MNIST MCDO writeups, making the conclusions more robust and the uncertainty behaviour easier to interpret.

