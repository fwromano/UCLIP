# Sim2 45° CLIP MCDO Noise Study

## 1. Introduction
We study how additive noise and progressive downsampling alter the geometry of CLIP image embeddings when Monte Carlo Dropout (MCDO) is applied to the vision tower. The goal is to quantify how perturbations change the stochastic embedding cloud (variance, orientation, and mean drift) so we can anticipate failure modes in downstream retrieval or classification settings.
All analyses focus on embedding-space diagnostics rather than classifier accuracy, mirroring a scientific report structure: we document the experimental configuration, define uncertainty metrics, and then interpret how those metrics respond to perturbations of increasing severity.

*Figure 1. Preview overview — all perturbations. Rows correspond to noise, smoothed crops, and raw crops respectively; columns step through increasing severity with the baseline at left.*

![Preview overview — all perturbations](../../report_assets/sim2_noise_study/modulation_overview_grid.png)

## 2. Methodology
- **Dataset:** data/car_sim/sim2_cropped_45deg
- **Model:** openai/clip-vit-base-patch32
- **Dropout p:** 0.01
- **Passes:** 64
- **Images:** 46
- **Dropout instrumentation:** DropoutAdapter wraps all twelve ViT encoder blocks plus the visual projection head (p = 0.01).
- **Sampling:** 64 stochastic passes per image (microbatch 4, deterministic seed). Deterministic baselines use a single pass with dropout disabled.
- **Perturbation sweep:** Gaussian noise σ∈{0.01,…,0.5} and downsampling up to 93% (224→16 px) followed by bicubic upsampling.
- **Pass-stability sweep:** Additional runs at T∈{2,4,8,16,32,64,128} for original, strongest-noise, and strongest-downsampling cases.

## 3. Metric Primer
- **Trace (Tr Σ):** Sum of covariance eigenvalues; measures total dispersion of the embedding cloud. Increases imply broader uncertainty.
- **Log-determinant (log det Σ):** Log-volume of the covariance ellipsoid. It captures how uncertainty spreads across dimensions; drops signal collapse into a lower-dimensional subspace.
- **Off-diagonal mass:** L₁ magnitude of non-diagonal covariance entries; indicates cross-dimensional coupling and anisotropy.
- **Anisotropy (corr-F):** Frobenius norm of the correlation matrix off-diagonals; values grow as variance concentrates into preferred axes instead of spreading uniformly.
- **Mean shift / Mahalanobis shift:** L2 distance and covariance-normalised distance between the stochastic mean and the deterministic embedding, quantifying drift induced by perturbations.
- **Tangent trace / λₘₐₓ:** Variance within the hypersphere orthogonal to the mean direction; highlights changes in directional uncertainty.
- **Spectral entropy & top-10 share:** Describe how variance concentrates across eigenmodes. Lower entropy or higher top-10 share means uncertainty is dominated by a few directions.
- **Circular variance:** Dispersion of unit-normalised samples; complements tangent analysis by examining orientation consistency.
- **Pass-count stability:** How rapidly trace/logdet estimates converge as the number of stochastic passes T increases.

## 4. Aggregate MCDO Embedding Metrics
Baseline trace is 12.31, with logdet -6376.03. Noise and crop-based degradations broaden the covariance while logdet declines; we report anisotropy using correlation-based Frobenius norms for scale-free comparison.

*Figure 2. Aggregate metrics overview. Aggressive downsampling emerges as the dominant driver of higher trace and lower log-volume across transforms.*

![Aggregate metrics overview](../../report_assets/sim2_noise_study/aggregate_overview.png)

*Figure 3. Mean shift across transforms. Mean embedding displacement mirrors the same ordering: raw (nearest) and smoothed (bicubic) crops introduce the largest drift from the deterministic baseline.*

![Mean shift across transforms](../../report_assets/sim2_noise_study/mean_shift_bar.png)

## 5. Deterministic Baseline (1 pass, dropout disabled)
With dropout disabled every run collapses to trace ≈ 5.12 × 10⁻⁴ and a singular covariance; logdet and anisotropy are therefore undefined.

## 6. Sensitivity to Noise
Trace grows steadily with σ; at σ=0.50 it reaches 12.80 (≈+4.0% vs baseline) while logdet shifts to -6383.93 (-7.89 vs baseline), signalling that variance expands yet concentrates into fewer dominant axes.

*Figure 4. Relative trace change under noise. Noise-only perturbations peak at +4.0% trace change when σ reaches its maximum setting.*

![Relative trace change under noise](../../report_assets/sim2_noise_study/noise_trace_relative.png)

*Figure 5. Logdet shift under noise. Log-volume steadily contracts with noise strength, finishing at -7.89 relative to baseline.*

![Logdet shift under noise](../../report_assets/sim2_noise_study/noise_logdet_relative.png)

*Figure 6. Trace vs noise severity. Trace increases monotonically with σ, indicating broader stochastic clouds as perturbations intensify.*

![Trace vs noise severity](../../report_assets/sim2_noise_study/noise_severity_trace.png)

*Figure 7. Logdet vs noise severity. Log-volume falls steadily with stronger noise, reflecting variance concentrating into fewer dominant modes.*

![Logdet vs noise severity](../../report_assets/sim2_noise_study/noise_severity_logdet.png)

*Figure 8. Correlation anisotropy vs noise severity. Cross-dimensional coupling grows gradually with σ, showing correlated drift rather than isotropic spread.*

![Correlation anisotropy vs noise severity](../../report_assets/sim2_noise_study/noise_severity_corr_anisotropy.png)

*Figure 9. Mean shift vs noise severity. Monte Carlo means drift further from the deterministic embedding as σ increases, tracking the speckle strength.*

![Mean shift vs noise severity](../../report_assets/sim2_noise_study/mean_shift_noise.png)

*Figure 10. Trace vs mean shift — noise severity. Trace and mean shift climb together with σ, illustrating that broader clouds coincide with larger centroid drift.*

![Trace vs mean shift — noise severity](../../report_assets/sim2_noise_study/trace_vs_meanshift_noise.png)

## 7. Sensitivity to Downsampling
Smoothed crops (bicubic upsample) beyond 60% sharply increase trace (e.g., 93% reduction → trace 12.83) while logdet becomes non-monotone—after a mild plateau it collapses to -6380.87 (volume ratio ≈ 0.008), consistent with aliasing and patch-token collapse once only a few pixels remain.

*Figure 11. Relative trace change under smoothed crop (bicubic). Smoothed crops peak around +4.2% trace change once spatial resolution drops to 93%.*

![Relative trace change under smoothed crop (bicubic)](../../report_assets/sim2_noise_study/downsample_trace_relative.png)

*Figure 12. Logdet shift under smoothed crop (bicubic). Covariance volume contracts by -4.84 at the harshest smoothed crop, underscoring aliasing-driven collapse.*

![Logdet shift under smoothed crop (bicubic)](../../report_assets/sim2_noise_study/downsample_logdet_relative.png)

*Figure 13. Trace distribution per transform. Trace variance fans out for the most aggressive downsampling settings, underscoring their broader uncertainty.*

![Trace distribution per transform](../../report_assets/sim2_noise_study/trace_violin.png)

*Figure 14. Trace vs downsampling severity. Trace rises almost linearly until about 60% reduction before the 93% case accelerates beyond +4.2%.*

![Trace vs downsampling severity](../../report_assets/sim2_noise_study/downsample_severity_trace.png)

*Figure 15. Logdet vs downsampling severity. Log-volume stays flat for mild reductions then plunges once images fall below 20% of their original side length.*

![Logdet vs downsampling severity](../../report_assets/sim2_noise_study/downsample_severity_logdet.png)

*Figure 16. Correlation anisotropy vs downsampling severity. Severe downsampling amplifies cross-dimensional coupling, showing the embedding cloud stretching along fewer axes.*

![Correlation anisotropy vs downsampling severity](../../report_assets/sim2_noise_study/downsample_severity_corr_anisotropy.png)

*Figure 17. Mean shift vs smoothed crop severity. Mean displacement spikes once resolution falls below 60%, matching the trace surge caused by aggressive smoothing.*

![Mean shift vs smoothed crop severity](../../report_assets/sim2_noise_study/mean_shift_downsample.png)

*Figure 18. Trace vs mean shift — smoothed crops. Smoothed crops show a coupled rise in trace and mean shift as detail disappears, linking spread and drift directly.*

![Trace vs mean shift — smoothed crops](../../report_assets/sim2_noise_study/trace_vs_meanshift_downsample.png)

*Figure 19. Logdet distribution per transform. The tail of extreme downsampling skews toward very low logdet values, reinforcing the volume collapse story.*

![Logdet distribution per transform](../../report_assets/sim2_noise_study/logdet_violin.png)

*Figure 20. Off-diagonal mass distribution per transform. Off-diagonal covariance mass swells as resolution drops, highlighting stronger axis entanglement.*

![Off-diagonal mass distribution per transform](../../report_assets/sim2_noise_study/offdiag_violin.png)

### Raw crop (nearest)
Keeping the raw nearest-neighbour crop (no smoothing) produces larger block artefacts; the harshest setting (pixel_downsample_93pct) yields trace 12.37 while logdet drops to -6382.75, indicating strong concentration of variance into a handful of axes.

*Figure 21. Relative trace change under raw crop (nearest). Trace rises steadily as the grid coarsens, confirming block artefacts inflate stochastic spread.*

![Relative trace change under raw crop (nearest)](../../report_assets/sim2_noise_study/pixel_trace_relative.png)

*Figure 22. Relative logdet change under raw crop (nearest). Log-volume decays more sharply once smoothing is removed, highlighting how raw crops squeeze variance into fewer modes.*

![Relative logdet change under raw crop (nearest)](../../report_assets/sim2_noise_study/pixel_logdet_relative.png)

*Figure 23. Trace vs raw crop severity. Nearest-neighbour crops reach +0.5% trace change once the short side is reduced by 93%.*

![Trace vs raw crop severity](../../report_assets/sim2_noise_study/pixel_severity_trace.png)

*Figure 24. Logdet vs raw crop severity. Logdet collapses to -6382.75 (-6.71 vs baseline) for the 93% setting, showing volume loss once smoothing is removed.*

![Logdet vs raw crop severity](../../report_assets/sim2_noise_study/pixel_severity_logdet.png)

*Figure 25. Correlation anisotropy vs raw crop severity. Raw crops amplify anisotropy faster than smoothed crops because hard edges align variance along token boundaries.*

![Correlation anisotropy vs raw crop severity](../../report_assets/sim2_noise_study/pixel_severity_corr_anisotropy.png)

*Figure 26. Mean shift vs raw crop severity. Nearest-neighbour reductions drive the largest centroid drift as aliasing artefacts dominate token activations.*

![Mean shift vs raw crop severity](../../report_assets/sim2_noise_study/mean_shift_pixel.png)

*Figure 27. Trace vs mean shift — raw crops. Raw crops yield the steepest trace/shift coupling, emphasising how aliasing inflates spread and drift together.*

![Trace vs mean shift — raw crops](../../report_assets/sim2_noise_study/trace_vs_meanshift_pixel.png)

## 8. Cross-Metric Geometry

*Figure 28. Logdet vs trace scatter. Smoothed crop points cluster in the high-trace, low-logdet corner, separating cleanly from the noise-induced shifts.*

![Logdet vs trace scatter](../../report_assets/sim2_noise_study/scatter_trace_logdet.png)

*Figure 29. Off-diagonal mass vs trace scatter. Off-diagonal coupling grows hand-in-hand with trace for the strongest smoothed crops, reinforcing anisotropy concerns.*

![Off-diagonal mass vs trace scatter](../../report_assets/sim2_noise_study/scatter_trace_offdiag.png)

## 9. Mean Shift Diagnostics
| transform | L2 shift | Mahalanobis shift |
| --- | --- | --- |
| original | 1.5934 ± 0.2906 | 806.9347 ± 97.6104 |
| gaussian_noise_010 | 2.7813 ± 0.6508 | 1666.2401 ± 255.5203 |
| gaussian_noise_020 | 3.6054 ± 0.6817 | 2227.3307 ± 358.3126 |
| gaussian_noise_050 | 4.3841 ± 0.6772 | 2788.7280 ± 342.4279 |
| gaussian_noise_100 | 5.1952 ± 0.9378 | 3367.2794 ± 419.5660 |
| gaussian_noise_200 | 6.5398 ± 1.0501 | 4228.4310 ± 535.9685 |
| gaussian_noise_500 | 8.7011 ± 0.8827 | 5537.6085 ± 582.0240 |
| saltpepper_5pct | 5.8839 ± 0.9645 | 3828.5899 ± 468.3757 |
| downsample_1pct | 2.9156 ± 0.7442 | 1763.3735 ± 465.4738 |
| pixel_downsample_1pct | 3.8731 ± 1.1712 | 2337.2949 ± 742.3540 |
| downsample_2pct | 2.9370 ± 0.7538 | 1767.1616 ± 482.2413 |
| pixel_downsample_2pct | 3.8726 ± 1.1516 | 2337.8885 ± 712.3581 |
| downsample_5pct | 3.0156 ± 0.7507 | 1832.2940 ± 465.8001 |
| pixel_downsample_5pct | 3.9265 ± 1.2046 | 2401.8979 ± 793.1497 |
| downsample_10pct | 3.1176 ± 0.7988 | 1908.4585 ± 494.0486 |
| pixel_downsample_10pct | 4.0919 ± 1.1790 | 2516.4781 ± 788.3843 |
| downsample_20pct | 3.3776 ± 0.8529 | 2091.0686 ± 520.2487 |
| pixel_downsample_20pct | 4.4315 ± 1.1498 | 2813.4398 ± 791.4507 |
| downsample_40pct | 4.0572 ± 0.8832 | 2561.1259 ± 518.2966 |
| pixel_downsample_40pct | 5.1926 ± 1.0419 | 3405.3319 ± 674.2687 |
| downsample_60pct | 5.1260 ± 0.8931 | 3215.4408 ± 481.4732 |
| pixel_downsample_60pct | 6.2757 ± 1.0407 | 4071.8815 ± 596.6683 |
| downsample_80pct | 6.9914 ± 0.7538 | 4419.8093 ± 534.7429 |
| pixel_downsample_80pct | 8.7442 ± 0.9495 | 5380.3463 ± 604.2322 |
| downsample_85pct | 7.7168 ± 0.6615 | 4915.8207 ± 549.4709 |
| pixel_downsample_85pct | 9.3134 ± 0.7255 | 5702.9129 ± 539.7879 |
| downsample_90pct | 8.8262 ± 0.6330 | 5610.9697 ± 584.7671 |
| pixel_downsample_90pct | 9.7077 ± 0.5933 | 6007.7963 ± 578.3062 |
| downsample_93pct | 9.3279 ± 0.6722 | 5948.2206 ± 621.6978 |
| pixel_downsample_93pct | 9.7685 ± 0.5089 | 6197.0443 ± 548.2255 |

*Figure 30. Mean shift vs noise severity. Mean L2 displacement tracks noise strength, reinforcing that stochastic means drift progressively under σ increases.*

![Mean shift vs noise severity](../../report_assets/sim2_noise_study/mean_shift_noise.png)

*Figure 31. Mean shift vs downsampling severity. Resolution loss beyond 60% triggers a rapid rise in mean shift, reflecting aliasing-induced drift.*

![Mean shift vs downsampling severity](../../report_assets/sim2_noise_study/mean_shift_downsample.png)

*Figure 32. Mean shift vs pixelated downsampling severity. Pixelation pushes mean drift even harder once large blocks emerge, underscoring the harsher aliasing penalty.*

![Mean shift vs pixelated downsampling severity](../../report_assets/sim2_noise_study/mean_shift_pixel.png)

*Figure 33. Mahalanobis shift vs noise severity. Covariance-normalised drift escalates with σ, showing that noise injects uncertainty aligned with high-variance directions.*

![Mahalanobis shift vs noise severity](../../report_assets/sim2_noise_study/mahal_shift_noise.png)

*Figure 34. Mahalanobis shift vs downsampling severity. Heavy downsampling rockets Mahalanobis distance, indicating the embedding cloud moves far relative to its contracted covariance.*

![Mahalanobis shift vs downsampling severity](../../report_assets/sim2_noise_study/mahal_shift_downsample.png)

*Figure 35. Mahalanobis shift vs pixelated downsampling severity. Nearest-neighbour reductions yield the largest covariance-normalised drift, showing block artefacts distort embeddings most severely.*

![Mahalanobis shift vs pixelated downsampling severity](../../report_assets/sim2_noise_study/mahal_shift_pixel.png)

## 10. Spectral & Tangent Geometry
| transform | tangent trace | tangent λmax | spectral entropy | top-10 share | circular variance |
| --- | --- | --- | --- | --- | --- |
| original | 8.5239 ± 1.6899 | 2.2936 ± 0.9171 | 2.5811 ± 0.2383 | 0.7565 ± 0.0436 | 0.0633 ± 0.0157 |
| gaussian_noise_010 | 8.7489 ± 1.7260 | 2.3569 ± 0.9274 | 2.5859 ± 0.2303 | 0.7557 ± 0.0424 | 0.0654 ± 0.0161 |
| gaussian_noise_020 | 8.7992 ± 1.6985 | 2.3006 ± 0.9398 | 2.6201 ± 0.2280 | 0.7518 ± 0.0420 | 0.0631 ± 0.0155 |
| gaussian_noise_050 | 8.5107 ± 1.6004 | 2.2645 ± 0.9120 | 2.5796 ± 0.2205 | 0.7574 ± 0.0400 | 0.0617 ± 0.0152 |
| gaussian_noise_100 | 8.4447 ± 1.5670 | 2.2813 ± 0.9015 | 2.5460 ± 0.2187 | 0.7620 ± 0.0389 | 0.0625 ± 0.0152 |
| gaussian_noise_200 | 8.2401 ± 1.3829 | 2.1899 ± 0.8128 | 2.5005 ± 0.2321 | 0.7685 ± 0.0389 | 0.0601 ± 0.0144 |
| gaussian_noise_500 | 7.7364 ± 1.4081 | 2.0397 ± 0.7857 | 2.3501 ± 0.2266 | 0.7948 ± 0.0384 | 0.0552 ± 0.0146 |
| saltpepper_5pct | 8.6093 ± 1.3903 | 2.2036 ± 0.8332 | 2.5686 ± 0.2253 | 0.7608 ± 0.0370 | 0.0626 ± 0.0142 |
| downsample_1pct | 8.6512 ± 1.6931 | 2.3227 ± 0.9141 | 2.5873 ± 0.2266 | 0.7572 ± 0.0406 | 0.0619 ± 0.0148 |
| pixel_downsample_1pct | 8.4435 ± 1.5675 | 2.2622 ± 0.8377 | 2.5307 ± 0.2185 | 0.7668 ± 0.0402 | 0.0624 ± 0.0152 |
| downsample_2pct | 8.6612 ± 1.6923 | 2.3173 ± 0.9090 | 2.5889 ± 0.2245 | 0.7570 ± 0.0402 | 0.0620 ± 0.0149 |
| pixel_downsample_2pct | 8.4704 ± 1.5897 | 2.2443 ± 0.8321 | 2.5379 ± 0.2177 | 0.7661 ± 0.0397 | 0.0625 ± 0.0153 |
| downsample_5pct | 8.6939 ± 1.6944 | 2.3229 ± 0.9144 | 2.5880 ± 0.2230 | 0.7574 ± 0.0400 | 0.0620 ± 0.0147 |
| pixel_downsample_5pct | 8.4340 ± 1.5407 | 2.2468 ± 0.8251 | 2.5309 ± 0.2111 | 0.7665 ± 0.0382 | 0.0622 ± 0.0147 |
| downsample_10pct | 8.6884 ± 1.7005 | 2.3288 ± 0.9098 | 2.5866 ± 0.2215 | 0.7574 ± 0.0397 | 0.0620 ± 0.0147 |
| pixel_downsample_10pct | 8.4308 ± 1.5376 | 2.2362 ± 0.8280 | 2.5284 ± 0.2178 | 0.7670 ± 0.0388 | 0.0621 ± 0.0150 |
| downsample_20pct | 8.7053 ± 1.6775 | 2.3396 ± 0.9162 | 2.5793 ± 0.2160 | 0.7591 ± 0.0387 | 0.0620 ± 0.0147 |
| pixel_downsample_20pct | 8.3116 ± 1.5453 | 2.2106 ± 0.8267 | 2.5163 ± 0.2115 | 0.7676 ± 0.0384 | 0.0615 ± 0.0151 |
| downsample_40pct | 8.8511 ± 1.6547 | 2.3567 ± 0.9101 | 2.5755 ± 0.2080 | 0.7617 ± 0.0360 | 0.0625 ± 0.0144 |
| pixel_downsample_40pct | 8.1782 ± 1.5154 | 2.1220 ± 0.7718 | 2.4896 ± 0.2002 | 0.7729 ± 0.0358 | 0.0608 ± 0.0148 |
| downsample_60pct | 8.9885 ± 1.5274 | 2.3026 ± 0.8298 | 2.5678 ± 0.1983 | 0.7650 ± 0.0341 | 0.0626 ± 0.0141 |
| pixel_downsample_60pct | 8.0991 ± 1.4896 | 2.0949 ± 0.7705 | 2.4697 ± 0.2094 | 0.7765 ± 0.0361 | 0.0588 ± 0.0142 |
| downsample_80pct | 9.0825 ± 1.4530 | 2.2022 ± 0.7401 | 2.5403 ± 0.2095 | 0.7721 ± 0.0330 | 0.0598 ± 0.0141 |
| pixel_downsample_80pct | 7.8798 ± 1.4360 | 2.0334 ± 0.7217 | 2.3964 ± 0.2044 | 0.7866 ± 0.0350 | 0.0553 ± 0.0138 |
| downsample_85pct | 8.8670 ± 1.4370 | 2.1181 ± 0.7449 | 2.5217 ± 0.2287 | 0.7726 ± 0.0357 | 0.0585 ± 0.0141 |
| pixel_downsample_85pct | 7.6474 ± 1.3430 | 1.9836 ± 0.6982 | 2.3743 ± 0.2179 | 0.7871 ± 0.0360 | 0.0542 ± 0.0134 |
| downsample_90pct | 8.6496 ± 1.3900 | 2.1075 ± 0.7272 | 2.5033 ± 0.2229 | 0.7787 ± 0.0332 | 0.0573 ± 0.0138 |
| pixel_downsample_90pct | 7.7745 ± 1.3452 | 2.0491 ± 0.6944 | 2.3728 ± 0.2128 | 0.7882 ± 0.0367 | 0.0547 ± 0.0128 |
| downsample_93pct | 8.2983 ± 1.4501 | 2.0909 ± 0.7254 | 2.4687 ± 0.2265 | 0.7821 ± 0.0337 | 0.0577 ± 0.0140 |
| pixel_downsample_93pct | 7.8503 ± 1.2907 | 2.1015 ± 0.6964 | 2.4103 ± 0.2141 | 0.7830 ± 0.0370 | 0.0558 ± 0.0130 |

*Figure 36. Trace by viewpoint angle. Flank viewpoints exhibit the largest trace gain once detail is stripped, pointing to orientation-sensitive uncertainty.*

![Trace by viewpoint angle](../../report_assets/sim2_noise_study/angle_radar.png)

*Figure 37. PCA embeddings — Original (baseline). Baseline embeddings remain compact with minimal directional bias.*

![PCA embeddings — Original (baseline)](../../report_assets/sim2_noise_study/pca_original.png)

*Figure 38. PCA embeddings — gaussian_noise_050 (noise min trace). Gaussian noise with σ=0.05 widens the cluster without shifting its centre markedly.*

![PCA embeddings — gaussian_noise_050 (noise min trace)](../../report_assets/sim2_noise_study/pca_gaussian_noise_050.png)

*Figure 39. PCA embeddings — gaussian_noise_500 (noise max trace). Gaussian noise with σ=0.50 widens the cluster without shifting its centre markedly.*

![PCA embeddings — gaussian_noise_500 (noise max trace)](../../report_assets/sim2_noise_study/pca_gaussian_noise_500.png)

*Figure 40. PCA embeddings — downsample_1pct (downsample min trace). Smoothed crop (bicubic upsample) at 1% reduction elongates the embedding cloud along a single axis as detail is lost.*

![PCA embeddings — downsample_1pct (downsample min trace)](../../report_assets/sim2_noise_study/pca_downsample_1pct.png)

*Figure 41. PCA embeddings — downsample_80pct (downsample max trace). Smoothed crop (bicubic upsample) at 80% reduction elongates the embedding cloud along a single axis as detail is lost.*

![PCA embeddings — downsample_80pct (downsample max trace)](../../report_assets/sim2_noise_study/pca_downsample_80pct.png)

*Figure 42. PCA embeddings — pixel_downsample_80pct (pixelated max trace). Pixelation at 80% produces discrete clusters as block artefacts dominate token activations.*

![PCA embeddings — pixel_downsample_80pct (pixelated max trace)](../../report_assets/sim2_noise_study/pca_pixel_downsample_80pct.png)

*Figure 43. PCA embeddings — pixel_downsample_93pct (pixelated min trace). Pixelation at 93% produces discrete clusters as block artefacts dominate token activations.*

![PCA embeddings — pixel_downsample_93pct (pixelated min trace)](../../report_assets/sim2_noise_study/pca_pixel_downsample_93pct.png)

## 11. Pass Count Stability
We evaluate trace and log-determinant stability across Monte Carlo pass counts T ∈ {2,4,8,16,32,64,128}.
Lower T increases estimator noise; curves should flatten as T grows. See stability plots in this section if generated.

*Figure 44. Trace stability vs passes. Trace spans shrink to 6.637 for the baseline sweep, while the harsh downsample stays within 5.915 once T≥32.*

![Trace stability vs passes](../../report_assets/sim2_noise_study/pass_stability_trace.png)

*Figure 45. Logdet stability vs passes. Logdet variance collapses as passes double, confirming that 64 draws are ample for stable volume estimates.*

![Logdet stability vs passes](../../report_assets/sim2_noise_study/pass_stability_logdet.png)

## 12. Detection-to-Embedding Pipeline (YOLOv8 Crop) and Camera Distance
Object scale shrinks as the camera moves away, mixing background into the crop and reducing effective resolution. To benchmark a practical workflow, we introduce a detection→crop→CLIP pipeline and study how distance-driven scale changes alter the embedding cloud.

### 12.1 Pipeline
- Detect vehicles in the full-resolution frame with YOLOv8 (COCO classes: car, bus, truck).
- Crop the predicted box, optionally expanding by ≈5% for context while staying inside image bounds.
- Resize the crop to CLIP's 224×224 input via either bicubic antialiased resampling or nearest-neighbour upsampling.
- Apply CLIP preprocessing and run MCDO (T=64, p=0.01) to estimate the stochastic embedding cloud.
- Compare crop embeddings to a reference (full frame or near-distance crop) using mean and Mahalanobis shift, trace/logdet, and anisotropy.

Notes:
- CLIP always consumes 224×224 inputs; “raw low-res” crops therefore require an upsample stage. Nearest-neighbour preserves the block grid, while bicubic smooths detail.
- Tiny crops widen trace and contract logdet, mirroring the high-severity downsampling response documented above.

### 12.2 Distance Effects (Expected/Observed)
- Smaller crops (greater distance) increase trace and Mahalanobis shift while logdet contracts.
- Mahalanobis shift reacts faster than raw mean shift because covariance volume shrinks as detail disappears.
- Raw (nearest) crops inject stronger anisotropy and larger shifts than smoothed (bicubic) crops at the same scale.

### 12.3 Practical Guidance
- Prefer bicubic antialiased resize before sending crops to CLIP; nearest-neighbour magnifies block artefacts.
- Maintain a minimum crop size by padding boxes with a narrow context band before resizing.
- Bin detections by object scale or distance and report mean/Mahalanobis shift, trace, logdet, and anisotropy per bin to visualise degradation.

## 13. Discussion & Outlook
- **High noise (σ=0.50)** expands trace by +4.0% and drops logdet by 7.89. The stochastic mean drifts 8.70 L2 units from the deterministic embedding, and tangent variance settles around 7.74, confirming broader but directional uncertainty.
- **Extreme downsampling (93% → 16px)** shifts trace by +4.2% and drops logdet by 4.84. Mahalanobis drift reaches 5948.2, while spectral entropy averages 2.47, signalling variance concentration into fewer modes.
- **Viewpoint sensitivity:** At 225°, trace shifts by +1.38 relative to the clean view, indicating that certain orientations (e.g. flank perspectives) become the most uncertain once detail is removed.
- **Pass-count stability:** Trace estimates converge within ±3.319 by 32 passes; high-noise and heavy-downsample settings widen the band only modestly, so T=64 remains a safe budget.
- **Predictive head:** Mutual information stays near 2.54e-05 for all conditions, so predictive entropy contributes little insight compared with embedding-space diagnostics.

## 14. Class-Level Trace Shifts
| class | trace (original) | trace (saltpepper_5pct) | Δ noise | trace (downsample_93pct) | Δ downsample |
| --- | --- | --- | --- | --- | --- |
| Indigo | 11.3669 ± 1.6120 | 11.7008 ± 1.3592 | 0.3339 | 12.2583 ± 1.2720 | 0.8914 |
| Magenta | 11.9280 ± 3.2723 | 12.0372 ± 3.4081 | 0.1091 | 12.6884 ± 3.7444 | 0.7603 |
| Moose3 | 12.6069 ± 3.3034 | 14.2672 ± 3.0083 | 1.6603 | 13.4894 ± 3.3945 | 0.8825 |
| White | 13.1045 ± 2.9819 | 12.9342 ± 2.4708 | -0.1703 | 13.2767 ± 2.5596 | 0.1722 |
| Wolf2 | 11.7160 ± 2.8515 | 12.8059 ± 2.5326 | 1.0899 | 12.2501 ± 2.4074 | 0.5341 |
| Yellow | 13.2319 ± 2.3014 | 13.0068 ± 2.1829 | -0.2251 | 13.1653 ± 2.5896 | -0.0666 |

## 15. Example Perturbations
Preview grids illustrate how each perturbation family reshapes the rendered jeep: baseline is shown alongside rising severities.

*Figure 46. Preview overview — all perturbations. Rows correspond to noise, smoothed crops, and raw crops respectively; columns step through increasing severity with the baseline at left.*

![Preview overview — all perturbations](../../report_assets/sim2_noise_study/modulation_overview_grid.png)

*Figure 47. Jeep previews — noise sweep. Gaussian noise progressively speckles the frame while salt & pepper corruption introduces isolated extreme pixels.*

![Jeep previews — noise sweep](../../report_assets/sim2_noise_study/noise_examples_grid.png)

*Figure 48. Jeep previews — smoothed crop sweep. Smoothing the coarse crop blurs structure as resolution falls, culminating in a soft, low-detail silhouette.*

![Jeep previews — smoothed crop sweep](../../report_assets/sim2_noise_study/downsample_examples_grid.png)

*Figure 49. Jeep previews — raw crop sweep. Nearest-neighbour reductions replace detail with large blocks, emphasising aliasing artefacts at coarse grids.*

![Jeep previews — raw crop sweep](../../report_assets/sim2_noise_study/pixel_examples_grid.png)

## Appendix: Detailed Metrics

### Aggregate metrics
| transform | trace | logdet | anisotropy (corr-F) |
| --- | --- | --- | --- |
| original | 12.3135 ± 2.6974 | -6376.0333 ± 10.0092 | 169.6903 ± 19.4235 |
| gaussian_noise_010 | 12.6023 ± 2.7577 | -6374.4188 ± 9.8618 | 169.3933 ± 18.9273 |
| gaussian_noise_020 | 12.5631 ± 2.6985 | -6373.7425 ± 9.3996 | 165.5048 ± 18.6221 |
| gaussian_noise_050 | 12.3926 ± 2.6261 | -6375.8614 ± 8.2435 | 166.7742 ± 18.1294 |
| gaussian_noise_100 | 12.4558 ± 2.5969 | -6376.6802 ± 7.1463 | 167.6568 ± 18.4162 |
| gaussian_noise_200 | 12.5629 ± 2.5728 | -6377.9926 ± 6.7516 | 166.6415 ± 19.0206 |
| gaussian_noise_500 | 12.8031 ± 2.7534 | -6383.9255 ± 6.6858 | 166.3067 ± 18.1931 |
| saltpepper_5pct | 12.7279 ± 2.5296 | -6375.3044 ± 6.3954 | 163.6120 ± 18.6287 |
| downsample_1pct | 12.3825 ± 2.6268 | -6375.9890 ± 9.1654 | 167.8046 ± 18.3878 |
| pixel_downsample_1pct | 12.5172 ± 2.6165 | -6377.2888 ± 8.4542 | 168.4906 ± 18.3043 |
| downsample_2pct | 12.3977 ± 2.6292 | -6375.7912 ± 9.2649 | 167.7650 ± 18.2637 |
| pixel_downsample_2pct | 12.5455 ± 2.6315 | -6376.9582 ± 8.7136 | 167.9582 ± 18.1520 |
| downsample_5pct | 12.4478 ± 2.6403 | -6375.7133 ± 9.0908 | 167.4770 ± 18.0066 |
| pixel_downsample_5pct | 12.5134 ± 2.5646 | -6377.1219 ± 8.2152 | 168.4439 ± 17.7280 |
| downsample_10pct | 12.4398 ± 2.6474 | -6375.7898 ± 9.0936 | 167.4970 ± 18.0020 |
| pixel_downsample_10pct | 12.5616 ± 2.6045 | -6377.0398 ± 8.0190 | 168.4702 ± 18.0116 |
| downsample_20pct | 12.4790 ± 2.6145 | -6375.9428 ± 8.7840 | 167.5528 ± 17.9934 |
| pixel_downsample_20pct | 12.4631 ± 2.5723 | -6377.8056 ± 7.6650 | 169.1383 ± 17.8052 |
| downsample_40pct | 12.6968 ± 2.5948 | -6375.4607 ± 8.3049 | 166.6609 ± 18.0419 |
| pixel_downsample_40pct | 12.4773 ± 2.5321 | -6378.9127 ± 8.2044 | 170.3822 ± 17.2721 |
| downsample_60pct | 13.0259 ± 2.5301 | -6374.7296 ± 7.5273 | 165.0812 ± 17.6869 |
| pixel_downsample_60pct | 12.5031 ± 2.5833 | -6380.0696 ± 7.6918 | 169.9147 ± 17.9723 |
| downsample_80pct | 13.5591 ± 2.5690 | -6374.3342 ± 6.5821 | 161.3558 ± 17.6353 |
| pixel_downsample_80pct | 12.6936 ± 2.6429 | -6382.2148 ± 7.4643 | 163.9830 ± 17.1538 |
| downsample_85pct | 13.5157 ± 2.6575 | -6375.1753 ± 6.4177 | 161.0859 ± 18.7373 |
| pixel_downsample_85pct | 12.4749 ± 2.5475 | -6383.4985 ± 7.2516 | 161.6255 ± 17.7324 |
| downsample_90pct | 13.2171 ± 2.5903 | -6377.9928 ± 6.2483 | 159.3951 ± 17.3251 |
| pixel_downsample_90pct | 12.5712 ± 2.5332 | -6383.3493 ± 6.4484 | 161.4009 ± 16.0536 |
| downsample_93pct | 12.8271 ± 2.6290 | -6380.8724 ± 6.6015 | 161.2282 ± 16.4835 |
| pixel_downsample_93pct | 12.3748 ± 2.4221 | -6382.7461 ± 6.0468 | 163.5343 ± 15.9229 |

### Noise severity deltas
| transform | Δ trace (%) | Δ logdet | Δ off-diag (%) |
| --- | --- | --- | --- |
| gaussian_noise_010 | 2.35 | 1.6145 | 1.71 |
| gaussian_noise_020 | 2.03 | 2.2908 | -1.95 |
| gaussian_noise_050 | 0.64 | 0.1719 | -3.46 |
| gaussian_noise_100 | 1.16 | -0.6468 | -3.42 |
| gaussian_noise_200 | 2.03 | -1.9593 | -5.86 |
| gaussian_noise_500 | 3.98 | -7.8921 | -11.82 |
| saltpepper_5pct | 3.37 | 0.7289 | -4.74 |

### Smoothed crop severity deltas
| transform | Δ trace (%) | Δ logdet | Δ off-diag (%) |
| --- | --- | --- | --- |
| downsample_1pct | 0.56 | 0.0444 | -1.76 |
| downsample_2pct | 0.68 | 0.2421 | -1.68 |
| downsample_5pct | 1.09 | 0.3200 | -1.61 |
| downsample_10pct | 1.03 | 0.2436 | -1.62 |
| downsample_20pct | 1.34 | 0.0905 | -1.52 |
| downsample_40pct | 3.11 | 0.5727 | -1.14 |
| downsample_60pct | 5.79 | 1.3038 | -0.81 |
| downsample_80pct | 10.12 | 1.6991 | -3.16 |
| downsample_85pct | 9.76 | 0.8580 | -4.94 |
| downsample_90pct | 7.34 | -1.9594 | -10.22 |
| downsample_93pct | 4.17 | -4.8391 | -12.43 |

### Raw crop severity deltas
| transform | Δ trace (%) | Δ logdet | Δ off-diag (%) |
| --- | --- | --- | --- |
| pixel_downsample_1pct | 1.65 | -1.2554 | -1.62 |
| pixel_downsample_2pct | 1.88 | -0.9249 | -1.73 |
| pixel_downsample_5pct | 1.62 | -1.0885 | -1.63 |
| pixel_downsample_10pct | 2.02 | -1.0065 | -1.52 |
| pixel_downsample_20pct | 1.22 | -1.7722 | -2.08 |
| pixel_downsample_40pct | 1.33 | -2.8794 | -2.58 |
| pixel_downsample_60pct | 1.54 | -4.0363 | -4.44 |
| pixel_downsample_80pct | 3.09 | -6.1815 | -11.66 |
| pixel_downsample_85pct | 1.31 | -7.4652 | -16.07 |
| pixel_downsample_90pct | 2.09 | -7.3160 | -15.49 |
| pixel_downsample_93pct | 0.50 | -6.7128 | -13.23 |

### YOLOv8 detection robustness vs downsampling
Using the Section 12 pipeline (YOLOv8n detection → crop → CLIP resize), we evaluated detections for every original frame and each smoothed (bicubic) / raw (nearest) crop variant. Vehicle detections cover COCO classes {car, bus, truck}. Detailed outputs live in `runs/sim2_noise_study/yolov8/variant_summary.csv` and the accompanying per-image CSV.
- Baseline detection succeeds on 43/46 views (93.5%), leaving 3 hard cases.
- Smoothed crop (bicubic) recall trends: 95.7% at 40% reduction, 82.6% at 60% reduction, 67.4% at 80% reduction, 21.7% at 90% reduction, and 4.3% at 93% reduction.
- Raw crop (nearest) recall drops to 82.6% at 10% reduction, 69.6% at 20% reduction, 54.3% at 40% reduction, and 2.2% at 60% reduction; detections vanish once reductions exceed 80%.
- Confidence slips from 0.61 (original) to 0.22 at 90% smoothed crop while boxes still span 94.1% → 90.2% of the frame.


| variant | detection rate (%) | mean best conf | mean box area (%) |
| --- | --- | --- | --- |
| original | 93.5 | 0.61 | 94.1 |
| downsample_1pct | 95.7 | 0.61 | 91.4 |
| downsample_2pct | 93.5 | 0.63 | 93.6 |
| downsample_5pct | 95.7 | 0.63 | 91.3 |
| downsample_10pct | 93.5 | 0.62 | 91.2 |
| downsample_20pct | 95.7 | 0.61 | 89.5 |
| downsample_40pct | 95.7 | 0.61 | 89.3 |
| downsample_60pct | 82.6 | 0.56 | 87.3 |
| downsample_80pct | 67.4 | 0.46 | 85.9 |
| downsample_85pct | 54.3 | 0.34 | 80.9 |
| downsample_90pct | 21.7 | 0.22 | 90.2 |
| downsample_93pct | 4.35 | 0.13 | 91.2 |
| pixel_downsample_1pct | 84.8 | 0.60 | 88.1 |
| pixel_downsample_2pct | 84.8 | 0.59 | 86.1 |
| pixel_downsample_5pct | 84.8 | 0.54 | 85.2 |
| pixel_downsample_10pct | 82.6 | 0.50 | 74.0 |
| pixel_downsample_20pct | 69.6 | 0.49 | 62.8 |
| pixel_downsample_40pct | 54.3 | 0.33 | 36.7 |
| pixel_downsample_60pct | 2.17 | 0.17 | 45.0 |
| pixel_downsample_80pct | 0.00 | — | — |
| pixel_downsample_85pct | 0.00 | — | — |
| pixel_downsample_90pct | 0.00 | — | — |
| pixel_downsample_93pct | 0.00 | — | — |


## Appendix: Predictive Diagnostics (MI & Entropy)
Predictive mutual information (epistemic) and entropy are derived from the CLIP text head over class prompts. They remain near-zero here due to prompt unanimity and low dropout; we include them for completeness.

*Figure 50. Mutual information across transforms. Predictive mutual information remains near zero for every perturbation, confirming the head stays confident despite dropout.*

![Mutual information across transforms](../../report_assets/sim2_noise_study/mi_line.png)

*Figure 51. Entropy across transforms. Predictive entropy barely moves across conditions, reinforcing that embedding diagnostics carry the informative signal.*

![Entropy across transforms](../../report_assets/sim2_noise_study/entropy_line.png)