# MNIST ↔ CLIP Bayesian Uncertainty Experiment Plan

Below is a **compact, executable experiment plan** that reuses your MNIST↔CLIP setup to *estimate Bayesian predictive uncertainty* from a vision encoder. It includes (A) a stronger MCDO design (contrasted with your current practice), plus (B–E) alternative Bayesian heads you can stack on top, with pros/cons and caveats.

---

## 0) Goal & Constraints

- **Goal:** well-calibrated *posterior predictive* \(p(y \mid x, \mathcal D)\) and usable **epistemic** signals; separate **aleatoric** via input perturbations.
- **Keep:** your frozen **CLIP ViT-B/32** image tower and MNIST data. Your own results show (i) projection-head dropout amplifies epistemic spread and (ii) low-\(p\) with few passes is unstable, so we’ll adjust design accordingly.
- **Use** your perturbation catalogue (Gaussian noise, downsampling) to probe OOD and aleatoric effects; note your finding that predictive MI stayed flat due to prompt mismatch, motivating a supervised head.

---

## 1) Shared Setup (All Methods)

- **Data & splits:** MNIST train → head fitting; hold-out **val (10k)** for calibration; **test (10k)** for final metrics. Use your **noise/res** test variants for OOD.
- **Encoder:** frozen CLIP vision tower; extract L2-normalised 512-D features \((z=\phi(x))\).
- **Metrics (report all):**
  - **Calibration:** NLL, Brier, ECE (and reliability plots).
  - **Uncertainty quality:** predictive entropy, **BALD** MI, variation ratio; empirical **95% coverage** of credible intervals.
  - **OOD:** AUROC/AUPRC for “ID vs OOD” using uncertainty as score (noise \(\sigma \in \{0.05,0.10,0.20,0.35\}\), downsample \(\in \{24,20,16,12\}\)).
  - **Geometry diagnostics (embeddings):** **tangent-space trace/logdet**, and **resultant length (R)** (circular variance \(1-R\)) to avoid radial artefacts you observed.

---

## 2) Method A — MCDO (Recommended Encoder-Side Baseline)

### A.1 Design (what changes vs your runs)

- **Where:** enable dropout **only in the last 2–4 ViT blocks + the projection head** (your grid showed projection dropout boosts uncertainty volume).
- **How much:** default **p = 0.10**; add **p ∈ \{0.10, 0.05\}**.
- **Passes:** **T = 50** (≥ 50 if p = 0.01); your own analysis flagged **T = 16 too low** at small p.
- **Eval mode:** keep model in eval; *only* dropout stays stochastic (LayerNorm stats stay fixed; CLIP has no BN).
- **Head:** replace prompt head with a **supervised logistic head** on \(z\) (needed because your prior MI/entropy were saturated by prompt mismatch).
- **Geometry:** compute **tangent-space** covariance + \(R\); your note showed ~30% ambient→tangent trace drop (removes radial artefact).

### A.2 Inference

- Draw \(T\) masks → logits \(\{\ell_t\}\). Predictive probs \(\hat p = \frac{1}{T}\sum \text{softmax}(\ell_t)\).
- Report entropy \(H(\hat p)\), **BALD** \((H(\hat p) - \frac{1}{T}\sum H(p_t))\), and tangent metrics on embeddings.
- **Calibration:** 1-param **temperature scaling** on val.

### A.3 Why This MCDO Is Better

- Later-block + projection noise gives stronger, more class-relevant epistemic signal than encoder-only, consistent with your projection-gain deltas.
- Higher \(T\) stabilises variance at low \(p\) (your \(\sigma \approx\) mean issue at \(p=0.01\)).
- Tangent geometry avoids the spherical-norm artefact you quantified (~30% energy trimmed).

**Pros:** minimal code; runs on the frozen encoder; on-device friendly.  
**Cons:** can be over/under-confident; OOD not as strong as ensembles; uncertainty sensitive to dropout placement.  
**Caveats:** keep dropout only where trained/assumed; use ≥ T = 50 at low p; always report tangent metrics.

---

## 3) Method B — Bayesian Last Layer (Laplace / Neural-Linear)

- **Train:** fit logistic head \(w\) with L2 prior \((w \sim \mathcal N(0,\tau^{-1}I))\) on frozen \(z\).
- **Posterior:** Laplace at \(\hat w\): \(q(w) = \mathcal N(\hat w, (H + \tau I)^{-1})\) with GGN/KFAC or diag Hessian approx.
- **Predict:** sample \(w^{(s)}\), average \(p(y \mid z, w^{(s)})\). Optionally combine with MCDO by sampling masks *and* \(w^{(s)}\).

**Pros:** strong calibration gains with tiny overhead; complements MCDO.  
**Cons:** only Bayesian in the head; quality depends on curvature approx.  
**Caveats:** tune prior precision (\(\tau\)) on val; use 20–50 weight samples.

---

## 4) Method C — Deep Ensembles (of Heads)

- **Train:** \(K=5\) independent logistic heads on \(z\) (different seeds; optional bootstrap).
- **Predict:** average probabilities across heads; uncertainty via disagreement & MI.

**Pros:** best empirical OOD & calibration per cost; trivially parallel.  
**Cons:** \(K\times\) storage/inference for the head (still cheap vs encoder).  
**Caveats:** use temperature scaling post-hoc; combine with MCDO if you want encoder epistemics too.

---

## 5) Method D — SWAG / SGLD on the Head (Cheap Sampling)

- **SWAG:** collect head weights during late-training; fit low-rank + diag Gaussian; sample.
- **SGLD/SGHMC:** replace optimizer with sampler for the head only.

**Pros:** closer to Bayesian than plain point-estimate; low overhead.  
**Cons:** sampler/tail tuning matters; still head-only.  
**Caveats:** monitor effective sample size (ESS) and predictive convergence.

---

## 6) Method E — Deep-Kernel GP on Frozen Features

- **Train:** GP on \(z\) with RBF (or arc-cosine) kernel; use inducing points (e.g., \(M = 1{-}2\)k).
- **Predict:** exact GP posterior (approximate with inducing points).

**Pros:** principled posterior with calibrated uncertainty when data are modest; excellent small-data behaviour.  
**Cons:** memory/scaling; kernel choice & inducing-set selection matter.  
**Caveats:** standardise \(z\); cross-validate kernel hyperparams; report runtime.

---

## 7) Aleatoric vs Epistemic Split (Shared Add-On)

Use your augmentation dial to decompose variance (law of total variance): **epistemic** from MCDO with fixed input; **aleatoric** from variability across corruptions averaged over weights/masks. You already sketched this path in your geometry note; adopt that protocol and report \(\Sigma_{\text{total}} \approx \Sigma_{\text{epi}} + \Sigma_{\text{ale}}\).

---

## 8) Evaluation Matrix (Concise)

| Axis        | Settings                                                                       | Primary metrics                                                 |
| ----------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| **ID**      | MNIST clean                                                                    | NLL, Brier, ECE; accuracy                                       |
| **Corrup.** | noise \(\sigma \in \{0.05,0.10,0.20,0.35\}\); downsample \(\in \{24,20,16,12\}\) | AUROC (ID vs each corrupt level) using entropy/MI; coverage@95% |
| **Design**  | Dropout scope: {last-4+proj, last-2+proj, proj-only}; \(p \in \{0.10,0.05\}\); \(T \in \{30,50\}\) | ECE/NLL; tangent-trace/logdet; \(R\)                            |
| **Head**    | {Laplace, Ensemble(5), SWAG, GP} ± MCDO                                        | Same as above + runtime/throughput                              |

Reference expectations: projection dropout increases embedding variance and broadens distributions; noise inflates trace and reduces log-det (anisotropic spread); downsampling slightly shrinks variance—exact patterns you measured.

---

## 9) What “Success” Looks Like

- **ID calibration:** ECE ↓ vs plain head; **Laplace** or **Ensemble** beats MCDO-only.
- **OOD separability:** AUROC (ID vs noisy/low-res) ≫ 0.5; **Ensemble** ≥ **Laplace+MCDO** ≥ **MCDO-only**.
- **Geometry sanity:** tangent-trace rises under noise and falls under downsampling, reproducing your trends without radial bias.

---

## 10) Pros/Cons/Caveats (At a Glance)

| Method                            | Pros                                   | Cons                                                             | Caveats                                            |
| --------------------------------- | -------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------- |
| **MCDO (late-blocks+proj, T=50)** | Cheap, encoder-aware epistemic; easy   | Calibration/OOD weaker than ensembles; sensitive to p, placement | Use tangent metrics; T ≥ 50 at low p; calibrate    |
| **Laplace head**                  | Big calibration win; tiny cost         | Head-only Bayesian; curvature approx                             | Tune prior; 20–50 samples; combine with MCDO       |
| **Ensemble(5) heads**             | Strong OOD & calibration               | 5× head compute                                                  | Post-hoc temperature; combine with MCDO if desired |
| **SWAG/SGLD head**                | Lightweight posterior sampling         | Tuning; still head-only                                          | Track ESS; check convergence diagnostics           |
| **Deep-kernel GP**                | Principled posterior; great small-data | Scaling/memory                                                   | Use inducing points; kernel CV                     |

---

### Minimal Run Order (Fast → Strong)

1. **MCDO (last-4+proj, p = 0.10, T = 50) + temp scaling** → establish encoder-side epistemic.
2. **Add Laplace head** → fix calibration; re-measure OOD.
3. **Swap Laplace for Ensemble(5)** → target best OOD/calibration.
4. Optional: **SWAG** (if you want sampling without \(K>1\)) or **GP** (if training set small).

This plan preserves your existing pipeline, corrects the two key weaknesses you surfaced (insufficient \(T\) at low \(p\); ambient-space artefacts), and adds heads that deliver calibrated posteriors with clear compute/quality trade-offs.
