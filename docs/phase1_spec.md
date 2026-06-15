# Phase-1 Specification — Supervised Sparse-Tx FMC Reconstruction

> Single, implementable spec synthesized from the three design proposals and four
> adversarial critiques. Every API reference below is checked against the real
> `fmcfast` code (`forward.py`, `freqdomain.py`, `completion.py`, `baselines.py`,
> `sampling.py`, `tfm.py`, `metrics.py`, `geometry.py`). Fixed config (from
> `configs/default.yaml`): `N=64`, `pitch=0.5 mm` (=0.40·λ), `c=6300 m/s`,
> `f0=5 MHz`, `λ=1.26 mm`, `fs=50 MHz`, `n_t=1024`, `frac_bw=0.6` →
> **62 in-band rfft bins** (verified). Acceleration is reported **relative to HMC**:
> `accel = (N/K)/2`. So `K∈{16,8,6,4}` → **2×, 4×, 5.3×, 8× rel HMC**.

## 0. The one decision that frames everything (read first)

The four critiques converge on a single, verified, uncomfortable fact:

- **On the point-scatterer model, TFM *detection* of a strong defect is already
  saturated by the receive aperture.** Verified directly: an 8-defect scene at
  K=8/K=4, full-aperture SNR ≈ 20.4 dB, **naive sparse-Tx ≈ 20.6/20.7 dB**,
  best-tuned Nyström ≈ 20.0/17.7 dB. Naive (zero reconstruction) *matches full
  aperture* and *beats Nyström*. So "beat Nyström on strong-defect TFM SNR" is a
  **hollow target** — naive already wins it.
- Therefore Phase-1 is split into **two claims with different burdens of proof**:
  - **Claim A (primary, where a win is genuine): cube / M_f phase fidelity on the
    unobserved block** — the `band_nrmse` axis. Here naive does *nothing* (it
    leaves unobserved rows zero) and best-tuned Nyström genuinely degrades when
    `rank(M_f) > K`. This is the honest arena and the headline deliverable.
  - **Claim B (secondary, gated, must survive the critiques): downstream
    detectability on metrics with real headroom** — NOT strong-defect SNR, but
    **weak-near-floor POD@CFAR** and **closely-spaced two-defect resolution**,
    where full-aperture itself beats the cheap baselines by a wide margin.
- **The bar to beat is the STRONG classical frontier, never plain rank-4
  Nyström**: `{best-tuned per-bin Nyström (rank+rcond swept), joint
  cross-frequency low-rank, model-based steering-OMP, soft-impute}`. Plain rank-4
  Nyström is a strawman that fails for *tuning* reasons.
- **Honest pre-registered kill condition**: if the learned net cannot beat the
  *best of that frontier* on `band_nrmse` (unobserved block) by ≥2× at K≤8, the
  Phase-1 result is the **negative result** — at point-scatterer complexity the
  extra identifiable signal is fully captured by classical model-based estimators,
  and the case for learning waits for k-Wave (mode conversion, directivity,
  frequency-dependent scattering) where the analytic steering dictionary is
  mis-specified. We commit to publishing that negative result.

---

## 1. Data regime & generator

### 1.1 Why `rank(M_f)` is the whole game (verified)

Each point scatterer at `(x,z)` contributes, at every rfft bin, an exact rank-1
outer product `M_f = a aᵀ`, `a_i = (refl/√d_i)·exp(-iω d_i/c)`. Verified in-repo:
`n_def ∈ {1,3,8,12}` → numerical rank at the center band bin = `{1,3,8,12}`
(rcond 1e-3). Nyström (`completion.lowrank_complete`) is **exact iff
rank(M_f) ≤ K** and falls off a cliff when `rank > K`. So the *only* regime where
a learned (or any) method can add information to the unobserved block is
**`effective rank(M_f) > K` with exploitable structure**. We engineer the
generator to live there, and we *measure* rank per scene so we never claim a win
where one is information-theoretically impossible.

### 1.2 Phantom families (all built from the existing point-reflector API)

Every scene = point scatterers fed to `forward.simulate_fmc`, so the cube stays
exactly reciprocal and fast. We add a richer phantom builder (Section 9) but do
**not** touch the forward physics. **DECISION: the headline regime is COHERENT,
RESOLVABLE multi-defect / multi-facet structure** (rank-inflation that survives
TFM), with grain as a *nuisance* stressor only.
*Rationale:* critiques 1 and 4 verified that pure-grain rank inflation is a
"NRMSE mirage" — TFM averages incoherent clutter away, so it never shows up in
detection. Coherent rank>K is the only inflation that can move a downstream metric.

| Family | Description | rank lever | Role |
|---|---|---|---|
| **F1 Multi-SDH** | `n_def ~ U{3..12}` point SDHs, `refl ~ LogU(0.3,1.0)`, `x∈[-10,10] mm`, `z∈[10,32] mm`, **pairwise separation ≥ 1.3·λ = 1.64 mm** | rank = `n_def` exactly | **primary** (clean rank>K) |
| **F2 Tilted/branched crack** | `M=40-60` collinear points, length `L~U(3,10) mm`, tilt `θ~U(-45,45)°`, optional `|cos(θ-incidence)|` aperture taper. **Facet groups spaced ≥1.3·λ** | eff-rank 3–8 (space-bandwidth) | **primary** (manifold) |
| **F3 Closely-spaced pair** | exactly 2 SDHs, separation `Δ ~ U(0.5,1.5)·λ` (sub- to near-resolution), equal refl | rank=2 but TFM-unresolvable | **resolution headroom test** |
| **F4 Weak-defect + grain floor** | 1 strong SDH (refl 1.0) + 1 weak SDH (`refl ~ U(0.1,0.25)`) + `n_grain ~ U{20,40}` (`grain_refl 0.04±0.02`) + `noise_db ~ U(30,40)` | grain rank ~12-26 (nuisance) | **weak-POD headroom test** |

**Composition per phantom:** draw one of {F1, F2} as the coherent backbone, then
with prob 0.5 superimpose the F4 grain+noise floor. F3 and the pure-F4 weak-pair
are *evaluation-only* families (Section 6) chosen specifically because they have
**downstream headroom** (full-aperture beats naive by ≫3 dB), unlike strong-SDH SNR.

### 1.3 Why the constraints (each from a verified critique)

- **Separation ≥ 1.3·λ**: critiques 1/3 verified a dense collinear crack collapses
  to eff-rank 3–7 (correlated steering vectors), making Nyström fine again — the
  regime self-defeats unless facets are resolvable.
- **Grain is nuisance, not rank source**: verified — pure grain gives rank 25–31
  but TFM SNR barely moves; a net winning on NRMSE there wins nothing downstream.
- **Noise breaks exact reciprocity** (verified ~1.4% cube asymmetry at 40 dB).
  **DECISION: symmetrize the *measured* cube's M_f up front** (`freqdomain.symmetrize`
  per bin) before *any* method sees it, so observed entries are self-consistent and
  the free denoising is given *equally* to net and all baselines. This removes the
  "data-consistency vs reciprocity fight" the completeness critique flagged.

### 1.4 Speed-of-sound / OOD (iron law)

Train at `c=6300`. **DECISION: also randomize `c ~ U(6150,6450)` (±2.4%) during
training and pass `c` as a scalar conditioning input** (Section 2), because critique 2
verified the net's regression target moves ~137% under a +6.3% c-shift at fixed
geometry while Nyström is c-invariant. A net that memorizes one `c` is dead on the
mandated OOD. Held-out OOD test sets: `c∈{5900,6700}` (outside train range) and
`noise_db=28`. OOD is a **PASS/FAIL gate**, not an afterthought.

### 1.5 Effective-rank ledger (computed, logged per scene)

For every generated phantom we log, at 3 band bins (low/center/high): `eff_rank`
(SV count > 1e-3·σ₁), `σ_min/σ₁` vs noise floor `10^(-noise_db/20)`, parametric
DOF (`3·n_def`), and target `K`. Scenes are **binned by `eff_rank − K`**; we only
claim learning wins in the band `eff_rank ∈ [K+2, ~2K]` (identifiable but
rank-deficient for Nyström). `eff_rank ≫ 2K` is dropped from win-claims as
information-starved for *everyone*.

---

## 2. Representation & I/O tensors

**Domain (DECISION): per-frequency complex `M_f`, in-band bins only.** All three
proposals agree; out-of-band bins stay zero and `from_freq` restores the cube for
TFM, matching `baselines.reconstruct_lowrank` exactly.

**Bins: use ALL 62 in-band bins** (not subsampled). *Rationale:* the joint-LR
baseline and the net both rely on cross-frequency coupling; subsampling would
handicap the very mechanism we test. 62 bins is cheap at 64×64.

### 2.1 Tensors

Per phantom, build `Mf = to_freq(cube,fs)[:,:,band]` → `(64,64,62)` complex, then
`symmetrize` each bin. Draw `S = uniform_tx_set(64,K)`, `mask = observed_mask(64,S)`.

- **Input** `X`, real, shape `(B, C_in, 64, 64)` with the 62 bins folded into
  channels as interleaved (Re,Im): `C_in = 2·62 + 4 = 128`.
  - channels `0:124` = `[Re(M_obs_bin), Im(M_obs_bin)]` for the 62 bins, where
    `M_obs = mask ⊙ Mf` (unobserved entries zeroed).
  - channel `124` = binary `observed_mask` (float), broadcast over bins.
  - channel `125` = normalized speed of sound `(c - 6300)/300`.
  - channels `126,127` = normalized `(tx_idx, rx_idx)` coordinate ramps in `[-1,1]`
    (give convs absolute element-pair geometry; the steering phase is geometric).
- **Output** `Ŷ`, real, shape `(B, 124, 64, 64)` = `[Re,Im]` of the full `M_f` stack.

### 2.2 Normalization (DECISION: robust per-phantom global complex scale)

`s = median over band of ||M_obs[bin]||_F` computed **only on observed entries**
(leakage-safe — uses only measured rows/cols). Divide input by `s`; multiply output
by `s` before loss/TFM. **Median (not RMS)** so a strong back-wall/primary defect
does not swamp a weak defect's scale (proposal-1 risk). **Never per-bin normalize**
— it destroys the cross-bin phase coupling the net must exploit.

### 2.3 Hard constraints (zero-param output layers, cannot be violated)

Applied in this order on the raw network output, per bin:
1. **Reciprocity** `Ŷ ← 0.5·(Ŷ + Ŷ.transpose(tx,rx))` on Re and Im independently
   (complex-symmetric, matches `freqdomain.symmetrize`; transpose is real-linear so
   commutes with the Re/Im split).
2. **Data consistency** `Ŷ ← mask⊙M_obs + (1-mask)⊙Ŷ`. Observed rows/cols are
   passed through exactly (they are the symmetrized measurements from §1.3), so the
   net only ever predicts the **unobserved block** — exactly Nyström's sub-problem,
   and it is structurally never worse than naive on measured entries.

**DECISION: randomize `S` per sample in training** (random K-subsets including the
uniform one), evaluate on the canonical `uniform_tx_set`. *Rationale:* critique 2
verified Nyström is mask-agnostic (identical error across masks, zero training);
the net must *learn* mask-invariance or it overfits one geometry. This refines the
plan's "fixed uniform" to "fixed-uniform at test, randomized at train" — same
deployed problem, no memorized geometry.

---

## 3. Model — v1 architecture

**DECISION: a small real-valued 2-channel-per-bin 2D U-Net over the (tx,rx)=(64,64)
element-pair plane, 62 band bins as feature channels, FiLM-conditioned on `c`, one
self-attention bottleneck block, wrapped in the §2.3 hard projection.** All three
proposals converged here; keep it simple per the plan.

*Rejected:* 3D U-Net over (tx,rx,freq) — bins are near-redundant smooth phase ramps,
a `c`-scalar + channel mixing handles freq more cheaply (v1.1 upgrade if it
underperforms); unrolled proximal-gradient — needs the sampling operator in the
loop, 3-5× compute, violates "simple first" (great v2); full N²-token attention —
O(N⁴), unnecessary.

### 3.1 Layer sketch

```
Input  (128, 64, 64)
Stem   Conv2d 128->64, 3x3
Enc1   2×[Conv3x3 64  + GroupNorm(8) + GELU] ; down s2  -> (64, 32,32)
Enc2   2×[Conv3x3 128 + GN + GELU]           ; down s2  -> (128,16,16)
Enc3   2×[Conv3x3 256 + GN + GELU]           ; down s2  -> (256, 8, 8)
Bott   2×[Conv3x3 256 + GN + GELU]
       + 1× MultiHeadSelfAttention(4 heads, 64 tokens=8x8, dim 256)  # global element coupling
       + Conv2d 1x1 256->256                  # cross-bin channel mixer
Dec3   upsample/convT -> 256, skip Enc3, 2×conv -> (256,16,16)
Dec2   upsample/convT -> 128, skip Enc2, 2×conv -> (128,32,32)
Dec1   upsample/convT -> 64 , skip Enc1, 2×conv -> (64, 64,64)
Head   Conv2d 64->124, 1x1                    # Re/Im of 62 bins
-> symmetrize  -> data-consistency  -> ×s (denorm)
FiLM   tiny MLP (1->2·Σchannels) from c-scalar injects (γ,β) per stage
```

### 3.2 Footprint

~10–13M params (256-ch convs dominate: `256·256·9 ≈ 0.59M` each × ~8 + attention +
FiLM). FP32 weights ≈ 50 MB. At batch 16: largest activation `16·64·64·64·4B ≈
67 MB`/layer, peak with skips ≈ 1–2 GB; weights+Adam ≈ 0.3 GB. **Total < 4 GB on
one Quadro GP100 (16 GB)** with wide headroom. **Train FP32** — Pascal sm_60 has
weak FP16 and no tensor cores; GP100's FP64 strength is irrelevant here. Second
GP100 = DDP to halve wall-clock or run a K=4 specialist.

---

## 4. Loss

All terms **phase-sensitive**, on the post-symmetrize / post-data-consistency /
denormalized output, computed **only on the unobserved block** `U = (1-mask)` (the
observed entries are pinned exact, so including them just dilutes the gradient).
**No envelope/`abs` term anywhere** (the explicit forbidden mode).

```
L = 1.0·L_cube + 0.3·L_rf + 0.1·L_tfm
```

- **`L_cube`** (primary, = training analog of `metrics.band_nrmse` on `U`):
  `|| U⊙(Ŷ - Mf_true) ||_F / || U⊙Mf_true ||_F`, complex (Re+Im jointly), over the
  62 in-band bins. Weight **1.0**. This is exactly the Claim-A headline axis.
- **`L_rf`** (time-domain phase anchor): `from_freq(Ŷ, n_t)` and `from_freq(Mf_true,
  n_t)` (out-of-band zero), L2 on the unobserved rows of the RF cube, relative.
  Penalizes arrival-time/pulse-shape error the band-edge frequency L2 underweights.
  Weight **0.3**.
- **`L_tfm`** (differentiable-TFM consistency, **ramped in after epoch 5**): a torch
  port of `tfm.tfm_image`'s delay-and-sum, L2 on the **complex coherent beamformed
  field BEFORE `abs`** (keep phase). Weight **0.1**. *Critical implementation note
  (completeness critique):* `tfm_image` uses `scipy.signal.hilbert` (analytic
  signal) when `analytic=True`; the subtle part is the FFT-based Hilbert step, not
  just the linear-interp gather. **Use `analytic=False`** in the loss (operate on the
  real reconstructed cube's coherent sum) to avoid the Hilbert mismatch, and
  **unit-test the torch TFM against `tfm_image(analytic=False)` to <1e-4** before
  enabling. TFM targets are fixed per phantom → **cache them** (≈440 ms/image
  measured; do not recompute every step).

*Rationale for weights:* `L_cube` carries the headline signal in the exact metric
space; `L_rf` is the mandated phase regularizer; `L_tfm` gently steers capacity
toward the entries TFM coherently sums, but is cheap-warmed-up to avoid starving
the GPU on CPU-bound TFM (completeness critique verified the cost).

---

## 5. Training

### 5.1 Data: on-the-fly with a measured cost budget

`simulate_fmc` is **not uniformly "ms-scale"** — verified: single-SDH ≈ 38 ms/cube,
but **12-defect ≈ 283 ms/cube** (Python loop over `n_tx` in `forward.py`). With 8
DataLoader workers ≈ 28 hard cubes/s → a 5k-phantom epoch ≈ 3 min pure gen. **This
holds *only if* `L_tfm` targets are cached** (else +440 ms/scene starves the GPU).
**DECISION: generate on-the-fly** (infinite fresh grain/geometry, no leakage), but
**freeze val/test/OOD to disk** as in-band `M_f` stacks (`62·64·64` complex64 ≈
2 MB each; 512 test + 256 val ≈ 1.5 GB) for reproducible scoring. If the GPU starves
in practice, fall back to a precomputed phantom-split corpus of M_f stacks (still
2 MB each, ~20 GB for 10k) rather than fighting the dataloader.

### 5.2 Splits — by configuration (iron law, never by (tx,rx) pair)

Disjoint RNG seed ranges; the whole cube of a phantom always lands on one side.
- **train**: seeds `[0, 1e6)`, families F1/F2 (+F4 floor 50%), `c~U(6150,6450)`.
- **val (in-dist)**: seeds `[1e6, 1.1e6)`.
- **test-ID**: seeds `[1.1e6, 1.2e6)`.
- **test-OOD-c**: `c∈{5900,6700}` only (physical OOD, the iron law).
- **test-OOD-geom**: `n_def∈{16..20}` or crack lengths/types outside train.
- **eval-only headroom families**: F3 (closely-spaced pair), F4 weak-pair — frozen
  sets where full-aperture provably beats naive (Section 6).

`K` is part of the input mask, randomized per sample over `{16,8,6,4}` in training
(one mask-conditioned model serves the whole acceleration sweep); evaluate per-K.

### 5.3 Optimizer / schedule / batch

AdamW, lr 3e-4 → cosine decay to 1e-5, 500-step warmup, weight_decay 1e-4,
grad-clip 1.0. Batch **16** phantoms. ~80k–150k steps. AMP **off** (FP32). Checkpoint
best **val `L_cube` on the unobserved block** (the Claim-A metric).

### 5.4 Wall-clock

Bottleneck is CPU data-gen, not the 10–13M-param GPU forward/backward. ~3–6 min/epoch
on one GP100 with 8 workers + cached TFM targets → **~3–6 h single-GPU, ~2–3 h on
2× GP100 DDP** — "hours not days" as required. Smoke-test the full loop on Mac
(torch CPU/MPS) at batch 2, 50 steps, before pushing to the server (conda env
`pytorch`). Sync via the established Mac↔GitHub↔server loop.

---

## 6. Evaluation — learned vs naive vs STRONG low-rank/CS

### 6.1 Baselines (the bar — make every one as strong as honestly possible)

The plan's plain rank-4 Nyström is a **strawman** (verified: it loses to its own
oracle-rank version by 5–7 dB). The honest frontier, all added to `baselines.py`:

1. **Naive** (`reconstruct_naive` / `tfm_image(tx_set=S)`) — first-class baseline on
   *detection*, because verified to match full-aperture on strong-SDH SNR.
2. **Best-tuned per-bin Nyström** — `lowrank_complete` with **rank swept
   `{2,4,8,12,16}` and `rcond` swept `{1e-8,1e-4,1e-2}` per (K, scene)**, report the
   **best** (oracle-tuned; never crippled).
3. **Joint cross-frequency low-rank** — soft-impute / SVT on the unfolded band stack
   (62 bins stacked), exploiting the verified 0.889 adjacent-bin subspace overlap.
   **This is the true ceiling for the "cross-frequency coupling" argument** — if the
   net cannot beat THIS, that prior is not learning-unique.
4. **Model-based steering-OMP** — joint-over-bins OMP on the analytic steering
   dictionary `exp(-iω d/c)/√d`. *Caveat (critique 4): on the idealized point model
   this is oracle-physics CS and near-unbeatable — its purpose is to **bound** the
   in-distribution win and to be the method that *fails* under k-Wave dictionary
   mismatch.* Report it; do not pretend to beat it on the point model.
5. **Soft-impute (per-bin nuclear norm)** and **bilinear row-interpolation** — cheap
   CS-style controls.

### 6.2 Metrics

**Reconstruction (Claim A, headline):** `metrics.band_nrmse`, evaluated **on the
unobserved block** and overall. This is where naive does nothing and Nyström breaks.

**Detection (Claim B, gated) — FIX THE METRIC FIRST:** the existing
`metrics.defect_metrics` is **single-defect-only and returns garbage on multi-defect
scenes** (verified: `artifact_db = +4.5 dB` positive, because neighbor defects are
counted as "background"). **DECISION: add a multi-target CFAR detector + POD/FA
evaluator** (Section 9). It runs `tfm_image`, CFAR peak-picks, matches peaks to the
known scatterer list within tolerance, returns per-image (TP, FA) and per-matched
`{snr_db, lat6_mm, ax6_mm}`, and aggregates **POD@fixed-FA** (e.g. 1 FA/image) across
≥200 test phantoms. Report detection **only on headroom families** (F3 resolution,
F4 weak-near-floor), never on strong-SDH SNR (saturated — cannot separate methods).

**Info-deficit guard:** per scene log the full-aperture reconstruction error floor
and `eff_rank` vs `K`, so "net can't recover it" is distinguished from "nobody can."

### 6.3 Protocol — quality-vs-acceleration curve

For each test phantom: full-aperture reference TFM (gold), then reconstruct at
`K∈{16,8,6,4}` (= 2×,4×,5.3×,8× rel HMC) with each method. X-axis = accel rel HMC;
Y-axis = median over ≥200 phantoms of `{band_nrmse(U), POD@1FA, two-defect resolution
rate, lat6/ax6 error}` with **bootstrap CIs**; **paired per-phantom** comparisons
(same scene, different method) to cut variance. Separate curves for test-ID,
test-OOD-c, test-OOD-geom, and **stratified by `eff_rank − K`** so the rank>K
crossover is visible. **Out-of-band handling identical for all methods** (zero the
451 out-of-band bins before `irfft`) so the comparison is apples-to-apples.

### 6.4 Ablations (attribute any gain honestly)

(a) remove reciprocity hard layer; (b) remove data-consistency projection;
(c) `L_cube` only vs +`L_rf` vs +`L_tfm`; (d) per-bin norm vs robust global norm;
(e) U-Net vs per-row MLP; (f) **remove cross-frequency mixing** (isolates the
coupling prior — if this closes the gap to joint-LR, coupling was *not*
learning-unique); (g) **train single-SDH, test multi-facet** (should FAIL → proves
the net needs the manifold); (h) test-time mask-shuffle (should be robust);
(i) with/without `c`-conditioning under OOD-c (isolates c-fragility); (j) single-K
vs curriculum/uniform-K sampling.

---

## 7. Falsifiable success criteria

**"Learning beats low-rank" is declared (PASS) iff ALL hold:**

- **A1 (headline, must hold):** on test-ID in the band `eff_rank ∈ [K+2, 2K]` at
  K≤8, learned `band_nrmse` on the unobserved block is **≥2× lower than the best of
  {oracle Nyström, joint cross-freq LR, soft-impute}**, with non-overlapping
  bootstrap CIs. (Steering-OMP excluded as oracle-physics on the point model.)
- **A2 (OOD gate):** on test-OOD-c, the net does **not** collapse below the best
  classical baseline; `band_nrmse(U)` stays within 1.5× of its test-ID value.
- **B1 (downstream, headroom families only):** on F4 weak-near-floor at K=8, learned
  **POD@1FA ≥ 0.95 where best-classical POD < 0.90**, with **zero new false alarm
  above −6 dB** and no defect-SNR regression vs full-aperture > 3 dB.
- **B2 (resolution, headroom family):** on F3 closely-spaced pairs at `Δ∈[0.5,1.0]·λ`,
  learned two-defect resolution rate exceeds best-classical by ≥10 pp.

**KILL CONDITION (report the negative result, per project rule):** if **A1 fails** —
i.e. the best of the *strong* classical frontier (especially joint cross-frequency
LR) matches the net on unobserved-block `band_nrmse` within CI at every K where the
net is usable — then at point-scatterer complexity the learned method adds nothing,
and the deliverable is the **frontier map + negative result**: "naive captures
strong-defect detection; classical model-based estimation captures the identifiable
rank>K signal; learning's edge requires k-Wave-class dictionary mismatch." This is a
first-class, publishable outcome.

---

## 8. Honest risks & the single most likely negative result

**Most likely negative result (high confidence, verified in critiques):** *The net
wins decisively on `band_nrmse` (Claim A) but that win does NOT translate to a
downstream detection advantage on the point model, because (i) TFM coherent sum is
dominated by the observed rows the data-consistency layer pins exactly, and (ii) the
identifiable rank>K signal is **equally recoverable by joint cross-frequency LR and
steering-OMP** with no training.* In that case A1 may still pass against per-bin
Nyström/soft-impute but **fail or tie against joint-LR**, and B1/B2 may show no gap.
The framing then becomes: *Claim A holds for downstream re-processing (alternate
beamformers, sizing) that weight the Tx block; Claim B does not, on the point model;
both require k-Wave to become a detection-relevant win.*

Other risks (mitigations baked into the spec above):
- **Steering-OMP oracle problem** — exact analytic dictionary makes it near-unbeatable
  on the point model. Mitigation: report it as a *bound*, not a beat-target; the
  real test is k-Wave (Phase-2 dependency, flagged as THE precondition for a positive
  detection result).
- **c-fragility** — the identifiability-restoring steering manifold degrades at ~1.6%
  c-error for net *and* model-based baselines. Mitigation: c-conditioning + OOD-c
  PASS/FAIL gate (A2).
- **Info-deficit at high accel** — `eff_rank ≫ K` is unrecoverable for everyone.
  Mitigation: the eff-rank ledger drops those scenes from win-claims.
- **Data-gen starvation** — verified 283 ms/hard-cube + 440 ms/TFM. Mitigation: cache
  TFM targets, 8 workers, fall back to precomputed M_f corpus.
- **Sim artifact** — exact rank-1-per-point is itself low-rank/steering-friendly.
  Mitigation: architecture operates on `M_f`, not the analytic amplitude law, so it
  ports to k-Wave; a small early k-Wave OOD probe is recommended over deferring all
  realism.

---

## 9. Implementation checklist (ordered, files under `fmcfast/` & `scripts/`)

1. **`fmcfast/phantom.py` (modify)** — add `make_multi_defect`, `make_crack`
   (tilted/branched line of points with optional `|cos|` taper), `make_closely_spaced_pair`,
   and a `make_phantom(family, rng, …)` dispatcher returning `Phantom` with the **full
   scatterer list** (not just one `defect_xz`). Enforce ≥1.3·λ separations. Keep
   `make_test_phantoms` for back-compat.
2. **`fmcfast/phantom.py` (modify)** — `Phantom` gains `defects: List[Tuple[x,z,refl]]`
   (all coherent targets, for multi-target metric matching) alongside grain.
3. **`fmcfast/metrics.py` (modify/add)** — `cfar_detect(img, gx, gz, …)` (peak-pick +
   CFAR background), `multi_defect_metrics(img, gx, gz, defects, …)` (per-defect SNR
   excluding *other* defects from background, TP/FA matching), and `pod_at_fa(...)`
   aggregator. Fixes the verified `artifact_db=+4.5 dB` bug.
4. **`fmcfast/baselines.py` (modify/add)** — `reconstruct_lowrank_tuned`
   (rank+rcond sweep, returns best per scene), `reconstruct_joint_lowrank`
   (band-stack soft-impute/SVT), `reconstruct_steering_omp` (joint-bin OMP on
   analytic dictionary), `reconstruct_softimpute` (per-bin nuclear norm),
   `reconstruct_interp` (bilinear row interp). Each <30–60 lines.
5. **`fmcfast/dataset.py` (new)** — `FMCDataset` (torch): on-the-fly `simulate_fmc`
   → `to_freq` → in-band → per-bin `symmetrize` (the §1.3 up-front denoise) →
   random `S`/`mask` → robust-median norm → `(X, Y, mask, s, c)` tensors of the §2.1
   shapes. `freeze_split(seeds, path)` to cache val/test/OOD M_f stacks + cached TFM
   targets.
6. **`fmcfast/torch_tfm.py` (new)** — differentiable `tfm_torch(cube, …,
   analytic=False)` mirroring `tfm.tfm_image`'s interp gather; **unit-test to <1e-4
   vs numpy** (test in `tests/`).
7. **`fmcfast/model.py` (new)** — the §3 U-Net with FiLM(`c`) + attention bottleneck,
   and the §2.3 `symmetrize` + data-consistency output layers as fixed modules.
8. **`fmcfast/losses.py` (new)** — `L_cube`, `L_rf`, `L_tfm` (Section 4) on the
   unobserved block, with the post-epoch-5 `L_tfm` ramp.
9. **`scripts/make_eval_sets.py` (new)** — generate & freeze val/test-ID/OOD-c/
   OOD-geom/F3/F4 sets to `.npz` (M_f stacks + cached full-aperture TFM + defect lists).
10. **`scripts/train_phase1.py` (new)** — DataLoader (8 workers, randomized K & S &
    c), AdamW+cosine, FP32, DDP-ready, checkpoint best val `L_cube(U)`. Mac smoke-test
    flag (batch 2, 50 steps).
11. **`scripts/eval_phase1.py` (new)** — run net + all §6.1 baselines over the frozen
    sets at all K; compute §6.2 metrics + eff-rank ledger + bootstrap-CI / paired
    stats; emit the quality-vs-accel tables.
12. **`scripts/plot_curves.py` (extend)** — add the learned curve, joint-LR/OMP
    curves, POD@FA and resolution-rate panels, stratified by `eff_rank − K`.
13. **`configs/phase1.yaml` (new)** — families/param-ranges (Section 1), `K` set,
    `c`-train range, OOD values, model/opt/loss hyperparameters, band = 62 in-band.
14. **`tests/` (add)** — torch-TFM≈numpy, hard-layer idempotence (`symmetrize∘DC` on
    observed = identity), eff-rank==n_def sanity, multi-defect metric on a
    2-defect toy (artifact_db must be sane, not +dB).
