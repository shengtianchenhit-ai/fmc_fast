# fmc_fast — learning-accelerated FMC acquisition

Can we transmit on only **K of N** array elements (each shot still received on the
full array), reconstruct the full FMC cube, and get a TFM image that is *as usable*
as the fully-sampled one? How small can K go before detection breaks — and does a
learned reconstructor actually beat cheap baselines?

This repo follows the phased plan in [`docs/plan.md`](docs/plan.md). **Phase 0** (here)
builds the end-to-end pipeline and the two baseline curves that define the bar a
learned method must clear.

## Why transmit count is the right knob

FMC acquisition time is ∝ **number of transmit events**, not the N² A-scans (every
shot is received on the whole array in parallel; the cost is waiting for the medium
to quiet between shots). So we sub-sample **transmits** (sparse-Tx): fire K elements,
receive on all N, then complete the missing rows of the N×N×T cube and run TFM as
usual. Acceleration ≈ N/K, and it maps directly to wall-clock time.

**Honesty convention — report acceleration relative to HMC.** A linear array is
reciprocal (M_ij = M_ji), and half-matrix capture (HMC) already banks ~2× for free.
We bake reciprocity into every method (the observed set of a sparse-Tx shot is the
union of rows *and* columns in S) so the 2× is not double-counted as our contribution.

## Layout

```
fmcfast/            core library (solver-agnostic)
  geometry.py       linear array + Gabor pulse
  forward.py        fast point-scatterer FMC forward model (Phase-0 solver)
  tfm.py            Total Focusing Method beamforming (full or sparse-Tx)
  sampling.py       uniform transmit selection + observed-entry mask
  freqdomain.py     time<->freq, band selection, reciprocity symmetrisation
  completion.py     Nyström low-rank completion (the baseline to beat)
  baselines.py      naive sub-sample + low-rank reconstruction
  metrics.py        phase-sensitive cube NRMSE + TFM detection metrics
  phantom.py        random side-drilled-hole (SDH) test specimens
  config.py         YAML config loader
configs/default.yaml
scripts/
  run_phase0.py     sweep transmit budgets -> results/phase0.csv (+ sample images)
  plot_curves.py    quality-vs-acceleration curves + dB TFM montage
tests/test_pipeline.py
```

## Run it (Phase 0, CPU, ~1 min)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_phase0.py --config configs/default.yaml
python scripts/plot_curves.py --csv results/phase0.csv
pytest -q
```

Outputs land in `results/`: `phase0.csv`, `curve_snr_vs_accel.png`,
`curve_nrmse_vs_accel.png`, `tfm_montage.png` (all git-ignored — regenerable).

## The two baselines (what learning must beat)

* **naive** — TFM over the K transmitted elements only, no reconstruction. Quantifies
  how fast TFM degrades (grating lobes, SNR) as transmits drop.
* **lowrank** — per-frequency completion of the cube. Each `M_f` is near low-rank
  (limited physical DOF), and sparse-Tx + reciprocity hands us whole rows/columns, so
  the **Nyström extension** `M̂ = M_{:,S} (M_{S,S})⁺ M_{:,S}ᵀ` completes it directly —
  exact when rank(M_f) ≤ K. Cheap, parameter-light, and *strong*.

## Phase-0 finding (default single-SDH set)

Low-rank completion holds full-aperture quality (SNR ≈ 60 dB, peak artifact ≈ −40 dB,
cube NRMSE < 0.02) out to **16×**, while naive collapses past ~4× (artifacts climb to
−29 dB). For a single point-like reflector this is expected — `M_f` is rank-1, which
Nyström nails — and it is the plan's honest checkpoint: **at this complexity learning
has almost no room.** The interesting regime is harder data where the true rank rises:
multiple/extended scatterers, cracks with orientation, grain noise, OOD speed of sound.
That is where Phase 1+ must show a learned method beats low-rank.

## Roadmap

| Phase | Goal |
|---|---|
| **0** (done) | Pipeline + baseline map (naive vs low-rank) |
| 1 | Supervised sparse-Tx reconstruction, fixed uniform sampling, frequency domain, reciprocity symmetrisation |
| 2 | OOD stress (defects / speed of sound) + physics constraints + self-supervised variant |
| 3 | Learn *which* K transmits to fire (active selection) |

Phase 1+ runs on the GPU server (`pytorch` env for the network, `kwave310` env for
k-Wave realism); Phase 0 is pure-CPU numpy and runs anywhere.
