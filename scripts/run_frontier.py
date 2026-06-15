#!/usr/bin/env python
"""Phase-1a: the strong-classical-frontier map (docs/phase1_spec.md s0, s6).

Before training any network, establish the bar. On harder, *coherent* rank>K
scenes (multi-SDH, cracks), measure the Phase-1 headline axis -- band_nrmse on the
UNOBSERVED block -- for the strongest cheap reconstructors, oracle-tuned per scene:

    * naive            -- predicts nothing on the unobserved block (=> ~1.0)
    * nystrom (tuned)  -- per-bin Nystrom, oracle rank+rcond per scene
    * joint-LR         -- joint cross-frequency low-rank (SVT on the band stack),
                          oracle lambda; the ceiling for the coupling prior

Stratify by ``eff_rank - K``. If the best classical method's unobserved-block error
RISES with eff_rank-K, that gap is the room a learned/parametric prior could fill
(=> build the net). If joint-LR already drives it to ~0, that's the negative result.

Usage:
    python scripts/run_frontier.py --config configs/default.yaml --n 36
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmcfast import metrics
from fmcfast.baselines import (reconstruct_joint_lowrank, reconstruct_naive,
                               reconstruct_nystrom, reconstruct_steering_oracle)
from fmcfast.config import build_array, build_grid, load_config
from fmcfast.forward import simulate_fmc
from fmcfast.freqdomain import scene_eff_rank
from fmcfast.phantom import make_phantom
from fmcfast.sampling import observed_mask, uniform_tx_set
from fmcfast.tfm import tfm_image  # noqa: F401  (kept for downstream extension)

NYSTROM_GRID = [{"rank": r, "rcond": rc} for r in (2, 4, 8, 12, 16)
                for rc in (1e-8, 1e-4, 1e-2)]
JOINT_GRID = [{"lam_rel": l} for l in (0.01, 0.05, 0.15)]


def oracle(method_fn, grid, cube, S, obs, fkw):
    """Best unobserved-block band_nrmse over a hyper-parameter grid (oracle-tuned)."""
    best = None
    for hp in grid:
        rec = method_fn(cube, S, **fkw, **hp)
        m = metrics.band_nrmse_blocks(rec, cube, obs, **fkw)
        if best is None or m["nrmse_unobs"] < best["nrmse_unobs"]:
            best = m
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=36, help="phantoms per family")
    ap.add_argument("--out", default="results/frontier.csv")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    array = build_array(cfg)
    n = array.n_elements
    med, pul, acq = cfg["medium"], cfg["pulse"], cfg["acq"]
    c, fs = float(med["c"]), float(acq["fs"])
    f0, n_cycles, frac_bw = float(pul["f0"]), float(pul["n_cycles"]), float(pul["frac_bw"])
    n_t = int(acq["n_t"])
    fkw = dict(fs=fs, f0=f0, frac_bw=frac_bw)
    k_set = [16, 8, 6, 4]

    rng = np.random.default_rng(args.seed)
    # Build a spread of scenes: multi-SDH with n_def spanning 2..16, plus cracks.
    specs = []
    for i in range(args.n):
        nd = int(2 + (i % 15))          # n_def 2..16 to sweep eff_rank
        specs.append(("multi", {"n_range": (nd, nd)}))
    for i in range(args.n // 2):
        specs.append(("crack", {}))

    rows = []
    for pid, (family, fam_kwargs) in enumerate(tqdm(specs, desc="scenes")):
        ph = make_phantom(family, rng, c=c, f0=f0, pid=pid, **fam_kwargs)
        cube = simulate_fmc(array, ph.scatterers, c=c, fs=fs, n_t=n_t,
                            f0=f0, n_cycles=n_cycles)  # noiseless: rank is coherent
        eff = scene_eff_rank(cube, fs, f0, frac_bw)
        for K in k_set:
            S = uniform_tx_set(n, K)
            n_tx = int(S.size)
            obs = observed_mask(n, S)

            naive = metrics.band_nrmse_blocks(reconstruct_naive(cube, S), cube, obs, **fkw)
            nys = oracle(reconstruct_nystrom, NYSTROM_GRID, cube, S, obs, fkw)
            joint = oracle(reconstruct_joint_lowrank, JOINT_GRID, cube, S, obs, fkw)
            steer = metrics.band_nrmse_blocks(
                reconstruct_steering_oracle(cube, S, ph.defects, array, c=c, **fkw),
                cube, obs, **fkw)

            base = dict(phantom=pid, family=family, eff_rank=eff, n_tx=n_tx,
                        accel_raw=n / n_tx, accel_hmc=(n / n_tx) / 2.0,
                        eff_minus_k=eff - n_tx)
            for name, m in (("naive", naive), ("nystrom", nys), ("joint_lr", joint),
                            ("steer_oracle", steer)):
                rows.append(dict(base, method=name,
                                 nrmse_unobs=m["nrmse_unobs"],
                                 nrmse_overall=m["nrmse_overall"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["phantom", "family", "eff_rank", "n_tx", "accel_raw", "accel_hmc",
            "eff_minus_k", "method", "nrmse_unobs", "nrmse_overall"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"\nWrote {len(rows)} rows -> {args.out}\n")

    # Summary: median unobserved-block NRMSE, stratified by (method, eff_rank - K bin).
    def bucket(d):
        if d <= 0:
            return "<=0 (rank<=K)"
        if d <= 4:
            return "1..4"
        if d <= 8:
            return "5..8"
        return ">8"
    agg = defaultdict(list)
    for r in rows:
        agg[(r["method"], bucket(r["eff_minus_k"]))].append(r["nrmse_unobs"])
    order = ["<=0 (rank<=K)", "1..4", "5..8", ">8"]
    print("median band_nrmse on UNOBSERVED block, by eff_rank - K:")
    print(f"{'method':>10} | " + " | ".join(f"{b:>14}" for b in order))
    for method in ("naive", "nystrom", "joint_lr", "steer_oracle"):
        cells = []
        for b in order:
            v = agg.get((method, b))
            cells.append(f"{np.median(v):>14.3f}" if v else f"{'-':>14}")
        print(f"{method:>10} | " + " | ".join(cells))
    print("\n(naive ~1.0 = predicts nothing; the gap between joint_lr and 0 in the")
    print(" rank>K columns is the room a learned/parametric prior could fill.)")


if __name__ == "__main__":
    main()
