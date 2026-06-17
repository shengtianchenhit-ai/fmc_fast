#!/usr/bin/env python
"""Phase-1b decisive eval: trained net vs the classical frontier on held-out
k-Wave TEST cubes, same metric (unobserved-block band_nrmse), same uniform masks.

Also reports the direct-wave ablation: re-score with the early direct-arrival
samples time-gated out, so we see how much of the win is the deterministic surface
wave vs the harder finite-size defect scattering.

    python scripts/eval_phase1.py --data-dir kwave_train --ckpt results/phase1_ckpt.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmcfast import LinearArray, metrics
from fmcfast.baselines import (reconstruct_naive, reconstruct_nystrom,
                               reconstruct_steering_oracle)
from fmcfast.phase1_data import KWaveMfDataset, split_files
from fmcfast.phase1_model import UNet, apply_constraints, cube_loss
from fmcfast.sampling import observed_mask, uniform_tx_set

FS, F0, BW = 50e6, 5e6, 0.6
NYS = [{"rank": r, "rcond": rc} for r in (2, 4, 8, 12, 16) for rc in (1e-8, 1e-4, 1e-2)]


def gate_direct(cube, array, c, fs, guard_mm=3.0):
    """Zero samples before the shortest defect-free two-way time + guard, i.e. remove
    the direct surface wave (which travels along the surface, arriving earliest)."""
    n, _, n_t = cube.shape
    x = array.x
    # direct path tx->rx along surface = |x_i - x_j|; gate a bit beyond the max direct.
    tt = (np.abs(x[:, None] - x[None, :]) / c)  # (N,N) one-way surface time
    gated = cube.copy()
    t = np.arange(n_t) / fs
    for i in range(n):
        for j in range(n):
            cut = tt[i, j] + guard_mm * 1e-3 / c
            gated[i, j, t < cut] = 0.0
    return gated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="kwave_train")
    ap.add_argument("--ckpt", default="results/phase1_ckpt.pt")
    ap.add_argument("--ks", default="16,8,6,4")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")

    test_files = split_files(args.data_dir)[2]
    print(f"test cubes: {len(test_files)}", flush=True)
    ks = [int(k) for k in args.ks.split(",")]

    ck = torch.load(args.ckpt, map_location=device)
    F = ck["F"]
    model = UNet(2 * F + 1, 2 * F, base=ck["base"]).to(device).eval()
    model.load_state_dict(ck["model"])

    ds = KWaveMfDataset(test_files, fs=FS, f0=F0, frac_bw=BW, train=False, fixed_k=ks[0])
    cubes = [np.load(f) for f in test_files]
    n = int(cubes[0]["n_elements"])
    array = LinearArray(n, float(cubes[0]["pitch"]))
    fkw = dict(fs=FS, f0=F0, frac_bw=BW)

    print(f"\n{'K (accel)':>12} | {'net':>7} {'naive':>7} {'nystrom':>8} {'steer_orc':>9}")
    rows = {}
    for K in ks:
        ds.fixed_k = K
        net_e, nai, nys, ste = [], [], [], []
        for idx, d in enumerate(cubes):
            cube = d["cube"].astype(float)
            defects = [(float(x), float(z), 1.0) for x, z, r in d["defects"]]
            S = uniform_tx_set(n, K); obs = observed_mask(n, S)
            # net
            x, y, m, _ = ds[idx]
            with torch.no_grad():
                pred = apply_constraints(model(x[None].to(device)), m[None].to(device),
                                         x[None, : 2 * F].to(device), F)
                net_e.append(cube_loss(pred, y[None].to(device), m[None].to(device)).item())
            # classical
            nai.append(metrics.band_nrmse_blocks(reconstruct_naive(cube, S), cube, obs, **fkw)["nrmse_unobs"])
            nys.append(min(metrics.band_nrmse_blocks(reconstruct_nystrom(cube, S, **fkw, **hp), cube, obs, **fkw)["nrmse_unobs"] for hp in NYS))
            ste.append(metrics.band_nrmse_blocks(reconstruct_steering_oracle(cube, S, defects, array, c=float(d["c0"]), **fkw), cube, obs, **fkw)["nrmse_unobs"])
        accel = (n / int(uniform_tx_set(n, K).size)) / 2.0
        rows[K] = (np.median(net_e), np.median(nai), np.median(nys), np.median(ste))
        print(f"K={K:>2} ({accel:>3.1f}x) | {rows[K][0]:>7.3f} {rows[K][1]:>7.3f} {rows[K][2]:>8.3f} {rows[K][3]:>9.3f}", flush=True)

    print("\n(unobserved-block band_nrmse, median over test cubes; lower=better)")
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
