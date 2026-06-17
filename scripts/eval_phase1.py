#!/usr/bin/env python
"""Phase-1b decisive eval: trained net vs the classical frontier on held-out
k-Wave TEST cubes, same metric (unobserved-block band_nrmse), same uniform masks.

Reports TWO scorings per method:
  * full   -- on the whole cube (energy-dominated by the direct surface wave)
  * gated  -- after time-gating out the direct arrival, so it scores reconstruction
              of the weak finite-size DEFECT echo, which is what matters for NDE.

If the net's win survives gating, it is learning real defect physics; if it
collapses, the headline win was mostly the deterministic direct wave.

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
from fmcfast.phase1_data import KWaveMfDataset, split_files, _band
from fmcfast.phase1_model import UNet, apply_constraints
from fmcfast.sampling import observed_mask, uniform_tx_set

FS, F0, BW = 50e6, 5e6, 0.6
NYS = [{"rank": r, "rcond": rc} for r in (2, 4, 8, 12, 16) for rc in (1e-8, 1e-4, 1e-2)]


def gate_direct(cube, x, c, fs, guard_mm=4.0):
    """Zero samples before |x_i - x_j|/c + guard -> removes the direct surface wave."""
    n_t = cube.shape[2]
    cut = np.abs(x[:, None] - x[None, :]) / c + guard_mm * 1e-3 / c   # (N,N)
    t = np.arange(n_t) / fs
    keep = (t[None, None, :] >= cut[:, :, None])                       # (N,N,n_t)
    return cube * keep


def net_cube(pred, s, F, band, n_t):
    """Net's M_f output -> band-limited time cube (mirrors baselines)."""
    p = pred[0].detach().cpu().numpy()
    mf = (p[:F] + 1j * p[F:]) * s                                     # (F,N,N)
    full = np.zeros((mf.shape[1], mf.shape[2], n_t // 2 + 1), complex)
    full[:, :, band] = np.transpose(mf, (1, 2, 0))
    return np.fft.irfft(full, n=n_t, axis=2)


def unobs(rec, true, obs):
    return metrics.band_nrmse_blocks(rec, true, obs, fs=FS, f0=F0, frac_bw=BW)["nrmse_unobs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="kwave_train")
    ap.add_argument("--ckpt", default="results/phase1_ckpt.pt")
    ap.add_argument("--ks", default="16,8,4")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    test_files = split_files(args.data_dir)[2]
    ks = [int(k) for k in args.ks.split(",")]
    print(f"test cubes: {len(test_files)}  device={device}", flush=True)

    ck = torch.load(args.ckpt, map_location=device)
    F = ck["F"]
    model = UNet(2 * F + 1, 2 * F, base=ck["base"]).to(device).eval()
    model.load_state_dict(ck["model"])

    ds = KWaveMfDataset(test_files, fs=FS, f0=F0, frac_bw=BW, train=False, fixed_k=ks[0])
    cubes = [np.load(f) for f in test_files]
    n = int(cubes[0]["n_elements"])
    array = LinearArray(n, float(cubes[0]["pitch"]))
    n_t = int(cubes[0]["n_t"])
    band = _band(FS, n_t, F0, BW)

    hdr = f"{'method':>9} | " + "  ".join(f"K={k}(full/gated)" for k in ks)
    print("\n" + hdr, flush=True)
    agg = {m: {k: ([], []) for k in ks} for m in ("net", "naive", "nystrom", "steer")}
    for K in ks:
        ds.fixed_k = K
        for idx, d in enumerate(cubes):
            cube = d["cube"].astype(float)
            c = float(d["c0"])
            defects = [(float(px), float(pz), 1.0) for px, pz, r in d["defects"]]
            S = uniform_tx_set(n, K); obs = observed_mask(n, S)
            cube_g = gate_direct(cube, array.x, c, FS)
            x, _, m, s = ds[idx]
            with torch.no_grad():
                pred = apply_constraints(model(x[None].to(device)), m[None].to(device),
                                         x[None, : 2 * F].to(device), F)
            recs = {
                "net": net_cube(pred, s, F, band, n_t),
                "naive": reconstruct_naive(cube, S),
                "nystrom": min((reconstruct_nystrom(cube, S, fs=FS, f0=F0, frac_bw=BW, **hp) for hp in NYS),
                               key=lambda r: unobs(r, cube, obs)),
                "steer": reconstruct_steering_oracle(cube, S, defects, array, c=c, fs=FS, f0=F0, frac_bw=BW),
            }
            for mth, rec in recs.items():
                agg[mth][K][0].append(unobs(rec, cube, obs))
                agg[mth][K][1].append(unobs(gate_direct(rec, array.x, c, FS), cube_g, obs))
    for mth in ("net", "naive", "nystrom", "steer"):
        cells = []
        for K in ks:
            f = np.median(agg[mth][K][0]); g = np.median(agg[mth][K][1])
            cells.append(f"{f:.3f}/{g:.3f}")
        print(f"{mth:>9} | " + "      ".join(cells), flush=True)
    print("\n(unobserved-block band_nrmse, median; full = whole cube, gated = direct wave removed)")
    print("EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
