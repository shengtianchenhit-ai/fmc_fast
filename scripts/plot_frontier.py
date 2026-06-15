#!/usr/bin/env python
"""Plot the Phase-1a classical-frontier map from results/frontier.csv.

Left:  unobserved-block band_nrmse vs (eff_rank - K) -- shows where generic
       low-rank breaks and whether the parametric ceiling still recovers it.
Right: unobserved-block band_nrmse vs acceleration (rel HMC), per method.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {"naive": ("o:", "0.5"), "nystrom": ("s-", "C0"),
         "joint_lr": ("^-", "C2"), "steer_oracle": ("D-", "C3")}


def load(path):
    with open(path) as fh:
        return [{k: (v) for k, v in r.items()} for r in csv.DictReader(fh)]


def med_by(rows, method, xkey, clip=2.0):
    by = defaultdict(list)
    for r in rows:
        if r["method"] != method:
            continue
        by[float(r[xkey])].append(min(float(r["nrmse_unobs"]), clip))
    xs = sorted(by)
    return np.array(xs), np.array([np.median(by[x]) for x in xs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/frontier.csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    rows = load(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for method, (sty, col) in STYLE.items():
        x, y = med_by(rows, method, "eff_minus_k")
        if x.size:
            axes[0].plot(x, y, sty, color=col, label=method, lw=2, ms=6)
        xa, ya = med_by(rows, method, "accel_hmc")
        if xa.size:
            axes[1].plot(xa, ya, sty, color=col, label=method, lw=2, ms=6)
    axes[0].axvline(0, color="k", ls=":", lw=1)
    axes[0].set_xlabel("eff_rank − K   (>0 ⇒ low-rank under-determined)")
    axes[0].set_ylabel("unobserved-block band_nrmse (median)")
    axes[0].set_title("Where does low-rank break, and is it recoverable?")
    axes[1].set_xlabel("acceleration  (×, rel HMC)")
    axes[1].set_ylabel("unobserved-block band_nrmse (median)")
    axes[1].set_title("Frontier vs acceleration")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(-0.05, 1.6)
    fig.suptitle("Phase-1a: strong classical frontier (the bar a learned method must beat)")
    fig.tight_layout()
    p = os.path.join(args.outdir, "frontier_map.png")
    fig.savefig(p, dpi=140)
    print("wrote", p)


if __name__ == "__main__":
    main()
