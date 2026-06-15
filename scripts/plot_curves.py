#!/usr/bin/env python
"""Plot the Phase-0 baseline map from the CSV produced by run_phase0.py.

Figure 1: detection SNR vs acceleration, one curve per baseline (+full reference).
Figure 2: qualitative TFM montage (full vs naive vs low-rank) for one phantom.

Usage:
    python scripts/plot_curves.py --csv results/phase0.csv
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


def load_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def agg(rows, method, field):
    """Median ``field`` vs acceleration for a given method."""
    by = defaultdict(list)
    for r in rows:
        if r["method"] != method:
            continue
        v = r[field]
        if v == "" or v != v:  # empty or NaN string
            continue
        try:
            by[float(r["accel"])].append(float(v))
        except ValueError:
            continue
    xs = sorted(by)
    return np.array(xs), np.array([np.median(by[x]) for x in xs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/phase0.csv")
    ap.add_argument("--images", default="results/phase0_images.npz")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    os.makedirs(args.outdir, exist_ok=True)

    # ---- Figure 1: detection vs acceleration (SNR + artifact level) --------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [("snr_db", "defect SNR [dB]  (higher better)", "max"),
              ("artifact_db", "peak artifact [dB rel. defect]  (lower better)", "min")]
    for ax, (field, ylabel, refkind) in zip(axes, panels):
        for method, style in [("naive", "o--"), ("lowrank", "s-")]:
            x, y = agg(rows, method, field)
            if x.size:
                ax.plot(x, y, style, label=method, linewidth=2, markersize=6)
        xf, yf = agg(rows, "full", field)
        if yf.size:
            ax.axhline(yf[0], color="k", ls=":", label="full (ref)")
        ax.set_xlabel("acceleration  N / K  (transmit reduction)")
        ax.set_ylabel(ylabel + "  (median)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Phase-0 baseline map: what a learned method must beat")
    fig.tight_layout()
    p1 = os.path.join(args.outdir, "curve_snr_vs_accel.png")
    fig.savefig(p1, dpi=140)
    print("wrote", p1)

    # ---- Figure 1b: NRMSE vs acceleration (low-rank fidelity) --------------
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x, y = agg(rows, "lowrank", "nrmse")
    if x.size:
        ax.plot(x, y, "s-", color="C1", linewidth=2, markersize=6, label="lowrank")
    ax.set_xlabel("acceleration  N / K")
    ax.set_ylabel("in-band cube NRMSE (phase-sensitive)")
    ax.set_title("Low-rank reconstruction fidelity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p1b = os.path.join(args.outdir, "curve_nrmse_vs_accel.png")
    fig.savefig(p1b, dpi=140)
    print("wrote", p1b)

    # ---- Figure 2: qualitative montage -------------------------------------
    if os.path.exists(args.images):
        d = np.load(args.images)
        gx, gz = d["gx"], d["gz"]
        ext = [gx[0] * 1e3, gx[-1] * 1e3, gz[-1] * 1e3, gz[0] * 1e3]
        panels = [k for k in ("full", "naive_k16", "lowrank_k16",
                              "naive_k8", "lowrank_k8") if k in d.files]
        if panels:
            fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2))
            axes = np.atleast_1d(axes)
            defect = d["defect"] * 1e3 if "defect" in d.files else None
            vmin = -40.0  # dB display floor, standard for TFM
            for ax, key in zip(axes, panels):
                img = d[key]
                img_db = 20.0 * np.log10(img / (img.max() + 1e-12) + 1e-12)
                im = ax.imshow(img_db, extent=ext, aspect="auto", cmap="inferno",
                               vmin=vmin, vmax=0)
                if defect is not None:
                    ax.plot(defect[0], defect[1], "c+", ms=10, mew=1.5)
                ax.set_title(key)
                ax.set_xlabel("x [mm]")
            fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label="dB")
            axes[0].set_ylabel("z [mm]")
            fig.tight_layout()
            p2 = os.path.join(args.outdir, "tfm_montage.png")
            fig.savefig(p2, dpi=140)
            print("wrote", p2)


if __name__ == "__main__":
    main()
