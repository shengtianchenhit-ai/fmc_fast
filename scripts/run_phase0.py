#!/usr/bin/env python
"""Phase-0 driver: produce the quality-vs-acceleration baseline map.

For every test phantom and every transmit budget K, score two cheap baselines
against the full-aperture reference:

    * naive   -- TFM over the K transmitted elements only (no reconstruction)
    * lowrank -- per-frequency low-rank completion of the full cube, then full TFM

Outputs a tidy CSV (one row per phantom x method x K) plus a few saved images for
a qualitative figure. These two curves are the "research map": they define what a
learned reconstructor must beat.

Usage:
    python scripts/run_phase0.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmcfast import metrics
from fmcfast.baselines import reconstruct_lowrank
from fmcfast.config import build_array, build_grid, load_config
from fmcfast.forward import simulate_fmc
from fmcfast.phantom import make_test_phantoms
from fmcfast.sampling import uniform_tx_set
from fmcfast.tfm import tfm_image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/phase0.csv")
    ap.add_argument("--images-out", default="results/phase0_images.npz")
    args = ap.parse_args()

    cfg = load_config(args.config)
    array = build_array(cfg)
    gx, gz = build_grid(cfg)
    n = array.n_elements

    med, pul, acq = cfg["medium"], cfg["pulse"], cfg["acq"]
    c, fs = float(med["c"]), float(acq["fs"])
    f0, n_cycles, frac_bw = float(pul["f0"]), float(pul["n_cycles"]), float(pul["frac_bw"])
    n_t = int(acq["n_t"])

    ds = cfg["dataset"]
    noise_db = ds.get("noise_db")
    phantoms = make_test_phantoms(
        int(ds["n_test"]),
        x_range=tuple(map(float, ds["defect"]["x_range"])),
        z_range=tuple(map(float, ds["defect"]["z_range"])),
        reflectivity=float(ds["defect"]["reflectivity"]),
        n_grain=int(ds.get("n_grain", 0)),
        seed=int(ds.get("seed", 0)),
    )

    tx_counts = [int(k) for k in cfg["sampling"]["tx_counts"]]
    rank = int(cfg["lowrank"]["rank"])

    def img_metrics(cube, tx_set, defect):
        img = tfm_image(cube, array, gx, gz, c=c, fs=fs, tx_set=tx_set)
        return img, metrics.defect_metrics(img, gx, gz, defect, c=c, f0=f0)

    rows = []
    saved_images = {"gx": gx, "gz": gz}
    rng = np.random.default_rng(int(ds.get("seed", 0)))

    for ph in tqdm(phantoms, desc="phantoms"):
        cube = simulate_fmc(array, ph.scatterers, c=c, fs=fs, n_t=n_t,
                            f0=f0, n_cycles=n_cycles, noise_db=noise_db, rng=rng)

        # Full-aperture reference.
        img_full, m_full = img_metrics(cube, None, ph.defect_xz)
        rows.append(dict(phantom=ph.id, method="full", n_tx=n, accel=1.0, nrmse=0.0, **m_full))
        if ph.id == 0:
            saved_images["full"] = img_full
            saved_images["defect"] = np.array(ph.defect_xz)

        for k in tx_counts:
            S = uniform_tx_set(n, k)
            n_tx = int(S.size)
            accel = n / n_tx

            # Naive: image over transmitted aperture only.
            img_nv, m_nv = img_metrics(cube, S, ph.defect_xz)
            rows.append(dict(phantom=ph.id, method="naive", n_tx=n_tx, accel=accel,
                             nrmse=np.nan, **m_nv))

            # Low-rank: complete the cube, then full TFM.
            cube_rec, _ = reconstruct_lowrank(cube, S, fs=fs, f0=f0, frac_bw=frac_bw,
                                              rank=rank)
            img_lr, m_lr = img_metrics(cube_rec, None, ph.defect_xz)
            nrmse = metrics.band_nrmse(cube_rec, cube, fs=fs, f0=f0, frac_bw=frac_bw)
            rows.append(dict(phantom=ph.id, method="lowrank", n_tx=n_tx, accel=accel,
                             nrmse=nrmse, **m_lr))

            if ph.id == 0 and n_tx in (8, 16):
                saved_images[f"naive_k{n_tx}"] = img_nv
                saved_images[f"lowrank_k{n_tx}"] = img_lr

    # Write CSV.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["phantom", "method", "n_tx", "accel", "snr_db", "contrast_db",
            "artifact_db", "api", "lat6_mm", "ax6_mm", "nrmse", "peak"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    np.savez(args.images_out, **saved_images)

    # Console summary: median SNR by method x acceleration.
    print(f"\nWrote {len(rows)} rows -> {args.out}")
    print(f"Saved sample images -> {args.images_out}\n")
    print(f"{'method':>8} {'n_tx':>5} {'accel':>6} {'SNR dB':>8} "
          f"{'artifact dB':>12} {'NRMSE':>8}")
    by = {}
    for r in rows:
        by.setdefault((r["method"], r["n_tx"]), []).append(r)
    for (method, n_tx), rs in sorted(by.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
        snr = np.median([x["snr_db"] for x in rs])
        art = np.median([x["artifact_db"] for x in rs])
        nrs = [x["nrmse"] for x in rs if not np.isnan(x["nrmse"])]
        nr = np.median(nrs) if nrs else np.nan
        accel = rs[0]["accel"]
        nr_s = "   -   " if np.isnan(nr) else f"{nr:8.4f}"
        print(f"{method:>8} {n_tx:>5} {accel:>6.2f} {snr:>8.2f} {art:>12.2f} {nr_s:>8}")


if __name__ == "__main__":
    main()
