#!/usr/bin/env python
"""Phase-1b: train the sparse-Tx reconstruction U-Net on k-Wave M_f stacks.

Headline metric = band_nrmse on the UNOBSERVED block (== val cube_loss). The bar
to beat is the strong classical frontier (~0.86-1.0 on k-Wave); the question is
whether the net drives this materially lower (learnable) or not (info-starved).

    # Mac smoke test (MPS), overfit the 6 probe cubes to prove the loop learns:
    python scripts/train_phase1.py --data-dir /tmp/kwave_data --smoke
    # Server training run:
    python scripts/train_phase1.py --data-dir kwave_train --epochs 200 --batch 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fmcfast.phase1_data import KWaveMfDataset, split_files
from fmcfast.phase1_model import UNet, apply_constraints, cube_loss

FS, F0 = 50e6, 5e6


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, val_ds, device, ks=(16, 8, 4)):
    model.eval()
    F = val_ds.F
    out = {}
    for K in ks:
        val_ds.fixed_k = K
        losses = []
        for i in range(len(val_ds)):
            x, y, m, _ = val_ds[i]
            x, y, m = x[None].to(device), y[None].to(device), m[None].to(device)
            pred = apply_constraints(model(x), m, x[:, : 2 * F], F)
            losses.append(cube_loss(pred, y, m).item())
        out[K] = float(np.mean(losses))
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/phase1_ckpt.pt")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true", help="resume from <out>_last.pt if present")
    ap.add_argument("--gate", action="store_true", help="gate out the direct wave (train on defect echo)")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.batch, args.base = 40, 2, 32

    device = pick_device(args.device)
    tr_files, va_files, te_files = split_files(args.data_dir)
    if args.smoke:                       # tiny set: overfit everything, eval on same
        import glob
        tr_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
        va_files = tr_files
    print(f"device={device}  train={len(tr_files)} val={len(va_files)} test={len(te_files)}", flush=True)

    train_ds = KWaveMfDataset(tr_files, fs=FS, f0=F0, train=True, gate=args.gate)
    val_ds = KWaveMfDataset(va_files, fs=FS, f0=F0, train=False, fixed_k=8, gate=args.gate)
    F = train_ds.F
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=False)

    model = UNet(2 * F + 1, 2 * F, base=args.base).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"F={F}  model params={n_par/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(loader)))

    best = 1e9
    start_ep = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    last_path = args.out.replace(".pt", "_last.pt")
    done_path = args.out.replace(".pt", "_done.txt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {start_ep} (best {best:.4f})", flush=True)
    for ep in range(start_ep, args.epochs):
        t0 = time.time()
        tot, nb = 0.0, 0
        for x, y, m, _ in loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = apply_constraints(model(x), m, x[:, : 2 * F], F)
            loss = cube_loss(pred, y, m)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
        if ep % max(1, args.epochs // 20) == 0 or ep == args.epochs - 1:
            ev = evaluate(model, val_ds, device)
            vmean = float(np.mean(list(ev.values())))
            if vmean < best:
                best = vmean
                torch.save({"model": model.state_dict(), "F": F, "base": args.base}, args.out)
            # checkpoint for resume (model + optimizer + schedule + epoch) every eval
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": ep, "best": best,
                        "F": F, "base": args.base}, last_path)
            print(f"ep {ep:3d}  train {tot/nb:.4f}  val nrmse_unobs "
                  f"K16={ev[16]:.3f} K8={ev[8]:.3f} K4={ev[4]:.3f}  ({time.time()-t0:.1f}s)", flush=True)
    with open(done_path, "w") as fh:
        fh.write(f"best_val={best:.4f}")
    print(f"DONE best_val={best:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
