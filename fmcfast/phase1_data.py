"""Phase-1b dataset: k-Wave FMC cubes -> in-band M_f stacks -> masked net tensors.

Each cube is loaded once, transformed to its in-band complex M_f stack (F, N, N),
and reciprocity-symmetrized up front (the free denoising given equally to net and
baselines). Per sample we draw a transmit set S (random in training, fixed-uniform
in eval), build the observed frame mask, robust-normalize, and emit:

    x: (2F+1, N, N)  = [Re(M_obs), Im(M_obs), mask]
    y: (2F,   N, N)  = [Re(M_f),   Im(M_f)]            (full target)
    mask: (N, N),  s: scalar normalization
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def _band(fs: float, n_t: int, f0: float, frac_bw: float) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_t, d=1.0 / fs)
    lo, hi = f0 * (1 - frac_bw / 2), f0 * (1 + frac_bw / 2)
    return np.where((freqs >= lo) & (freqs <= hi))[0]


class KWaveMfDataset(Dataset):
    def __init__(self, files: Sequence[str], *, fs: float, f0: float, frac_bw: float = 0.6,
                 train: bool = True, k_choices: Sequence[int] = (16, 8, 6, 4),
                 fixed_k: Optional[int] = None):
        self.mfs: List[np.ndarray] = []
        for fpath in files:
            d = np.load(fpath)
            cube = d["cube"].astype(np.float32)
            Mf = np.fft.rfft(cube, axis=2)
            band = _band(fs, cube.shape[2], f0, frac_bw)
            mf = np.transpose(Mf[:, :, band], (2, 0, 1)).astype(np.complex64)  # (F,N,N)
            mf = 0.5 * (mf + np.transpose(mf, (0, 2, 1)))                       # reciprocity
            self.mfs.append(mf)
        self.F, self.N = self.mfs[0].shape[0], self.mfs[0].shape[1]
        self.train = train
        self.k_choices = list(k_choices)
        self.fixed_k = fixed_k

    def __len__(self):
        return len(self.mfs)

    def _mask(self, rng) -> np.ndarray:
        N = self.N
        if self.fixed_k is not None:
            K = self.fixed_k
            S = np.unique(np.round(np.linspace(0, N - 1, K)).astype(int))
        else:
            K = int(rng.choice(self.k_choices))
            S = np.sort(rng.choice(N, size=K, replace=False))
        m = np.zeros((N, N), np.float32)
        m[S, :] = 1.0
        m[:, S] = 1.0
        return m

    def __getitem__(self, idx):
        mf = self.mfs[idx]                                  # (F,N,N) complex
        rng = np.random.default_rng() if self.train else np.random.default_rng(1000 + idx)
        m = self._mask(rng)                                 # (N,N)
        norms = np.sqrt((np.abs(mf) ** 2 * m[None]).sum(axis=(1, 2)))
        s = float(np.median(norms)) + 1e-12
        mfn = mf / s
        mobs = mfn * m[None]
        x = np.concatenate([mobs.real, mobs.imag, m[None]], 0).astype(np.float32)
        y = np.concatenate([mfn.real, mfn.imag], 0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m), s


def split_files(data_dir: str, val_frac: float = 0.1, test_frac: float = 0.1):
    """Phantom-level split (each file is an independent config -> no (tx,rx) leakage)."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    n = len(files)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    return files[: n - n_val - n_test], files[n - n_val - n_test: n - n_test], files[n - n_test:]
