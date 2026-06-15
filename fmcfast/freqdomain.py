"""Frequency-domain helpers and reciprocity symmetrisation.

Working per frequency bin turns the cube into a stack of N x N complex matrices
M_f. Each M_f is near low-rank (limited physical degrees of freedom), which is
exactly what the low-rank baseline exploits.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def to_freq(cube: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Real FFT of an FMC cube along time.

    Returns ``(Mf, freqs)`` where ``Mf`` has shape (N, N, F) complex and ``freqs``
    are the F rfft bin frequencies [Hz].
    """
    Mf = np.fft.rfft(cube, axis=2)
    freqs = np.fft.rfftfreq(cube.shape[2], d=1.0 / fs)
    return Mf, freqs


def from_freq(Mf: np.ndarray, n_t: int) -> np.ndarray:
    """Inverse real FFT back to a time cube of length ``n_t``."""
    return np.fft.irfft(Mf, n=n_t, axis=2)


def band_bins(freqs: np.ndarray, f0: float, frac_bw: float) -> np.ndarray:
    """Indices of frequency bins inside the pulse passband [f0(1±frac_bw/2)]."""
    lo = f0 * (1.0 - frac_bw / 2.0)
    hi = f0 * (1.0 + frac_bw / 2.0)
    return np.where((freqs >= lo) & (freqs <= hi))[0]


def symmetrize(M: np.ndarray) -> np.ndarray:
    """Enforce reciprocity M = M^T (complex-symmetric, not Hermitian)."""
    return 0.5 * (M + M.T)
