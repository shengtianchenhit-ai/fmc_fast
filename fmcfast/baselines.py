"""Cheap sparse-Tx baselines that a learned method must beat.

* ``reconstruct_naive``   -- no reconstruction; just expose the K transmitted rows
                             (missing rows stay zero). TFM is then done over the
                             transmitted aperture only.
* ``reconstruct_lowrank`` -- per-frequency low-rank completion of the full cube,
                             using reciprocity to seed the observed frame.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .completion import lowrank_complete
from .freqdomain import band_bins, from_freq, to_freq


def reconstruct_naive(cube: np.ndarray, tx_set: np.ndarray) -> np.ndarray:
    """Return a cube with only the transmitted rows kept (others zeroed).

    The naive baseline does not reconstruct; this is its honest "knowledge" of the
    full cube. For imaging, prefer passing ``tx_set`` straight to ``tfm_image``.
    """
    n = cube.shape[0]
    keep = np.zeros(n, dtype=bool)
    keep[tx_set] = True
    out = np.zeros_like(cube)
    out[keep] = cube[keep]
    return out


def reconstruct_lowrank(
    cube: np.ndarray,
    tx_set: np.ndarray,
    *,
    fs: float,
    f0: float,
    frac_bw: float,
    rank: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Low-rank-complete the full cube from a sparse-Tx acquisition.

    Completion (Nyström) runs per in-band frequency bin; out-of-band bins
    (negligible pulse energy) are left zero, giving a band-limited reconstruction.
    Returns ``(cube_rec, band_idx)``.
    """
    _, _, n_t = cube.shape
    Mf, freqs = to_freq(cube, fs)
    band = band_bins(freqs, f0, frac_bw)

    Mf_rec = np.zeros_like(Mf)
    for f in band:
        Mf_rec[:, :, f] = lowrank_complete(Mf[:, :, f], tx_set, rank=rank)

    cube_rec = from_freq(Mf_rec, n_t)
    return cube_rec, band
