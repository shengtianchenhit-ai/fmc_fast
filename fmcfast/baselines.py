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
from .freqdomain import band_bins, from_freq, symmetrize, to_freq
from .sampling import observed_mask


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


# ============================================================================
# Phase-1 strong classical frontier (docs/phase1_spec.md s6.1). Each returns a
# band-limited reconstructed cube for a *fixed* hyper-parameter; the eval script
# oracle-selects the hyper-parameter per scene so every baseline is at its best.
# ============================================================================

def _perbin(cube: np.ndarray, fs: float, f0: float, frac_bw: float, fn) -> Tuple[np.ndarray, np.ndarray]:
    """Apply per-bin completion ``fn(M_bin) -> N x N`` over the in-band bins."""
    Mf, freqs = to_freq(cube, fs)
    band = band_bins(freqs, f0, frac_bw)
    Mf_rec = np.zeros_like(Mf)
    for fb in band:
        Mf_rec[:, :, fb] = fn(Mf[:, :, fb])
    return from_freq(Mf_rec, cube.shape[2]), band


def _soft_impute(M, obs, lam, *, n_iter=100, tol=1e-5, symmetric=True):
    """Singular-value-thresholding completion with a hard data-consistency step."""
    Mo = np.where(obs, M, 0.0).astype(complex)
    X = Mo.copy()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for _ in range(n_iter):
            if symmetric:
                X = 0.5 * (X + X.T)
            U, s, Vh = np.linalg.svd(X, full_matrices=False)
            s2 = np.maximum(s - lam, 0.0)
            Xr = (U * s2) @ Vh
            Xn = np.where(obs, Mo, Xr)
            if np.linalg.norm(Xn - X) / (np.linalg.norm(X) + 1e-12) < tol:
                X = Xn
                break
            X = Xn
        if symmetric:
            X = 0.5 * (X + X.T)
    return X


def reconstruct_nystrom(cube, tx_set, *, fs, f0, frac_bw, rank, rcond=1e-8):
    """Per-bin Nystrom completion with explicit (rank, rcond) — oracle-swept upstream."""
    fn = lambda M: lowrank_complete(symmetrize(M), tx_set, rank=rank, rcond=rcond)
    cube_rec, _ = _perbin(cube, fs, f0, frac_bw, fn)
    return cube_rec


def reconstruct_softimpute(cube, tx_set, *, fs, f0, frac_bw, lam_rel, n_iter=100):
    """Per-bin nuclear-norm (SVT) completion; ``lam`` set relative to each bin's sigma_max."""
    n = cube.shape[0]
    obs = observed_mask(n, np.asarray(tx_set))

    def fn(M):
        Ms = symmetrize(M)
        Mo = np.where(obs, Ms, 0.0)
        s0 = np.linalg.svd(Mo, compute_uv=False)
        lam = lam_rel * (s0[0] if s0.size else 0.0)
        return _soft_impute(Ms, obs, lam, n_iter=n_iter)

    cube_rec, _ = _perbin(cube, fs, f0, frac_bw, fn)
    return cube_rec


def reconstruct_joint_lowrank(cube, tx_set, *, fs, f0, frac_bw, lam_rel, n_iter=100):
    """Joint cross-frequency low-rank: SVT on the in-band M_f stack unfolded over tx.

    Exploits that all bins share the same scatterer positions (verified high
    adjacent-bin subspace overlap). This is the ceiling for the 'cross-frequency
    coupling' prior — if the net cannot beat THIS, coupling is not learning-unique.
    """
    n, _, n_t = cube.shape
    Mf, freqs = to_freq(cube, fs)
    band = band_bins(freqs, f0, frac_bw)
    obs = observed_mask(n, np.asarray(tx_set))

    bins = [symmetrize(Mf[:, :, fb]) for fb in band]
    B = np.concatenate(bins, axis=1)                  # N x (F*N)
    obs_stack = np.concatenate([obs] * len(band), axis=1)
    s0 = np.linalg.svd(np.where(obs_stack, B, 0.0), compute_uv=False)
    lam = lam_rel * (s0[0] if s0.size else 0.0)
    Bc = _soft_impute(B, obs_stack, lam, n_iter=n_iter, symmetric=False)

    Mf_rec = np.zeros_like(Mf)
    for k, fb in enumerate(band):
        blk = Bc[:, k * n:(k + 1) * n]
        Mf_rec[:, :, fb] = 0.5 * (blk + blk.T)        # re-impose per-bin reciprocity
    return from_freq(Mf_rec, n_t)


def reconstruct_steering_oracle(cube, tx_set, defects, array, *, fs, f0, frac_bw, c,
                                rcond=1e-3):
    """Parametric ceiling: oracle-SUPPORT steering fit (upper bound on any prior).

    Given the TRUE defect positions, fit a complex amplitude per defect per bin from
    the observed entries only (least squares), then predict the full matrix from the
    analytic steering model M_f = sum_k g_k(f) a_f(p_k) a_f(p_k)^T. This answers the
    identifiability question: if even this recovers the unobserved block at rank>K,
    the rank>K signal IS there from K transmits (room a learned/OMP method could
    reach); if it cannot, no method can. A real method must also recover the support.
    """
    Mf, freqs = to_freq(cube, fs)
    band = band_bins(freqs, f0, frac_bw)
    n, _, n_t = cube.shape
    obs = observed_mask(n, np.asarray(tx_set))
    ii, jj = np.where(obs)
    pos = array.positions
    P = np.array([(x, z) for (x, z, _) in defects], dtype=float)
    if P.size == 0:
        return np.zeros_like(cube)
    dx = pos[:, 0][:, None] - P[:, 0][None, :]
    dz = pos[:, 1][:, None] - P[:, 1][None, :]
    dist = np.maximum(np.sqrt(dx * dx + dz * dz), 1e-9)        # (N, D) element->defect
    Mf_rec = np.zeros_like(Mf)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for fb in band:
            w = 2.0 * np.pi * freqs[fb]
            A = np.exp(-1j * w * dist / c) / np.sqrt(dist)     # (N, D) steering atoms
            Gd = A[ii, :] * A[jj, :]                            # (|O|, D) over observed
            g, *_ = np.linalg.lstsq(Gd, Mf[ii, jj, fb], rcond=rcond)
            Mf_rec[:, :, fb] = (A * g) @ A.T                    # sum_k g_k a_k a_k^T
    return from_freq(Mf_rec, n_t)
