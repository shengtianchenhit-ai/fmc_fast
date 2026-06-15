"""Metrics — three layers, per the plan.

Reconstruction fidelity is a proxy; what matters is whether detection survives.
So we expose a *phase-sensitive* cube error plus TFM-image detection metrics.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .freqdomain import band_bins, to_freq


def nrmse_complex(rec: np.ndarray, true: np.ndarray) -> float:
    """Phase-sensitive normalised RMSE over complex values: ||rec-true|| / ||true||."""
    num = np.linalg.norm((rec - true).ravel())
    den = np.linalg.norm(true.ravel()) + 1e-12
    return float(num / den)


def band_nrmse(cube_rec: np.ndarray, cube_true: np.ndarray, *,
               fs: float, f0: float, frac_bw: float) -> float:
    """Cube NRMSE evaluated on the in-band complex spectrum (phase-sensitive)."""
    Mr, freqs = to_freq(cube_rec, fs)
    Mt, _ = to_freq(cube_true, fs)
    band = band_bins(freqs, f0, frac_bw)
    return nrmse_complex(Mr[:, :, band], Mt[:, :, band])


def band_nrmse_blocks(cube_rec: np.ndarray, cube_true: np.ndarray, observed_mask: np.ndarray,
                      *, fs: float, f0: float, frac_bw: float) -> Dict[str, float]:
    """Phase-sensitive in-band NRMSE split by observed / unobserved entries.

    The *unobserved-block* NRMSE is the Phase-1 headline axis (Claim A): naive
    leaves it at 1.0 (predicts nothing), and low-rank completion only recovers it
    when rank(M_f) <= K. ``observed_mask`` is the (N, N) frame from
    ``sampling.observed_mask``.
    """
    Mr, freqs = to_freq(cube_rec, fs)
    Mt, _ = to_freq(cube_true, fs)
    band = band_bins(freqs, f0, frac_bw)
    R = Mr[:, :, band]
    T = Mt[:, :, band]
    obs = observed_mask[:, :, None]
    unobs = ~observed_mask
    unobs3 = unobs[:, :, None]

    def _nrmse(mask3):
        num = np.linalg.norm((R - T)[np.broadcast_to(mask3, R.shape)])
        den = np.linalg.norm(T[np.broadcast_to(mask3, T.shape)]) + 1e-12
        return float(num / den)

    return {
        "nrmse_overall": nrmse_complex(R, T),
        "nrmse_unobs": _nrmse(unobs3),
        "nrmse_obs": _nrmse(obs),
        "frac_unobs": float(unobs.mean()),
    }


def defect_metrics(
    img: np.ndarray,
    gx: np.ndarray,
    gz: np.ndarray,
    defect_xz: Tuple[float, float],
    *,
    c: float,
    f0: float,
    roi_radius: float = 2.0e-3,
) -> Dict[str, float]:
    """Detection metrics for a single known defect in a TFM envelope image.

    Returns peak amplitude, defect SNR / contrast (dB), highest artifact outside
    the ROI (dB rel. peak), API (-6 dB area / wavelength^2), and -6 dB extents.
    """
    gxx, gzz = np.meshgrid(gx, gz)
    dx_pix = float(gx[1] - gx[0])
    dz_pix = float(gz[1] - gz[0])
    lam = c / f0

    dist = np.hypot(gxx - defect_xz[0], gzz - defect_xz[1])
    roi = dist <= roi_radius
    bg = dist > 2.0 * roi_radius

    if not roi.any():
        roi = dist <= (2.0 * max(dx_pix, dz_pix))

    peak = float(img[roi].max())
    bg_vals = img[bg]
    bg_rms = float(np.sqrt(np.mean(bg_vals ** 2))) + 1e-12
    bg_mean = float(np.mean(bg_vals)) + 1e-12
    bg_max = float(bg_vals.max()) if bg_vals.size else 0.0

    # -6 dB (half-amplitude) footprint of the indication, inside the ROI.
    half = peak / 2.0
    roi_hot = roi & (img >= half)
    if roi_hot.any():
        xs = gxx[roi_hot]
        zs = gzz[roi_hot]
        lat6 = float(xs.max() - xs.min())
        ax6 = float(zs.max() - zs.min())
    else:
        lat6 = ax6 = 0.0

    # API: total -6 dB area over the whole image, normalised by wavelength^2.
    api = float((img >= half).sum()) * dx_pix * dz_pix / (lam ** 2)

    return {
        "peak": peak,
        "snr_db": 20.0 * np.log10(peak / bg_rms),
        "contrast_db": 20.0 * np.log10(peak / bg_mean),
        "artifact_db": 20.0 * np.log10((bg_max + 1e-12) / (peak + 1e-12)),
        "api": api,
        "lat6_mm": lat6 * 1e3,
        "ax6_mm": ax6 * 1e3,
    }
