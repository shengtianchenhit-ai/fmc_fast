"""Total Focusing Method (TFM) beamforming.

TFM is the known, differentiable delay-and-sum that maps an FMC cube to an image.
Phase-0 uses it as the downstream task: detection quality is measured on TFM
images, not on raw cube error.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.signal import hilbert

from .geometry import LinearArray


def pixel_grid(x_min: float, x_max: float, z_min: float, z_max: float,
               dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (gx, gz) 1-D pixel coordinate axes for the imaging region."""
    gx = np.arange(x_min, x_max + 0.5 * dx, dx)
    gz = np.arange(z_min, z_max + 0.5 * dx, dx)
    return gx, gz


def tfm_image(
    cube: np.ndarray,
    array: LinearArray,
    gx: np.ndarray,
    gz: np.ndarray,
    *,
    c: float,
    fs: float,
    tx_set: Optional[Sequence[int]] = None,
    analytic: bool = True,
) -> np.ndarray:
    """Beamform an FMC ``cube`` into a (n_z, n_x) envelope image.

    ``tx_set`` restricts the transmit elements summed over (used by the naive
    sparse-Tx baseline). All receivers are always summed. With ``analytic=True``
    the cube is Hilbert-transformed so the coherent sum yields an envelope.
    """
    n = array.n_elements
    if tx_set is None:
        tx_set = range(n)

    data = hilbert(cube, axis=2) if analytic else cube.astype(complex)
    n_t = data.shape[2]

    # Pixel coordinates flattened to (P, 2).
    gxx, gzz = np.meshgrid(gx, gz)  # (n_z, n_x)
    px = gxx.ravel()
    pz = gzz.ravel()
    n_pix = px.size

    pos = array.positions  # (N, 2)
    # Delay (in samples) from each element to each pixel: (N, P).
    d_e2p = np.sqrt((pos[:, 0][:, None] - px[None, :]) ** 2
                    + (pos[:, 1][:, None] - pz[None, :]) ** 2)
    idx = (d_e2p / c) * fs  # (N, P) one-way time-of-flight in samples

    img = np.zeros(n_pix, dtype=complex)
    all_j = np.arange(n)
    for i in tx_set:
        ts = idx[i][None, :] + idx          # (N, P) two-way sample index for each rx j
        i0 = np.floor(ts).astype(np.int64)
        frac = ts - i0
        valid = (i0 >= 0) & (i0 < n_t - 1)
        i0c = np.clip(i0, 0, n_t - 2)
        di = data[i]                        # (N, T)
        g0 = np.take_along_axis(di, i0c, axis=1)
        g1 = np.take_along_axis(di, i0c + 1, axis=1)
        contrib = (g0 * (1.0 - frac) + g1 * frac) * valid
        img += contrib.sum(axis=0)
    _ = all_j

    return np.abs(img).reshape(gz.size, gx.size)
