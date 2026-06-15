"""Fast point-scatterer forward model for FMC.

This is the Phase-0 solver: every defect is a point reflector, each A-scan is a
sum of delayed/attenuated pulses. It is fast, runs anywhere, and is *exactly*
reciprocal (s_ij == s_ji), which is what we need to build and validate the whole
pipeline (TFM, sparse-Tx, baselines, metrics) before bringing in k-Wave for
realism in Phase-1.

Convention for the data cube: ``A[i, j, t]`` = signal transmitted on element i,
received on element j, at time sample t.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .geometry import LinearArray, gabor_pulse

# A scatterer is (x [m], z [m], reflectivity).
Scatterer = Tuple[float, float, float]


def simulate_fmc(
    array: LinearArray,
    scatterers: Sequence[Scatterer],
    *,
    c: float,
    fs: float,
    n_t: int,
    f0: float,
    n_cycles: float = 2.0,
    noise_db: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate a full N x N x T FMC cube for the given point scatterers.

    Amplitude per leg follows 2-D cylindrical spreading (1/sqrt(distance)), so the
    pair amplitude is ``refl / sqrt(d_tx * d_rx)``. ``noise_db`` (if given) adds
    white Gaussian noise at that signal-to-noise ratio.
    """
    if rng is None:
        rng = np.random.default_rng()

    pos = array.positions  # (N, 2)
    n = array.n_elements
    t = np.arange(n_t) / fs  # (T,)
    cube = np.zeros((n, n, n_t), dtype=np.float64)

    for sx, sz, refl in scatterers:
        d = np.sqrt((pos[:, 0] - sx) ** 2 + (pos[:, 1] - sz) ** 2)  # (N,) element->scatterer
        d = np.maximum(d, 1e-9)
        for i in range(n):
            tau = (d[i] + d) / c          # (N,) round-trip delay for each rx j
            amp = refl / np.sqrt(d[i] * d)  # (N,) spreading
            # pulse evaluated at t - tau for every receiver j: (N, T)
            cube[i] += amp[:, None] * gabor_pulse(t[None, :] - tau[:, None], f0, n_cycles)

    if noise_db is not None:
        sig_rms = np.sqrt(np.mean(cube ** 2)) + 1e-12
        noise_rms = sig_rms * 10.0 ** (-noise_db / 20.0)
        cube += rng.normal(0.0, noise_rms, size=cube.shape)

    return cube


def required_n_t(array: LinearArray, max_depth: float, *, c: float, fs: float,
                 margin: float = 1.2) -> int:
    """Samples needed so the deepest two-way path fits in the time window."""
    half_aperture = 0.5 * (array.n_elements - 1) * array.pitch
    max_leg = np.hypot(half_aperture + abs(array.x).max(), max_depth)
    t_max = 2.0 * max_leg / c
    return int(np.ceil(margin * t_max * fs))
