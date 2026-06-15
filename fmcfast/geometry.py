"""Array geometry and excitation pulse."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearArray:
    """A linear phased-array of point-like elements on the surface (z = 0)."""

    n_elements: int
    pitch: float  # element spacing [m]

    @property
    def x(self) -> np.ndarray:
        """Lateral element coordinates, centred on the origin [m]."""
        n = self.n_elements
        return (np.arange(n) - (n - 1) / 2.0) * self.pitch

    @property
    def positions(self) -> np.ndarray:
        """(N, 2) array of element (x, z) coordinates."""
        return np.stack([self.x, np.zeros(self.n_elements)], axis=1)


def gabor_pulse(t: np.ndarray, f0: float, n_cycles: float = 2.0) -> np.ndarray:
    """Gaussian-windowed cosine centred at t = 0.

    ``n_cycles`` sets the temporal width (smaller -> broader band). The pulse is
    symmetric about zero; arrival times are imposed later via ``t - tau``.
    """
    sigma = n_cycles / (2.0 * f0)
    return np.cos(2.0 * np.pi * f0 * t) * np.exp(-(t ** 2) / (2.0 * sigma ** 2))
