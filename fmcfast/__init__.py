"""fmcfast — learning-accelerated FMC acquisition (Phase-0 baseline toolkit).

Phase-0 deliverable: a solver-agnostic pipeline that turns full-matrix-capture
(FMC) cubes into TFM images, sub-samples the transmit aperture (sparse-Tx),
reconstructs with cheap baselines, and scores detection quality vs acceleration.
"""

from .geometry import LinearArray, gabor_pulse
from .forward import simulate_fmc
from .tfm import tfm_image, pixel_grid
from .sampling import uniform_tx_set, observed_mask
from .freqdomain import to_freq, from_freq, band_bins, symmetrize
from .completion import lowrank_complete
from .baselines import reconstruct_naive, reconstruct_lowrank
from . import metrics

__all__ = [
    "LinearArray",
    "gabor_pulse",
    "simulate_fmc",
    "tfm_image",
    "pixel_grid",
    "uniform_tx_set",
    "observed_mask",
    "to_freq",
    "from_freq",
    "band_bins",
    "symmetrize",
    "lowrank_complete",
    "reconstruct_naive",
    "reconstruct_lowrank",
    "metrics",
]
