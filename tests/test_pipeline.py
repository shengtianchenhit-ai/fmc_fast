"""End-to-end smoke test: the Phase-0 loop must run and behave sanely on a tiny case."""
import numpy as np

from fmcfast import (
    LinearArray,
    metrics,
    reconstruct_lowrank,
    simulate_fmc,
    tfm_image,
    uniform_tx_set,
)
from fmcfast.tfm import pixel_grid

C, FS, F0, NT = 6300.0, 50e6, 5e6, 1024


def _setup():
    array = LinearArray(n_elements=32, pitch=0.6e-3)
    gx, gz = pixel_grid(-8e-3, 8e-3, 4e-3, 28e-3, 0.5e-3)
    defect = (1.5e-3, 18.0e-3)
    cube = simulate_fmc(array, [(*defect, 1.0)], c=C, fs=FS, n_t=NT, f0=F0, n_cycles=2.0)
    return array, gx, gz, defect, cube


def test_full_tfm_focuses_on_defect():
    array, gx, gz, defect, cube = _setup()
    img = tfm_image(cube, array, gx, gz, c=C, fs=FS)
    iz, ix = np.unravel_index(np.argmax(img), img.shape)
    assert abs(gx[ix] - defect[0]) < 1e-3   # peak within 1 mm laterally
    assert abs(gz[iz] - defect[1]) < 1e-3   # and axially
    m = metrics.defect_metrics(img, gx, gz, defect, c=C, f0=F0)
    assert m["snr_db"] > 12.0               # clean full-aperture image


def test_lowrank_beats_naive_at_high_accel():
    # At high acceleration the naive image collapses (grating lobes) while low-rank
    # reconstructs the near-rank-1 cube faithfully and keeps imaging cleanly.
    array, gx, gz, defect, cube = _setup()
    n = array.n_elements
    S = uniform_tx_set(n, n // 8)           # 8x acceleration

    naive = tfm_image(cube, array, gx, gz, c=C, fs=FS, tx_set=S)
    cube_rec, _ = reconstruct_lowrank(cube, S, fs=FS, f0=F0, frac_bw=0.6, rank=4)
    lr = tfm_image(cube_rec, array, gx, gz, c=C, fs=FS)

    m_naive = metrics.defect_metrics(naive, gx, gz, defect, c=C, f0=F0)
    m_lr = metrics.defect_metrics(lr, gx, gz, defect, c=C, f0=F0)
    nrmse = metrics.band_nrmse(cube_rec, cube, fs=FS, f0=F0, frac_bw=0.6)

    assert nrmse < 0.1                       # faithful cube reconstruction
    assert m_lr["snr_db"] > m_naive["snr_db"]            # cleaner image
    assert m_lr["artifact_db"] < m_naive["artifact_db"]  # fewer grating lobes
