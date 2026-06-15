"""Synthetic test specimens (phantoms) for Phase 0 and Phase 1.

Phase 0 used a single side-drilled hole (SDH). Phase 1 needs harder, *coherent*
structure that pushes the per-frequency matrix M_f above rank K so low-rank
completion genuinely breaks (see docs/phase1_spec.md s1):

* F1 multi-SDH        -- n distinct point SDHs, pairwise separation >= 1.3*lambda;
                         rank(M_f) = n exactly.
* F2 crack            -- a tilted dense line of points; effective rank ~ space-
                         bandwidth product (a few), a learnable manifold.
* F3 closely-spaced   -- exactly two SDHs at sub-/near-resolution separation;
                         rank 2 but TFM-unresolvable (resolution headroom test).

Grain + noise are a *nuisance* floor (F4), not a rank source: TFM averages
incoherent clutter away. Every phantom carries an explicit ``defects`` list (the
coherent targets, for multi-target metrics) alongside the full ``scatterers`` list
(coherent + grain) fed to the forward model. Phantom-level splits never leak
(tx, rx) pairs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

Scatterer = Tuple[float, float, float]  # (x [m], z [m], reflectivity)


@dataclass
class Phantom:
    id: int
    family: str
    defects: List[Scatterer]                       # coherent targets only
    scatterers: List[Scatterer] = field(default_factory=list)  # coherent + grain
    c: float = 6300.0
    noise_db: Optional[float] = None

    @property
    def defect_xz(self) -> Tuple[float, float]:
        """First coherent defect location (back-compat with Phase-0 single-SDH code)."""
        return (self.defects[0][0], self.defects[0][1]) if self.defects else (0.0, 0.0)


# ----------------------------------------------------------------------------- helpers
def _rejection_sample(n: int, x_range, z_range, min_sep: float,
                      rng: np.random.Generator, max_tries: int = 20000) -> List[Tuple[float, float]]:
    """Sample up to ``n`` points in the box with pairwise spacing >= ``min_sep``."""
    pts: List[Tuple[float, float]] = []
    tries = 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        x = float(rng.uniform(*x_range))
        z = float(rng.uniform(*z_range))
        if all((x - px) ** 2 + (z - pz) ** 2 >= min_sep ** 2 for px, pz in pts):
            pts.append((x, z))
    return pts


def _logu(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _add_grain(scat: List[Scatterer], rng: np.random.Generator, *, n_grain: int,
               grain_refl: float, x_range, z_range) -> None:
    for _ in range(n_grain):
        gx = float(rng.uniform(x_range[0] * 1.3, x_range[1] * 1.3))
        gz = float(rng.uniform(z_range[0], z_range[1]))
        scat.append((gx, gz, grain_refl * float(rng.uniform(0.5, 1.5))))


# ----------------------------------------------------------------------------- families
def make_multi_defect(rng, *, n_range=(3, 12), x_range, z_range, min_sep,
                      refl_range=(0.3, 1.0)) -> List[Scatterer]:
    """F1: n distinct point SDHs, log-uniform reflectivity, separation >= min_sep."""
    n = int(rng.integers(n_range[0], n_range[1] + 1))
    pts = _rejection_sample(n, x_range, z_range, min_sep, rng)
    return [(x, z, _logu(rng, *refl_range)) for x, z in pts]


def make_crack(rng, *, length_range=(3e-3, 10e-3), tilt_deg=(-45.0, 45.0),
               n_points=(40, 60), x_range, z_range, taper=True,
               refl_range=(0.4, 1.0)) -> List[Scatterer]:
    """F2: a tilted dense line of points (a crack), effective rank ~ a few.

    ``tilt`` is measured from the horizontal; an optional |cos| aperture taper
    weights facets by their angle to the (vertical) insonification.
    """
    L = float(rng.uniform(*length_range))
    theta = np.deg2rad(float(rng.uniform(*tilt_deg)))
    m = int(rng.integers(n_points[0], n_points[1] + 1))
    # Centre placed so the whole crack stays inside the region.
    half = 0.5 * L
    cx = float(rng.uniform(x_range[0] + half, x_range[1] - half))
    cz = float(rng.uniform(z_range[0] + half, z_range[1] - half))
    t = np.linspace(-half, half, m)
    xs = cx + t * np.cos(theta)
    zs = cz + t * np.sin(theta)
    base = float(rng.uniform(*refl_range)) / np.sqrt(m)  # spread energy over facets
    if taper:
        # facet normal ~ perpendicular to the crack line; weight by |cos| to vertical.
        w = np.abs(np.cos(theta)) * 0.5 + 0.5
    else:
        w = 1.0
    return [(float(x), float(z), base * w) for x, z in zip(xs, zs)]


def make_closely_spaced_pair(rng, *, x_range, z_range, lam, sep_lam=(0.5, 1.5),
                             refl=1.0) -> List[Scatterer]:
    """F3: two equal SDHs separated by Delta in [sep_lam]*lambda, random orientation."""
    delta = float(rng.uniform(*sep_lam)) * lam
    ang = float(rng.uniform(0, np.pi))
    margin = delta
    cx = float(rng.uniform(x_range[0] + margin, x_range[1] - margin))
    cz = float(rng.uniform(z_range[0] + margin, z_range[1] - margin))
    dx, dz = 0.5 * delta * np.cos(ang), 0.5 * delta * np.sin(ang)
    return [(cx - dx, cz - dz, refl), (cx + dx, cz + dz, refl)]


# ----------------------------------------------------------------------------- dispatcher
def make_phantom(
    family: str,
    rng: np.random.Generator,
    *,
    c: float,
    f0: float,
    x_range=(-10e-3, 10e-3),
    z_range=(10e-3, 32e-3),
    pid: int = 0,
    grain: bool = False,
    n_grain_range=(20, 40),
    grain_refl: float = 0.04,
    noise_db: Optional[float] = None,
    **fam_kwargs,
) -> Phantom:
    """Build one phantom of the requested family with an optional grain+noise floor."""
    lam = c / f0
    min_sep = 1.3 * lam
    if family == "multi":
        defects = make_multi_defect(rng, x_range=x_range, z_range=z_range,
                                    min_sep=min_sep, **fam_kwargs)
    elif family == "crack":
        defects = make_crack(rng, x_range=x_range, z_range=z_range, **fam_kwargs)
    elif family == "pair":
        defects = make_closely_spaced_pair(rng, x_range=x_range, z_range=z_range,
                                           lam=lam, **fam_kwargs)
    elif family == "single":
        x = float(rng.uniform(*x_range)); z = float(rng.uniform(*z_range))
        defects = [(x, z, 1.0)]
    else:
        raise ValueError(f"unknown family {family!r}")

    scat = list(defects)
    if grain:
        ng = int(rng.integers(n_grain_range[0], n_grain_range[1] + 1))
        _add_grain(scat, rng, n_grain=ng, grain_refl=grain_refl,
                   x_range=x_range, z_range=z_range)
    return Phantom(id=pid, family=family, defects=defects, scatterers=scat,
                   c=c, noise_db=noise_db)


# ----------------------------------------------------------------------------- back-compat
def make_test_phantoms(
    n: int,
    *,
    x_range: Tuple[float, float],
    z_range: Tuple[float, float],
    reflectivity: float = 1.0,
    n_grain: int = 0,
    grain_refl: float = 0.05,
    seed: int = 0,
) -> List[Phantom]:
    """Phase-0 single-SDH set (kept so scripts/run_phase0.py is unchanged)."""
    rng = np.random.default_rng(seed)
    phantoms: List[Phantom] = []
    for k in range(n):
        x = float(rng.uniform(*x_range))
        z = float(rng.uniform(*z_range))
        scat: List[Scatterer] = [(x, z, reflectivity)]
        for _ in range(n_grain):
            gx = float(rng.uniform(x_range[0] * 1.3, x_range[1] * 1.3))
            gz = float(rng.uniform(z_range[0], z_range[1]))
            scat.append((gx, gz, grain_refl * float(rng.uniform(0.5, 1.5))))
        phantoms.append(Phantom(id=k, family="single", defects=[(x, z, reflectivity)],
                                scatterers=scat))
    return phantoms
