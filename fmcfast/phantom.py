"""Synthetic test specimens (phantoms).

Phase-0 main set: a single side-drilled-hole (SDH) per flat specimen, modelled as
a point reflector, with optional weak grain scatterers. Each phantom is an
independent defect configuration, so a phantom-level split never leaks (tx, rx)
pairs across train/test (plan section 2.3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

Scatterer = Tuple[float, float, float]


@dataclass
class Phantom:
    id: int
    defect_xz: Tuple[float, float]
    scatterers: List[Scatterer] = field(default_factory=list)


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
    """Draw ``n`` independent single-SDH phantoms with random defect placement."""
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
        phantoms.append(Phantom(id=k, defect_xz=(x, z), scatterers=scat))
    return phantoms
