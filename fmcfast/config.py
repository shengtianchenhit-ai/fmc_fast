"""Tiny config loader (YAML -> nested dict) and object builders."""
from __future__ import annotations

from typing import Any, Dict

import yaml

from .geometry import LinearArray
from .tfm import pixel_grid


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_array(cfg: Dict[str, Any]) -> LinearArray:
    a = cfg["array"]
    return LinearArray(n_elements=int(a["n_elements"]), pitch=float(a["pitch"]))


def build_grid(cfg: Dict[str, Any]):
    g = cfg["grid"]
    return pixel_grid(float(g["x_min"]), float(g["x_max"]),
                      float(g["z_min"]), float(g["z_max"]), float(g["dx"]))
