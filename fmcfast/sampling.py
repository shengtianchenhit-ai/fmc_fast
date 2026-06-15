"""Sparse-Tx transmit selection and the resulting observed-entry mask."""
from __future__ import annotations

import numpy as np


def uniform_tx_set(n: int, k: int) -> np.ndarray:
    """Pick ``k`` transmit elements spread uniformly across the aperture.

    Endpoints are always included so the effective aperture is preserved.
    Returns a sorted array of unique indices (length may be < k only if k > n).
    """
    k = min(k, n)
    idx = np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))
    return idx


def observed_mask(n: int, tx_set: np.ndarray) -> np.ndarray:
    """Boolean (N, N) mask of entries we *know* after a sparse-Tx acquisition.

    We physically measure every row ``i`` in ``tx_set`` (transmit i, receive all).
    Reciprocity (M_ij = M_ji) then hands us every column ``j`` in ``tx_set`` for
    free. So the observed set is the union of those rows and columns — a "frame".
    """
    mask = np.zeros((n, n), dtype=bool)
    mask[tx_set, :] = True
    mask[:, tx_set] = True
    return mask
