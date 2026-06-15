"""Per-frequency low-rank completion (the baseline to beat).

The sparse-Tx + reciprocity sampling gives us *entire rows and columns* of the
N x N matrix M_f for the transmitted set S. For a low-rank, complex-symmetric M_f
(M_f = M_f^T), the Nyström extension reconstructs the unobserved block directly
and is *exact* whenever rank(M_f) <= |S|:

    M_hat = C @ pinv(W) @ C.T ,   C = M[:, S] ,   W = M[S, S]

No iteration, no step size, no rank-overfitting — cheap and strong, which is what
makes it the most feared opponent (plan section 4). A rank cap / rcond truncates
the pseudo-inverse so noise in the tiny singular values is not amplified.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _sym_pinv(W: np.ndarray, rank: Optional[int], rcond: float) -> np.ndarray:
    """Truncated pseudo-inverse keeping at most ``rank`` singular directions."""
    U, s, Vh = np.linalg.svd(W)
    if s.size == 0:
        return np.zeros_like(W)
    tol = rcond * s[0]
    r = int(np.count_nonzero(s > tol))
    if rank is not None:
        r = min(r, int(rank))
    if r == 0:
        return np.zeros_like(W)
    return (Vh[:r].conj().T * (1.0 / s[:r])) @ U[:, :r].conj().T


def lowrank_complete(
    M: np.ndarray,
    tx_set: np.ndarray,
    *,
    rank: Optional[int] = None,
    rcond: float = 1e-8,
) -> np.ndarray:
    """Nyström completion of a frame-sampled complex-symmetric matrix.

    ``M`` need only be correct on rows/columns in ``tx_set``; everything else is
    reconstructed. Observed rows/columns are reproduced (near-)exactly.
    """
    S = np.asarray(tx_set)
    # numpy 2.0's complex matmul raises spurious FPE flags ("divide by zero") even
    # for finite, correct results; silence them locally rather than globally.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        C = M[:, S]                   # (N, K) — observed columns
        W = M[np.ix_(S, S)]           # (K, K) — observed core
        Winv = _sym_pinv(W, rank, rcond)
        return C @ Winv @ C.T
