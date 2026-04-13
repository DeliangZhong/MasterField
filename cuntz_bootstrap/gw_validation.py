"""Exact Gross-Witten moment targets at large N.

Strong coupling (lambda >= 1, ungapped phase):
    rho(theta) = (1 + cos(theta) / lambda) / (2 pi)
    w_1 = 1 / (2 lambda), w_k = 0 for k >= 2

Weak coupling (lambda < 1, gapped phase):
    sin^2(theta_c / 2) = lambda
    rho(theta) = (1 / (pi lambda)) cos(theta / 2) sqrt(lambda - sin^2(theta / 2))
                 on |theta| <= theta_c, else 0
    w_1 = 1 - lambda / 2, w_2 = (1 - lambda)^2

Higher-order weak-coupling moments need numerical integration.

Wilson loops in GW are real (symmetric eigenvalue density rho(theta) = rho(-theta)).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def gw_moments(lam: float, K: int) -> np.ndarray:
    """Return [w_0, w_1, ..., w_K] for GW at coupling lambda = lam."""
    w = np.zeros(K + 1, dtype=np.float64)
    w[0] = 1.0
    if lam >= 1.0:
        if K >= 1:
            w[1] = 1.0 / (2.0 * lam)
        # w_k = 0 for k >= 2 (closed-form, strong-coupling phase)
        return w

    # Weak-coupling phase: numerical integration over the gapped support.
    theta_c = 2.0 * np.arcsin(np.sqrt(lam))

    def rho(theta: float) -> float:
        s = np.sin(theta / 2.0)
        inside = lam - s * s
        if inside <= 0.0:
            return 0.0
        return (1.0 / (np.pi * lam)) * np.cos(theta / 2.0) * np.sqrt(inside)

    for k in range(1, K + 1):
        val, _ = quad(
            lambda t, k=k: rho(t) * np.cos(k * t), -theta_c, theta_c
        )
        w[k] = val

    return w
