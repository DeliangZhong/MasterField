# matrix_master_field/free_fisher.py
"""M5c — free Fisher information helpers + the g=0 two-matrix-QM anchor.

Φ*[σ] = (4π²/3)∫σ³ (Voiculescu free Fisher information of a 1-D density). The
two-matrix-QM ground-state kinetic energy is ¼Φ* of the joint distribution; this
module holds the 1-D building block + the exact g=0 anchor. See
`derivations/m5c-two-matrix-qm.md`.
"""
import numpy as np

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy 2.0 renamed trapz


def phi_star_density(sigma, ys):
    """Free Fisher information of a 1-D probability density σ on grid ys: (4π²/3)∫σ³."""
    return float(4.0 * np.pi**2 / 3.0 * _trapz(np.asarray(sigma) ** 3, np.asarray(ys)))


def free_oscillator_anchor(m):
    """g=0 anchor: two free matrix oscillators H1=Tr P²+m²Tr X² (spectrum (2n+1)m).

    Returns the convention-independent large-N values: E/N²=2m, m[X̃²]=1/(2m),
    m[P̃²]=m/2 (per matrix m[X̃²], m[P̃²]; energy is the two-matrix total).
    """
    return {"energy": 2.0 * m, "m2": 1.0 / (2.0 * m), "p2": m / 2.0}
