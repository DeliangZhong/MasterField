"""M5c — anchor (g=0 free matrix oscillator) + free-Fisher kinetic identity (T1, T4)."""
import numpy as np

from matrix_master_field.free_fisher import (
    free_oscillator_anchor,
    phi_star_density,
)


def test_g0_anchor_is_2m():
    # Two free matrix oscillators: spectrum (2n+1)m per mode, ground m, x2 matrices.
    for m in (0.5, 1.0, 2.0):
        a = free_oscillator_anchor(m)
        assert abs(a["energy"] - 2.0 * m) < 1e-12       # E/N² = 2m
        assert abs(a["m2"] - 1.0 / (2.0 * m)) < 1e-12   # m[X̃²] = 1/(2m)
        assert abs(a["p2"] - m / 2.0) < 1e-12           # m[P̃²] = m/2


def test_phi_star_semicircle_is_one():
    # Standard semicircle on [-2,2], variance 1: Φ* = 1 (Voiculescu).
    ys = np.linspace(-2.0, 2.0, 200001)
    sigma = np.sqrt(np.clip(4.0 - ys**2, 0.0, None)) / (2.0 * np.pi)
    assert abs(phi_star_density(sigma, ys) - 1.0) < 1e-4


def test_quarter_phi_star_reduces_to_m5b_kinetic():
    # M5b g=0 density σ=(1/π)√(2−y²) (variance ½): ¼Φ* = ∫π²σ³/3 = ½ = m[P̃²].
    ys = np.linspace(-np.sqrt(2.0), np.sqrt(2.0), 200001)
    sigma = np.sqrt(np.clip(2.0 - ys**2, 0.0, None)) / np.pi
    quarter_phi = 0.25 * phi_star_density(sigma, ys)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    m5b_kinetic = _trapz(np.pi**2 * sigma**3 / 3.0, ys)
    assert abs(quarter_phi - m5b_kinetic) < 1e-4
    assert abs(quarter_phi - 0.5) < 1e-3
