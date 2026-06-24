import numpy as np
import pytest

from matrix_master_field import one_matrix as om


def test_gaussian_moments_are_catalan():
    m = om.gaussian_moments(10)
    # m_{2k} = Catalan C_k: 1, 1, 2, 5, 14, 42
    assert np.isclose(m[0], 1.0)
    assert np.isclose(m[2], 1.0)
    assert np.isclose(m[4], 2.0)
    assert np.isclose(m[6], 5.0)
    assert np.isclose(m[8], 14.0)
    assert np.isclose(m[10], 42.0)
    # odd moments vanish
    assert np.allclose(m[1:11:2], 0.0)


def test_gaussian_free_cumulants():
    kappa = om.r_transform_from_moments(om.gaussian_moments(10))
    # Gaussian/semicircle: kappa_2 = 1, all others 0
    assert np.isclose(kappa[1], 0.0, atol=1e-9)  # kappa_1
    assert np.isclose(kappa[2], 1.0, atol=1e-9)  # kappa_2
    assert np.allclose(kappa[3:], 0.0, atol=1e-9)


def test_quartic_sd_moments_match_density_moments():
    g = 0.5
    m_sd = om.quartic_moments_from_sd(g, max_power=8)
    x, rho = om.quartic_eigenvalue_density(g, n_points=4000)
    m_rho = om.moments_from_density(x, rho, 8)
    for k in range(0, 9, 2):
        assert abs(m_sd[k] - m_rho[k]) < 1e-3, f"m_{k}: {m_sd[k]} vs {m_rho[k]}"


def test_gaussian_resolvent_asymptotics_both_sheets():
    # R(ζ) ~ 1/ζ as |ζ|→∞ on BOTH real sheets (the bug: principal branch gave
    # R≈ζ for ζ<0). Check positive, negative, and complex ζ.
    for z in [50.0, -50.0, 40 + 1j, -40 - 2j]:
        R = complex(om.gaussian_resolvent(z))
        assert abs(R - 1.0 / z) < 1e-2, f"R({z})={R} vs 1/z={1/z}"


def test_gaussian_resolvent_density_positive_on_cut():
    # ρ(x) = -(1/π) Im R(x+i0⁺) must be positive on the support [-2,2].
    for x in [0.0, 1.0, -1.0]:
        R = complex(om.gaussian_resolvent(x + 1j * 1e-7))
        assert -R.imag / np.pi > 0.0, f"density at {x} not positive"


def test_quartic_resolvent_raises():
    with pytest.raises(NotImplementedError):
        om.quartic_resolvent(2.0, 0.5)
