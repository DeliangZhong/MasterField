import numpy as np

from matrix_master_field import one_matrix as om
from matrix_master_field.loss import one_matrix_sd_residual


def test_gaussian_exact_moments_zero_residual():
    m = om.gaussian_moments(8)
    r = float(one_matrix_sd_residual(m, [0.0, 1.0]))
    assert r < 1e-9, f"residual {r}"


def test_quartic_exact_moments_small_residual():
    g = 0.5
    m = om.quartic_moments_from_sd(g, max_power=8)
    r = float(one_matrix_sd_residual(m, [0.0, 1.0, 0.0, g]))
    assert r < 1e-4, f"residual {r}"


def test_perturbed_moments_have_positive_residual():
    m = om.gaussian_moments(8).copy()
    m[2] += 0.3  # violate m_2 = 1
    r = float(one_matrix_sd_residual(m, [0.0, 1.0]))
    assert r > 1e-3, f"residual {r}"
