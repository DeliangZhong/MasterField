import numpy as np

from matrix_master_field import one_matrix as om
from matrix_master_field.operator_field import one_matrix_master_field_from_moments


def test_gaussian_master_field_operator_recovers_catalan():
    target = om.gaussian_moments(8)
    _, model = one_matrix_master_field_from_moments(target, fock_length=10)
    assert np.max(np.abs(model[:9] - target[:9])) < 1e-6


def test_quartic_master_field_operator_recovers_exact_moments():
    g = 0.5
    target = om.quartic_moments_from_sd(g, max_power=8)
    _, model = one_matrix_master_field_from_moments(target, fock_length=12)
    err = np.max(np.abs(model[:9] - target[:9]))
    assert err < 1e-6, f"max err {err:.2e}"
