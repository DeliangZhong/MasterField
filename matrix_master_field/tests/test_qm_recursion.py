"""M5a — the stationarity recursion (HHK Eq 6) holds on exact-diag moments."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from matrix_master_field.loss import qm_anharmonic_recursion_residual  # noqa: E402
from matrix_master_field.qm_fock import ground_state, moment, xp_operators  # noqa: E402


def test_recursion_residual_vanishes_on_exact_moments():
    # The recursion is EXACT for true eigenstate moments; the residual here is bounded
    # by the finite-K exact-diag referee's truncation error on the highest moment used
    # (m_12), which is ~1e-9 (the t=1 residual alone is ~1e-13). 1e-8 is an 8-digit check.
    for g in (0.0, 0.5, 1.0, 2.0):
        E0, omega = ground_state(120, g)
        m = [moment(omega, k) for k in range(13)]
        res = qm_anharmonic_recursion_residual(m, E0, g)
        assert max(abs(r) for r in res) < 1e-8


def test_recursion_t1_closed_form():
    # t=1: 4E - 8 m2 - 12 g m4 = 0  (m0=1).
    g = 1.0
    E0, omega = ground_state(80, g)
    m2, m4 = moment(omega, 2), moment(omega, 4)
    assert abs(4 * E0 - 8 * m2 - 12 * g * m4) < 1e-9


def test_energy_relation_D1_operator_form():
    # <x^{t-1} p^2> = E m_{t-1} - m_{t+1} - g m_{t+3}, checked with the operators directly.
    g, K = 1.0, 60
    E0, omega = ground_state(K, g)
    X, P = xp_operators(K + 6)
    psi = np.zeros(X.shape[0], dtype=complex)
    psi[: K + 1] = np.asarray(omega)
    P2 = np.asarray(P @ P)
    Xn = np.asarray(X)
    for t in (1, 3, 5):
        lhs = psi.conj() @ np.linalg.matrix_power(Xn, t - 1) @ P2 @ psi
        rhs = E0 * moment(omega, t - 1) - moment(omega, t + 1) - g * moment(omega, t + 3)
        assert abs(lhs.real - rhs) < 1e-6
