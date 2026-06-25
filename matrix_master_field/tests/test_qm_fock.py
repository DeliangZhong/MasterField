"""M5a — truncated bosonic oscillator Fock space (H=p^2+x^2+g x^4, [x,p]=i)."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from matrix_master_field.qm_fock import (  # noqa: E402
    ground_state,
    hamiltonian_anharmonic,
    ladder,
    moment,
    xp_operators,
)


def test_ladder_canonical():
    a, adag = ladder(10)
    comm = np.asarray(a @ adag - adag @ a)
    # [a,adag]=I on the interior (levels 0..K-1); only the top corner is truncated.
    assert np.allclose(np.diag(comm)[:10], 1.0, atol=1e-12)


def test_xp_commutator_interior_is_i():
    K = 30
    X, P = xp_operators(K)
    comm = np.asarray(X @ P - P @ X)
    assert np.allclose(comm[:K, :K], 1j * np.eye(K), atol=1e-12)


def test_xp_hermitian():
    X, P = xp_operators(20)
    assert np.allclose(np.asarray(X), np.conj(np.asarray(X)).T, atol=1e-12)
    assert np.allclose(np.asarray(P), np.conj(np.asarray(P)).T, atol=1e-12)


def test_truncation_corner_is_minus_iK():
    K = 12
    X, P = xp_operators(K)
    comm = np.asarray(X @ P - P @ X)
    assert np.isclose(comm[K, K], -1j * K, atol=1e-9)


def test_g0_ground_state_exact():
    E0, omega = ground_state(40, 0.0)
    assert np.isclose(E0, 1.0, atol=1e-9)
    assert np.isclose(moment(omega, 2), 0.5, atol=1e-9)
    assert np.isclose(moment(omega, 4), 0.75, atol=1e-9)


def test_g0_low_spectrum():
    H = np.asarray(hamiltonian_anharmonic(40, 0.0))
    w = np.sort(np.linalg.eigvalsh(H))
    assert np.allclose(w[:4], [1.0, 3.0, 5.0, 7.0], atol=1e-7)


def test_g1_reference_values():
    E0, omega = ground_state(60, 1.0)
    assert np.isclose(E0, 1.392352, atol=1e-5)  # HHK reference; verified in the spec
    assert np.isclose(moment(omega, 2), 0.305814, atol=1e-5)


def test_variational_upper_bound_monotone():
    Es = [ground_state(K, 1.0)[0] for K in (8, 16, 32, 60)]
    for lo, hi in zip(Es[1:], Es[:-1]):
        assert lo <= hi + 1e-12  # non-increasing in K
    assert Es[-1] >= 1.392352 - 1e-6  # still an upper bound


def test_odd_moments_vanish():
    _, omega = ground_state(40, 1.0)
    assert abs(moment(omega, 1)) < 1e-9
    assert abs(moment(omega, 3)) < 1e-9
