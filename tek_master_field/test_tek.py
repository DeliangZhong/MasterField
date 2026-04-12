"""Unit and integration tests for tek_master_field.

Mirrors the pytest style used in master_field/test_qcd2.py:
    - @pytest.mark.unit for pure-Python structural tests (fast)
    - @pytest.mark.integration for physics validation (medium)
    - @pytest.mark.slow for long-running phases (optional)
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest
from jax import random

jax.config.update("jax_enable_x64", True)

from gross_witten import solve_gw  # noqa: E402
from observables import (  # noqa: E402
    center_symmetry_order,
    polyakov_loop,
    wilson_loop_plaquette,
    wilson_loop_rectangular,
)
from tek import (  # noqa: E402
    build_clock_matrix,
    build_link_matrices,
    build_twist,
    hermitianize,
    init_H_list_random,
    init_H_list_zero,
    plaquette_average,
    tek_loss,
)


# ═══════════════════════════════════════════════════════════════
# Unit: clock matrix
# ═══════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.parametrize("N", [9, 25, 49, 121])
def test_clock_matrix_unitary(N: int):
    Gamma = build_clock_matrix(N)
    eye = jnp.eye(N, dtype=jnp.complex128)
    err = float(jnp.linalg.norm(Gamma @ jnp.conj(Gamma.T) - eye))
    assert err < 1e-12, f"Γ not unitary at N={N}: err={err:.3e}"


@pytest.mark.unit
@pytest.mark.parametrize("N", [9, 25, 49, 121])
def test_clock_matrix_periodic(N: int):
    Gamma = build_clock_matrix(N)
    Gamma_N = jnp.linalg.matrix_power(Gamma, N)
    eye = jnp.eye(N, dtype=jnp.complex128)
    err = float(jnp.linalg.norm(Gamma_N - eye))
    assert err < 1e-10, f"Γ^N ≠ I at N={N}: err={err:.3e}"


@pytest.mark.unit
def test_clock_matrix_eigenvalues():
    N = 7
    Gamma = build_clock_matrix(N)
    eigs = jnp.linalg.eigvals(Gamma)
    phases = sorted([float(jnp.angle(e)) for e in eigs])
    expected = sorted([
        float(jnp.angle(jnp.exp(2j * jnp.pi * k / N))) for k in range(N)
    ])
    for a, b in zip(phases, expected):
        assert abs(a - b) < 1e-12


# ═══════════════════════════════════════════════════════════════
# Unit: twist tensor
# ═══════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.parametrize("D,N,L,k", [(2, 9, 3, 1), (2, 49, 7, 1), (3, 25, 5, 1), (4, 49, 7, 1), (4, 121, 11, 3)])
def test_twist_antisymmetry(D: int, N: int, L: int, k: int):
    z = build_twist(D, N, L, k=k)
    for mu in range(D):
        for nu in range(D):
            err = float(abs(z[mu, nu] - jnp.conj(z[nu, mu])))
            assert err < 1e-14, f"z[{mu},{nu}] ≠ conj(z[{nu},{mu}]) at D={D}, N={N}: err={err:.3e}"


@pytest.mark.unit
def test_twist_diagonal_unit():
    z = build_twist(D=4, N=49, L=7, k=1)
    for mu in range(4):
        assert abs(z[mu, mu] - 1.0) < 1e-14


@pytest.mark.unit
def test_twist_wrong_N_raises():
    with pytest.raises(ValueError, match="Expected N = L²"):
        build_twist(D=2, N=10, L=3)


@pytest.mark.unit
def test_twist_wrong_D_raises():
    with pytest.raises(ValueError, match="D must be"):
        build_twist(D=5, N=49, L=7)


# ═══════════════════════════════════════════════════════════════
# Unit: hermitianize
# ═══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_hermitianize_idempotent():
    key = random.PRNGKey(0)
    A = random.normal(key, (7, 7)) + 1j * random.normal(random.PRNGKey(1), (7, 7))
    H = hermitianize(A)
    H2 = hermitianize(H)
    assert float(jnp.linalg.norm(H - H2)) < 1e-14
    # And H is Hermitian
    assert float(jnp.linalg.norm(H - jnp.conj(H.T))) < 1e-14


# ═══════════════════════════════════════════════════════════════
# Unit: link matrices unitarity
# ═══════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.parametrize("D,N,L", [(2, 9, 3), (3, 25, 5), (4, 49, 7)])
def test_link_matrices_unitary(D: int, N: int, L: int):
    Gamma = build_clock_matrix(N)
    key = random.PRNGKey(42)
    H_list = init_H_list_random(D, N, key, scale=0.3)
    U = build_link_matrices(H_list, Gamma)
    assert len(U) == D
    eye = jnp.eye(N, dtype=jnp.complex128)
    for mu in range(D):
        err = float(jnp.linalg.norm(U[mu] @ jnp.conj(U[mu].T) - eye))
        assert err < 1e-10, f"U_{mu+1} not unitary: err={err:.3e}"


# ═══════════════════════════════════════════════════════════════
# Unit: H=0 plaquette = mean Re(z_μν)
# ═══════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.parametrize("D,N,L,k", [(2, 9, 3, 1), (2, 49, 7, 1), (3, 25, 5, 1), (4, 49, 7, 1)])
def test_plaquette_at_H0_equals_mean_Re_z(D: int, N: int, L: int, k: int):
    """At H=0, every U_μ = Γ. Then U_μ U_ν U_μ† U_ν† = I, so Tr/N = 1 and
    plaquette_{μν} = Re(z_μν). Mean plaquette = mean over ordered pairs."""
    Gamma = build_clock_matrix(N)
    z = build_twist(D, N, L, k=k)
    H_list = init_H_list_zero(D, N)
    plaq = float(plaquette_average(H_list, Gamma, z, D))

    z_re = [float(jnp.real(z[mu, nu])) for mu in range(D) for nu in range(mu + 1, D)]
    expected = sum(z_re) / len(z_re)
    assert abs(plaq - expected) < 1e-12, f"D={D} N={N}: plaq={plaq}, expected={expected}"


# ═══════════════════════════════════════════════════════════════
# Unit: plaquette observable ↔ mean consistency
# ═══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_wilson_loop_plaquette_matches_mean():
    """plaquette_average = mean over pairs of wilson_loop_plaquette."""
    D, N, L = 3, 25, 5
    Gamma = build_clock_matrix(N)
    z = build_twist(D, N, L, k=1)
    key = random.PRNGKey(7)
    H_list = init_H_list_random(D, N, key, scale=0.2)
    U = build_link_matrices(H_list, Gamma)

    pairs = [(mu, nu) for mu in range(D) for nu in range(mu + 1, D)]
    per_pair = [float(wilson_loop_plaquette(U, mu, nu, z)) for mu, nu in pairs]
    mean_per_pair = sum(per_pair) / len(per_pair)
    mean_via_loss = float(plaquette_average(H_list, Gamma, z, D))
    assert abs(mean_per_pair - mean_via_loss) < 1e-12


@pytest.mark.unit
def test_wilson_loop_plaquette_same_index_raises():
    Gamma = build_clock_matrix(9)
    z = build_twist(D=2, N=9, L=3, k=1)
    H_list = init_H_list_zero(D=2, N=9)
    U = build_link_matrices(H_list, Gamma)
    with pytest.raises(ValueError):
        wilson_loop_plaquette(U, 0, 0, z)


# ═══════════════════════════════════════════════════════════════
# Unit: center-symmetry measurement at H=0
# ═══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_center_symmetry_at_H0_exact():
    """At H=0, Tr(Γ)/N = sum of N-th roots of unity / N = 0 exactly.
    So center_symmetry_order = 0."""
    D, N, L = 2, 49, 7
    Gamma = build_clock_matrix(N)
    H_list = init_H_list_zero(D, N)
    U = build_link_matrices(H_list, Gamma)
    val = float(center_symmetry_order(U))
    assert val < 1e-20, f"center_symmetry_order at H=0: {val:.3e}"


@pytest.mark.unit
def test_polyakov_loop_at_H0_is_zero():
    N = 49
    Gamma = build_clock_matrix(N)
    H_list = init_H_list_zero(D=2, N=N)
    U = build_link_matrices(H_list, Gamma)
    p1 = polyakov_loop(U, mu=0)
    assert abs(complex(p1)) < 1e-12


# ═══════════════════════════════════════════════════════════════
# Unit: wilson_loop_rectangular is gated (R2)
# ═══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_rectangular_wilson_loop_raises():
    """R2 is gated: rectangular loops require the twist phase formula from
    arXiv:1708.00841 or PRD 27 (1983) eq. (3.5). Must raise until transcribed."""
    Gamma = build_clock_matrix(9)
    z = build_twist(D=2, N=9, L=3, k=1)
    H_list = init_H_list_zero(D=2, N=9)
    U = build_link_matrices(H_list, Gamma)
    with pytest.raises(NotImplementedError, match="twist phase"):
        wilson_loop_rectangular(U, z, R=2, T=1)


# ═══════════════════════════════════════════════════════════════
# Integration: Phase A gate — Gross-Witten matches exact to 1e-6
# ═══════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.parametrize("t", [0.3, 0.5, 0.8, 1.0, 1.2, 2.0, 5.0])
def test_phase_a_gw_wilson_loops(t: float):
    res = solve_gw(t, validate=True)
    if t < 1.0:
        # Weak phase: a = √t
        assert abs(res.a - math.sqrt(t)) < 1e-6
        assert abs(res.w1 - (1.0 - t / 2.0)) < 1e-6
        assert abs(res.w2 - (1.0 - t) ** 2) < 1e-6
    else:
        # Strong phase
        assert abs(res.w1 - 1.0 / (2.0 * t)) < 1e-6
        assert abs(res.w2) < 1e-6


# ═══════════════════════════════════════════════════════════════
# Integration: TEK gradient descent reduces loss
# ═══════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_optimizer_decreases_loss():
    """A few Adam steps on D=2 N=9 strictly decrease the loss."""
    # lazy-import to avoid JIT cost at collection time
    from optimize import optimize_tek

    res = optimize_tek(
        D=2, N=9, lam=5.0, n_steps=60, lr=0.05, twist=True, log_every=10, verbose=False
    )
    # Loss should decrease from initial
    loss_hist = res.history["loss"]
    assert loss_hist[-1] < loss_hist[0] - 0.05, (
        f"Loss did not decrease enough: start={loss_hist[0]:.4f}, end={loss_hist[-1]:.4f}"
    )


@pytest.mark.integration
def test_optimizer_preserves_link_unitarity():
    """After optimization, U_μ remain unitary to machine precision."""
    from optimize import optimize_tek

    res = optimize_tek(
        D=2, N=9, lam=5.0, n_steps=40, lr=0.05, twist=True, log_every=50, verbose=False
    )
    Gamma = build_clock_matrix(9)
    U = build_link_matrices(res.H_list, Gamma)
    eye = jnp.eye(9, dtype=jnp.complex128)
    for mu in range(2):
        err = float(jnp.linalg.norm(U[mu] @ jnp.conj(U[mu].T) - eye))
        assert err < 1e-10, f"U_{mu+1} not unitary after opt: err={err:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
