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
    build_link_matrices_full,
    build_twist,
    hermitianize,
    init_H_list_random,
    init_H_list_zero,
    init_M_list_random,
    init_M_list_zero,
    plaquette_average,
    plaquette_average_full,
    tek_loss,
    tek_loss_full,
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
@pytest.mark.parametrize("N,L", [(9, 3), (25, 5), (49, 7)])
def test_clock_matrix_eigenvalues_are_L_roots_L_degenerate(N: int, L: int):
    """Γ = P_L ⊗ I_L has eigenvalues = L-th roots of unity, each with
    multiplicity L. Total N = L² eigenvalues."""
    Gamma = build_clock_matrix(N)
    eigs = jnp.linalg.eigvals(Gamma)
    phases_sorted = sorted([float(jnp.angle(e)) for e in eigs])

    # Expected: L distinct L-th roots, each repeated L times.
    expected = sorted([
        float(jnp.angle(jnp.exp(2j * jnp.pi * k / L)))
        for k in range(L)
        for _ in range(L)
    ])
    assert len(phases_sorted) == N
    assert len(expected) == N
    for a, b in zip(phases_sorted, expected):
        assert abs(a - b) < 1e-10, f"phase mismatch: {a} vs {b}"


@pytest.mark.unit
@pytest.mark.parametrize("N,L", [(9, 3), (25, 5), (49, 7), (121, 11)])
def test_clock_matrix_L_periodic(N: int, L: int):
    """Γ^L = I (stronger than Γ^N = I: the TEK tensor-product form has
    period L, not N)."""
    Gamma = build_clock_matrix(N)
    Gamma_L = jnp.linalg.matrix_power(Gamma, L)
    eye = jnp.eye(N, dtype=jnp.complex128)
    err = float(jnp.linalg.norm(Gamma_L - eye))
    assert err < 1e-10, f"Γ^L ≠ I at N={N} L={L}: err={err:.3e}"


@pytest.mark.unit
def test_clock_matrix_rejects_non_perfect_square():
    with pytest.raises(ValueError, match="perfect square"):
        build_clock_matrix(7)


@pytest.mark.unit
@pytest.mark.parametrize("N,L", [(9, 3), (25, 5), (49, 7)])
def test_clock_matrix_traceless(N: int, L: int):
    """For L > 1, Tr(Γ) = L · Tr(P_L) = L · 0 = 0."""
    Gamma = build_clock_matrix(N)
    assert abs(complex(jnp.trace(Gamma))) < 1e-12


@pytest.mark.unit
@pytest.mark.parametrize("N,L", [(9, 3), (25, 5), (49, 7)])
def test_clock_matrix_matches_tek_classical_saddle_U1(N: int, L: int):
    """The TEK classical saddle uses U_1 = P_L ⊗ I_L. Our build_clock_matrix
    should return exactly this matrix (so that Ω Γ Ω† can reach U_2 of the
    saddle via a unitary rotation)."""
    Gamma = build_clock_matrix(N)
    jk = jnp.arange(L)
    P_L = jnp.diag(jnp.exp(2j * jnp.pi * jk / L)).astype(jnp.complex128)
    I_L = jnp.eye(L, dtype=jnp.complex128)
    expected = jnp.kron(P_L, I_L)
    err = float(jnp.linalg.norm(Gamma - expected))
    assert err < 1e-14, f"Γ does not match P_L ⊗ I_L: err={err:.3e}"


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
# Unit: wilson_loop_rectangular (R2 resolved, f(R,T) = R·T)
# ═══════════════════════════════════════════════════════════════


@pytest.mark.unit
def test_rectangular_reduces_to_plaquette_at_R_T_1():
    """W[1×1]_{μν} must equal wilson_loop_plaquette(μ, ν)."""
    D, N, L = 2, 49, 7
    Gamma = build_clock_matrix(N)
    z = build_twist(D, N, L, k=1)
    key = random.PRNGKey(13)
    H_list = init_H_list_random(D, N, key, scale=0.2)
    U = build_link_matrices(H_list, Gamma)

    a = float(wilson_loop_rectangular(U, z, R=1, T=1, mu=0, nu=1))
    b = float(wilson_loop_plaquette(U, 0, 1, z))
    assert abs(a - b) < 1e-12, f"W[1×1] ({a}) ≠ plaquette ({b})"


@pytest.mark.unit
@pytest.mark.parametrize("R,T", [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (2, 3), (3, 3)])
def test_rectangular_at_H0_equals_Re_z_to_RT(R: int, T: int):
    """At H=0 all U_μ = Γ (diagonal), so the bare trace part = I/N = 1 and
    W[R×T] = Re(z_μν^{R·T}). Directly validates the twist-phase formula."""
    D, N, L, k = 2, 49, 7, 1
    Gamma = build_clock_matrix(N)
    z = build_twist(D, N, L, k=k)
    H_list = init_H_list_zero(D, N)
    U = build_link_matrices(H_list, Gamma)

    computed = float(wilson_loop_rectangular(U, z, R, T, mu=0, nu=1))
    expected = float(jnp.real(z[0, 1] ** (R * T)))
    assert abs(computed - expected) < 1e-10, (
        f"W[{R}×{T}] at H=0: computed={computed}, expected Re(z^{R*T})={expected}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("N,L,k", [(9, 3, 1), (25, 5, 1), (49, 7, 1), (49, 7, 3)])
def test_rectangular_twist_phase_at_H0_various_twists(N: int, L: int, k: int):
    """Check W[2×3] = Re(z^6) at H=0 for several flux values."""
    D = 2
    Gamma = build_clock_matrix(N)
    z = build_twist(D, N, L, k=k)
    H_list = init_H_list_zero(D, N)
    U = build_link_matrices(H_list, Gamma)

    R, T = 2, 3
    computed = float(wilson_loop_rectangular(U, z, R, T))
    expected = float(jnp.real(z[0, 1] ** (R * T)))
    assert abs(computed - expected) < 1e-10


@pytest.mark.unit
def test_rectangular_invalid_args_raise():
    Gamma = build_clock_matrix(9)
    z = build_twist(D=2, N=9, L=3, k=1)
    H_list = init_H_list_zero(D=2, N=9)
    U = build_link_matrices(H_list, Gamma)
    with pytest.raises(ValueError, match="mu and nu must differ"):
        wilson_loop_rectangular(U, z, R=1, T=1, mu=0, nu=0)
    with pytest.raises(ValueError, match="positive"):
        wilson_loop_rectangular(U, z, R=0, T=1)
    with pytest.raises(ValueError, match="positive"):
        wilson_loop_rectangular(U, z, R=1, T=-1)


def _build_tek_classical_saddle(L: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Construct the TEK classical saddle for N = L² with symmetric twist k=1:
        P_L = L×L clock, Q_L = L×L shift
        U_1 = P_L ⊗ I_L,  U_2 = Q_L ⊗ P_L
        These satisfy  U_1 U_2 = ω_L^{-1} U_2 U_1  with  ω_L = exp(2πi/L).
    In our convention z_12 = ω_L, so U_1 U_2 = z_12^{-1} U_2 U_1 ⇒ classical TEK
    saddle (plaquette = 1 by construction).

    Returns (U_1, U_2, z_12).
    """
    N = L * L
    jk = jnp.arange(L)
    P = jnp.diag(jnp.exp(2j * jnp.pi * jk / L)).astype(jnp.complex128)
    Q = jnp.roll(jnp.eye(L, dtype=jnp.complex128), shift=-1, axis=0)
    I_L = jnp.eye(L, dtype=jnp.complex128)
    U1 = jnp.kron(P, I_L)  # (N, N)
    U2 = jnp.kron(Q, P)    # (N, N)
    z12 = jnp.exp(2j * jnp.pi / L)
    assert U1.shape == (N, N) and U2.shape == (N, N)
    return U1, U2, z12


@pytest.mark.unit
@pytest.mark.parametrize("L", [3, 5, 7])
def test_tek_classical_saddle_heisenberg_relation(L: int):
    """Verify U_1 U_2 = z_12^{-1} U_2 U_1 exactly at the TEK classical saddle."""
    U1, U2, z12 = _build_tek_classical_saddle(L)
    lhs = U1 @ U2
    rhs = jnp.conj(z12) * (U2 @ U1)  # z^{-1} = conj(z) on unit circle
    err = float(jnp.linalg.norm(lhs - rhs))
    assert err < 1e-12, f"Heisenberg relation violated at L={L}: err={err:.3e}"


@pytest.mark.unit
@pytest.mark.parametrize("L,R,T", [(3, 1, 1), (3, 2, 1), (3, 1, 3), (3, 2, 3),
                                    (5, 1, 1), (5, 2, 2), (5, 3, 2), (5, 4, 3),
                                    (7, 2, 2), (7, 3, 3)])
def test_rectangular_at_classical_saddle_equals_one(L: int, R: int, T: int):
    """KEY VERIFICATION: at the TEK classical saddle, W[R×T] = 1 exactly,
    by construction of the twist phase z_12^{R·T}. This independently
    confirms the formula from arXiv:1708.00841 eq. (2.4)."""
    N = L * L
    U1, U2, z12 = _build_tek_classical_saddle(L)
    z = jnp.ones((2, 2), dtype=jnp.complex128)
    z = z.at[0, 1].set(z12)
    z = z.at[1, 0].set(jnp.conj(z12))
    U = [U1, U2]

    w = float(wilson_loop_rectangular(U, z, R, T, mu=0, nu=1))
    assert abs(w - 1.0) < 1e-10, (
        f"W[{R}×{T}] at TEK classical saddle (L={L}, N={N}): {w}, expected 1"
    )


@pytest.mark.unit
def test_rectangular_symmetric_under_axis_swap_untwisted():
    """Without twist (z=1), the Wilson loop of an R×T rectangle in (μ,ν) plane
    should equal the loop of a T×R rectangle in (ν,μ) plane (geometric
    rotation)."""
    D, N, L = 2, 49, 7
    Gamma = build_clock_matrix(N)
    z = jnp.ones((D, D), dtype=jnp.complex128)  # untwisted
    key = random.PRNGKey(5)
    H_list = init_H_list_random(D, N, key, scale=0.1)
    U = build_link_matrices(H_list, Gamma)

    R, T = 2, 3
    a = float(wilson_loop_rectangular(U, z, R, T, mu=0, nu=1))
    # For the same rectangular shape viewed from the other side, R and T swap
    # and (mu, nu) swap; the loop traces out the same SHAPE but with opposite
    # orientation. In pure Tr(U_μ^R U_ν^T U_μ^{-R} U_ν^{-T}) without twist,
    # this is the adjoint of the original — and Tr(A) + Tr(A†) = 2 Re Tr(A),
    # so the real parts match.
    b = float(wilson_loop_rectangular(U, z, T, R, mu=1, nu=0))
    assert abs(a - b) < 1e-10, f"untwisted 2×3 at (0,1) = {a}, 3×2 at (1,0) = {b}"


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
    U = build_link_matrices(res.params, Gamma)
    eye = jnp.eye(9, dtype=jnp.complex128)
    for mu in range(2):
        err = float(jnp.linalg.norm(U[mu] @ jnp.conj(U[mu].T) - eye))
        assert err < 1e-10, f"U_{mu+1} not unitary after opt: err={err:.3e}"


# ═══════════════════════════════════════════════════════════════
# R4: Full U(N) ansatz
# ═══════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.parametrize("D,N", [(2, 9), (3, 25), (4, 49)])
def test_full_ansatz_builds_unitary(D: int, N: int):
    """U_μ = expm(i M_μ) is unitary when M_μ is Hermitian."""
    key = random.PRNGKey(42)
    M_list = init_M_list_random(D, N, key, scale=0.3)
    U = build_link_matrices_full(M_list)
    assert len(U) == D
    eye = jnp.eye(N, dtype=jnp.complex128)
    for mu in range(D):
        err = float(jnp.linalg.norm(U[mu] @ jnp.conj(U[mu].T) - eye))
        assert err < 1e-10, f"full-ansatz U_{mu+1} not unitary: err={err:.3e}"


@pytest.mark.unit
@pytest.mark.parametrize("D,N,L,k", [(2, 9, 3, 1), (2, 49, 7, 1), (4, 49, 7, 1)])
def test_full_plaquette_at_M0_equals_mean_Re_z(D: int, N: int, L: int, k: int):
    """At M=0 every U_μ = I, so plaquette_{μν} = Re(z_μν); mean plaquette =
    mean Re(z_μν) over ordered pairs. Matches the orientation ansatz at H=0."""
    z = build_twist(D, N, L, k=k)
    M_list = init_M_list_zero(D, N)
    plaq = float(plaquette_average_full(M_list, z, D))

    z_re = [float(jnp.real(z[mu, nu])) for mu in range(D) for nu in range(mu + 1, D)]
    expected = sum(z_re) / len(z_re)
    assert abs(plaq - expected) < 1e-12


@pytest.mark.unit
def test_full_and_orientation_agree_at_identity_init():
    """At H=0 (orientation) and M=0 (full) the loss value should match,
    since both give all-I plaquette product."""
    D, N, L = 2, 49, 7
    Gamma = build_clock_matrix(N)
    z = build_twist(D, N, L, k=1)
    H_list = init_H_list_zero(D, N)
    M_list = init_M_list_zero(D, N)
    a = float(tek_loss(H_list, Gamma, z, D))
    b = float(tek_loss_full(M_list, z, D))
    assert abs(a - b) < 1e-12


@pytest.mark.integration
def test_full_ansatz_optimizer_decreases_loss():
    """Full ansatz: a few Adam steps strictly decrease the loss on D=2 N=9."""
    from optimize import optimize_tek

    res = optimize_tek(
        D=2, N=9, lam=5.0, n_steps=60, lr=0.05, twist=True,
        ansatz="full", log_every=10, verbose=False,
    )
    assert res.ansatz == "full"
    assert len(res.params) == 2  # D=2 matrices (not D-1)
    loss_hist = res.history["loss"]
    assert loss_hist[-1] < loss_hist[0] - 0.05, (
        f"Full-ansatz loss did not decrease: start={loss_hist[0]:.4f}, "
        f"end={loss_hist[-1]:.4f}"
    )


@pytest.mark.integration
def test_full_ansatz_preserves_unitarity():
    """After full-ansatz optimization, each U_μ remains unitary."""
    from optimize import optimize_tek

    res = optimize_tek(
        D=2, N=9, lam=5.0, n_steps=50, lr=0.05, twist=True,
        ansatz="full", log_every=50, verbose=False,
    )
    U = build_link_matrices_full(res.params)
    eye = jnp.eye(9, dtype=jnp.complex128)
    for mu in range(2):
        err = float(jnp.linalg.norm(U[mu] @ jnp.conj(U[mu].T) - eye))
        assert err < 1e-10, f"full-ansatz U_{mu+1} not unitary after opt: err={err:.3e}"


@pytest.mark.unit
def test_optimize_tek_rejects_unknown_ansatz():
    from optimize import optimize_tek

    with pytest.raises(ValueError, match="Unknown ansatz"):
        optimize_tek(D=2, N=9, lam=1.0, n_steps=10, ansatz="garbage", verbose=False)


@pytest.mark.unit
def test_optimize_tek_rejects_wrong_init_params_length():
    """If init_params has the wrong length for the chosen ansatz, raise."""
    from optimize import optimize_tek

    key = random.PRNGKey(0)
    # For D=2 orientation expects 1 matrix; pass 2 → should raise.
    wrong = init_M_list_random(2, 9, key)
    with pytest.raises(ValueError, match="expects"):
        optimize_tek(
            D=2, N=9, lam=1.0, n_steps=10, ansatz="orientation",
            init_params=wrong, verbose=False,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
