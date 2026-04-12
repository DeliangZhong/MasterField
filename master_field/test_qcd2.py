"""Phase 0 validation tests for the QCD₂ master field infrastructure.

These tests are smoke tests, not high-precision verifications. They document the
observed behavior of the single-α Gaussian ansatz on a truncated Cuntz-Fock space.
"""

from __future__ import annotations

import math

import pytest

from cuntz_fock import CuntzFockSpace
from lattice import (
    abs_area_2d,
    cyclic_canonical,
    enumerate_nonself_intersecting_2d,
    lattice_symmetry_orbit,
    reduce_backtracks,
    signed_area_2d,
)
from qcd2 import solve_alpha_for_plaquette

# ═══════════════════════════════════════════════════════════
# Lattice encoding tests
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
def test_reduce_backtracks_basic():
    assert reduce_backtracks((1, -1, 2, 3, -3, 2)) == (2, 2)
    assert reduce_backtracks((1, -1)) == ()
    assert reduce_backtracks(()) == ()


@pytest.mark.unit
def test_reduce_backtracks_cyclic_wrap():
    # Cyclic wrap: last element cancels with first
    assert reduce_backtracks((2, 1, 2, -2, -1, -2)) == ()


@pytest.mark.unit
def test_reduce_backtracks_preserves_plaquette():
    # Plaquette has no adjacent cancellations
    assert reduce_backtracks((1, 2, -1, -2)) == (1, 2, -1, -2)


@pytest.mark.unit
def test_cyclic_canonical_stable():
    w = (1, 2, -1, -2)
    w_rot = (-2, 1, 2, -1)
    # Both rotations give same canonical form
    assert cyclic_canonical(w) == cyclic_canonical(w_rot)


@pytest.mark.unit
def test_signed_area_plaquette():
    # Counterclockwise plaquette has area +1
    assert signed_area_2d((1, 2, -1, -2)) == 1
    # Clockwise has area -1
    assert signed_area_2d((2, 1, -2, -1)) == -1


@pytest.mark.unit
def test_abs_area_rectangle():
    assert abs_area_2d((1, 1, 2, -1, -1, -2)) == 2  # 2x1
    assert abs_area_2d((1, 1, 2, 2, -1, -1, -2, -2)) == 4  # 2x2


@pytest.mark.unit
def test_lattice_symmetry_orbit_plaquette():
    """Plaquette orbit under B_2 (2D hyperoctahedral) has 2 distinct canonical reps
    (clockwise and counterclockwise)."""
    orbit = lattice_symmetry_orbit((1, 2, -1, -2), D=2)
    assert len(orbit) == 2  # ±1 orientation
    # All orbit members should be actual plaquettes (area ±1)
    for w in orbit:
        assert abs_area_2d(w) == 1


@pytest.mark.unit
def test_enumerate_2d_loops_includes_plaquette():
    loops = enumerate_nonself_intersecting_2d(4)
    # Two orientations of plaquette
    assert len(loops) == 2
    for w in loops:
        assert abs_area_2d(w) == 1


# ═══════════════════════════════════════════════════════════
# Cuntz-Fock unitary operator tests
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
def test_build_unitary_gaussian_is_unitary():
    fock = CuntzFockSpace(n_matrices=2, max_length=4)
    U = fock.build_unitary_gaussian(0.5, matrix_idx=0)
    is_u, err = fock.check_unitarity(U, tol=1e-10)
    assert is_u, f"Unitarity violation: {err}"


@pytest.mark.unit
def test_wilson_loop_vev_empty_word():
    fock = CuntzFockSpace(n_matrices=2, max_length=4)
    ops = {1: fock.build_unitary_gaussian(0.3, 0), 2: fock.build_unitary_gaussian(0.3, 1)}
    assert abs(fock.wilson_loop_vev(ops, ()) - 1.0) < 1e-12


@pytest.mark.unit
def test_wilson_loop_vev_unit_loop():
    """<Ω| U U† |Ω> = <Ω|I|Ω> = 1 for unitary U (exact in truncated space too
    since expm gives exact unitary)."""
    fock = CuntzFockSpace(n_matrices=2, max_length=4)
    ops = {1: fock.build_unitary_gaussian(0.3, 0), 2: fock.build_unitary_gaussian(0.3, 1)}
    val = fock.wilson_loop_vev(ops, (1, -1))
    assert abs(val - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════
# QCD₂ plaquette matching (Phase 0 core acceptance)
# ═══════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.parametrize("lam", [0.5, 1.0, 2.0])
def test_plaquette_matches_area_law_exactly(lam: float):
    """After solving for α, W[□] matches exp(-λ/2) to machine precision."""
    fock = CuntzFockSpace(n_matrices=2, max_length=4)
    alpha = solve_alpha_for_plaquette(lam, fock)
    ops = {
        1: fock.build_unitary_gaussian(alpha, 0),
        2: fock.build_unitary_gaussian(alpha, 1),
    }
    w_plaq = fock.wilson_loop_vev(ops, (1, 2, -1, -2)).real
    assert abs(w_plaq - math.exp(-lam / 2)) < 1e-10


@pytest.mark.integration
def test_alpha_converges_with_L():
    """α(λ, L) converges rapidly as L grows."""
    lam = 1.0
    alphas = []
    for L in [4, 6, 8]:
        fock = CuntzFockSpace(n_matrices=2, max_length=L)
        alphas.append(solve_alpha_for_plaquette(lam, fock))
    # Differences should shrink as L grows
    diff_4_6 = abs(alphas[1] - alphas[0])
    diff_6_8 = abs(alphas[2] - alphas[1])
    assert diff_6_8 < diff_4_6, f"α not converging: {alphas}"


@pytest.mark.integration
def test_single_alpha_ansatz_misses_area_law_for_2x1():
    """KEY PHASE 0 FINDING: single-α Gaussian ansatz matches plaquette by construction
    but fails the 2D factorization W[2×1] = W[□]². Error is ~5e-3 at λ=1 and
    L-independent — a genuine physical limitation of the simplest ansatz that
    motivates the ML approach in Phase 1+.
    """
    lam = 1.0
    fock = CuntzFockSpace(n_matrices=2, max_length=6)
    alpha = solve_alpha_for_plaquette(lam, fock)
    ops = {
        1: fock.build_unitary_gaussian(alpha, 0),
        2: fock.build_unitary_gaussian(alpha, 1),
    }
    w_2x1 = fock.wilson_loop_vev(ops, (1, 1, 2, -1, -1, -2)).real
    target = math.exp(-lam)
    err = abs(w_2x1 - target)
    # Document: error is order 5e-3 at L=6, not 1e-4
    assert 1e-3 < err < 1e-2, f"Unexpected 2×1 error at L=6, λ=1: {err}"


@pytest.mark.integration
def test_exchange_symmetry_2x1_vs_1x2():
    """Exchange of matrix indices (reflection symmetry): W[2×1] = W[1×2]."""
    lam = 1.0
    fock = CuntzFockSpace(n_matrices=2, max_length=6)
    alpha = solve_alpha_for_plaquette(lam, fock)
    ops = {
        1: fock.build_unitary_gaussian(alpha, 0),
        2: fock.build_unitary_gaussian(alpha, 1),
    }
    w_2x1 = fock.wilson_loop_vev(ops, (1, 1, 2, -1, -1, -2)).real
    w_1x2 = fock.wilson_loop_vev(ops, (1, 2, 2, -1, -2, -2)).real
    assert abs(w_2x1 - w_1x2) < 1e-10


@pytest.mark.integration
def test_backtrack_invariance():
    """Inserting a backtrack pair into a loop leaves W invariant at N=∞."""
    lam = 1.0
    fock = CuntzFockSpace(n_matrices=2, max_length=6)
    alpha = solve_alpha_for_plaquette(lam, fock)
    ops = {
        1: fock.build_unitary_gaussian(alpha, 0),
        2: fock.build_unitary_gaussian(alpha, 1),
    }
    w_plain = fock.wilson_loop_vev(ops, (1, 2, -1, -2)).real
    # Insert (1, -1) backtrack
    w_backtrack = fock.wilson_loop_vev(ops, (1, 2, -1, 1, -1, -2)).real
    # Small but nonzero difference from truncation
    assert abs(w_plain - w_backtrack) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
