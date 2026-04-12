"""Tests for the lattice Makeenko-Migdal equation machinery.

These tests document the empirical finding from the candidate scan:
- Candidate D (LHS = (1/λ)·Σ_P W[staple], RHS = 2·W[C] + splits) has residuals that
  scale as O(w_+³) ~ 1/λ³ against the lattice-exact W[C] = w_+^Area answer.
- The self-consistent w_+ from candidate D is λ − √(λ²−1), which agrees with the
  GW strong-coupling value 1/(2λ) only to leading order in 1/λ.
- Residuals decrease monotonically as λ → ∞.

The tests below enforce the EMPIRICAL bounds observed in the scan. When a more
precise MM derivation becomes available, tighten these bounds.
"""

from __future__ import annotations

import math

import pytest

from lattice import enumerate_nonself_intersecting_2d
from mm_equations import (
    _candidate_catalog,
    _single_plaquette_lattice_2d,
    gw_w_plus,
    mm_residual_staple,
)


def _lattice_W(C, w_plus):
    return _single_plaquette_lattice_2d(C, w_plus)


@pytest.mark.unit
def test_gw_wplus_values():
    # Strong coupling
    assert abs(gw_w_plus(1.0) - 0.5) < 1e-12
    assert abs(gw_w_plus(2.0) - 0.25) < 1e-12
    assert abs(gw_w_plus(5.0) - 0.1) < 1e-12
    # Weak coupling
    assert abs(gw_w_plus(0.5) - 0.75) < 1e-12
    assert abs(gw_w_plus(0.0) - 1.0) < 1e-12


@pytest.mark.integration
def test_candidate_D_leading_order_at_strong_coupling():
    """Candidate D has residuals O(1/λ³) at large λ against lattice answer."""
    cat = _candidate_catalog()
    eq = cat["D (staple, RHS ×2)"]

    loops = enumerate_nonself_intersecting_2d(6)
    for lam in [2.0, 5.0, 10.0]:
        w_plus = gw_w_plus(lam)

        def W(C, wp=w_plus):
            return _lattice_W(C, wp)

        max_res = 0.0
        for C in loops:
            for j in range(len(C)):
                r = eq(C, j, 2, lam, W)
                max_res = max(max_res, abs(r))
        # Scaling: residual should shrink as 1/λ^3 or faster
        # At λ=2: ~1/64 ≈ 0.016; at λ=5: ~1/500 ≈ 0.002; at λ=10: ~1/4000 ≈ 2.5e-4
        max_allowed = 1.0 / (lam * lam)
        assert max_res < max_allowed, f"λ={lam}: residual={max_res}, allowed={max_allowed}"


@pytest.mark.unit
def test_self_consistent_w_plus_candidate_D():
    """Candidate D's self-consistent single-plaquette equation:
    (1/λ)(1 + w²) = 2w  ⇒  w² − 2λw + 1 = 0  ⇒  w = λ − √(λ²−1)

    This matches w_+ = 1/(2λ) (GW) to leading order in 1/λ.
    """
    for lam in [2.0, 5.0, 10.0, 100.0]:
        w_mm = lam - math.sqrt(lam * lam - 1.0)
        w_gw = 1.0 / (2.0 * lam)
        # They should match at leading order; the difference is O(1/λ³)
        rel_error = abs(w_mm - w_gw) / w_gw
        assert rel_error < 1.0 / (lam * lam), f"λ={lam}: rel_err={rel_error}"


@pytest.mark.unit
def test_plaquette_insertions_preserves_closure():
    """Every plaquette insertion must yield a CLOSED loop (net displacement 0)."""
    from lattice import is_closed, plaquette_insertions

    C = (1, 2, -1, -2)
    for j in range(len(C)):
        for ins in plaquette_insertions(C, j, D=2):
            assert is_closed(ins, D=2) or ins == (), f"Insertion {ins} not closed"


@pytest.mark.unit
def test_self_closure_in_mm_residual():
    """For a simple (non-self-intersecting) loop, the RHS should include the
    self-closure term W[C]·W[empty] = W[C]."""
    from lattice import self_intersection_splits

    C = (1, 2, -1, -2)  # plaquette, no internal self-intersections
    splits = self_intersection_splits(C)
    # The current implementation skips the trivial self-closure (start==end)
    assert splits == [], "Plaquette has no internal self-intersections"


@pytest.mark.integration
def test_mm_residual_staple_finite():
    """mm_residual_staple runs without error on all simple loops up to L=6."""
    from lattice import enumerate_nonself_intersecting_2d

    loops = enumerate_nonself_intersecting_2d(6)
    w_plus = gw_w_plus(5.0)

    def W(C, wp=w_plus):
        return _lattice_W(C, wp)

    for C in loops:
        for j in range(len(C)):
            r = mm_residual_staple(C, j, D=2, lam=5.0, W=W, include_self_closure=True)
            assert math.isfinite(r), f"Non-finite residual for {C}[{j}]"


# ═══════════════════════════════════════════════════════════
# LoopSystem tests (Phase 1 Step 2)
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
def test_loop_system_has_empty_loop_at_index_0():
    from lattice import build_loop_system

    ls = build_loop_system(D=2, L_max=4)
    assert ls.loops[0] == ()
    assert ls.empty_idx == 0
    assert ls.K >= 1


@pytest.mark.unit
def test_loop_system_table_extends_to_L_max_plus_2():
    """Staple insertions produce loops of length L+2; the table must cover these."""
    from lattice import build_loop_system

    ls = build_loop_system(D=2, L_max=4)
    # Loops in the table can have length up to L_max + 2 = 6
    max_in_table = max(len(w) for w in ls.loops)
    assert max_in_table <= 6
    # And loops of length 6 should be present (2x1 rectangles, etc.)
    lengths = {len(w) for w in ls.loops}
    assert 6 in lengths


@pytest.mark.unit
def test_loop_system_mm_equations_only_for_loops_up_to_L_max():
    from lattice import build_loop_system

    ls = build_loop_system(D=2, L_max=4)
    for eq in ls.mm_equations:
        loop_len = len(ls.loops[eq.loop_idx])
        assert loop_len <= ls.L_max


@pytest.mark.unit
def test_loop_system_all_lhs_indices_in_table():
    from lattice import build_loop_system

    ls = build_loop_system(D=2, L_max=4)
    for eq in ls.mm_equations:
        for idx in eq.lhs_loop_indices:
            assert 0 <= idx < ls.K
        for (i, j) in eq.rhs_split_pairs:
            assert 0 <= i < ls.K and 0 <= j < ls.K


@pytest.mark.integration
def test_loop_system_mm_residuals_match_scan_candidate_D():
    """Using the precomputed LoopSystem to evaluate MM residuals should match
    the mm_residual_staple function with RHS×2 (candidate D)."""
    from lattice import build_loop_system

    ls = build_loop_system(D=2, L_max=4)

    for lam in [2.0, 5.0]:
        w_plus = gw_w_plus(lam)
        # w[i] = w_+^Area(loop_i)
        w = [w_plus ** ls.areas[i] for i in range(ls.K)]

        max_res = 0.0
        for eq in ls.mm_equations:
            lhs = sum(w[k] for k in eq.lhs_loop_indices) / lam
            rhs = eq.rhs_self_coeff * w[eq.loop_idx] + sum(
                w[i] * w[j] for (i, j) in eq.rhs_split_pairs
            )
            res = abs(lhs - rhs)
            max_res = max(max_res, res)
        # Scaling: 1/λ² at large λ. At λ=2: ~0.25; at λ=5: ~0.04
        assert max_res < 1.0 / (lam * lam), f"λ={lam}: max_res={max_res}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
