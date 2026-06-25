import pytest

from matrix_master_field import bootstrap_sdp as bs

pytestmark = pytest.mark.skipif(not bs.HAS_CVXPY, reason="cvxpy not installed")


def test_two_matrix_sdp_brackets_free_at_g0():
    # At g=0 the loop equations pin tr M0^2 = 1 (free semicircular); the SDP
    # island must be tight there. This validates the relaxation (moment matrix
    # + product matrix + commutator loop equations + symmetries).
    lo = bs.bootstrap_two_matrix(0.0, max_word_len=4, target_word=(0, 0), maximize=False)
    hi = bs.bootstrap_two_matrix(0.0, max_word_len=4, target_word=(0, 0), maximize=True)
    assert lo is not None and hi is not None
    assert abs(lo - 1.0) < 1e-3, f"g=0 lower bound {lo} != 1"
    assert abs(hi - 1.0) < 1e-3, f"g=0 upper bound {hi} != 1"


def test_two_matrix_sdp_parity_forbidden_is_zero():
    # tr M0 M1 has odd counts of both generators -> Z2xZ2 forbids it -> exactly 0.
    assert bs.bootstrap_two_matrix(1.0, max_word_len=4, target_word=(0, 1)) == 0.0


def test_two_matrix_sdp_is_valid_outer_bound_at_g_positive():
    # Outer relaxation: lower <= upper, and the confining interaction keeps the
    # upper bound on tr M0^2 at or below the free value 1.
    lo = bs.bootstrap_two_matrix(1.0, max_word_len=4, target_word=(0, 0), maximize=False)
    hi = bs.bootstrap_two_matrix(1.0, max_word_len=4, target_word=(0, 0), maximize=True)
    assert lo is not None and hi is not None
    assert lo <= hi + 1e-6
    assert hi <= 1.0 + 1e-3


def test_sdp_island_tightens_with_relaxation_order():
    # The loose L=6 island admits low-max_word_len operator artifacts; raising the
    # relaxation order tightens it dramatically. At g=1 the upper bound on tr M0^2
    # falls from ~1.0 (L=6) to <0.75 (L=8) — this is what exposes max_word_len=2
    # operator solves (~0.80) as truncation artifacts (see the M3 results doc).
    ub6 = bs.bootstrap_two_matrix(1.0, max_word_len=6, target_word=(0, 0), maximize=True)
    ub8 = bs.bootstrap_two_matrix(1.0, max_word_len=8, target_word=(0, 0), maximize=True)
    assert ub6 is not None and ub8 is not None
    assert ub6 > 0.99           # L=6 upper bound is loose (~1.0)
    assert ub8 < 0.75           # L=8 upper bound is tight (~0.727)
    assert ub8 < ub6 - 0.2      # dramatic tightening with relaxation order
