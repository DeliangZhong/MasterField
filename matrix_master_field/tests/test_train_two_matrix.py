import jax.numpy as jnp
import pytest

from matrix_master_field.ansatz import MultiMonomialAnsatz
from matrix_master_field.bootstrap_sdp import HAS_CVXPY
from matrix_master_field.fock_jax import FockOps, word_moment
from matrix_master_field.train import solve_two_matrix


def _comm_sq(ops):
    # tr[M0,M1]^2 = 2(tr M0M1M0M1 - tr M0^2 M1^2)
    a = float(word_moment(ops, (0, 1, 0, 1)))
    b = float(word_moment(ops, (0, 0, 1, 1)))
    return 2.0 * (a - b)


def test_truncation_guard_rejects_small_fock_cutoff():
    # SD residual evaluates commutator words of length |w|+3; cutoff 4 < 3+3=6
    # must be rejected (the old test silently used a contaminated cutoff).
    ops = FockOps(2, 4)
    ans = MultiMonomialAnsatz(ops, degree=2)
    with pytest.raises(ValueError):
        solve_two_matrix(ans, ops, 1.0, max_word_len=3, validate=False)


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy needed for the SDP-island gate")
def test_g0_two_matrix_validated():
    ops = FockOps(2, 6)
    ans = MultiMonomialAnsatz(ops, degree=2)
    r = solve_two_matrix(ans, ops, 0.0, max_word_len=3, g_schedule=[0.0],
                         steps=1500, sdp_word_len=4)
    assert r["sd_loss"] < 1e-9 and r["sym_loss"] < 1e-9
    O = [jnp.asarray(o) for o in r["operators"]]
    assert abs(float(word_moment(O, (0, 0))) - 1.0) < 1e-6   # tr M0^2 = 1 (free)
    assert abs(_comm_sq(O) - (-2.0)) < 1e-6                   # tr[M0,M1]^2 = -2
    assert r["validated"] is True                            # passes the fail-closed gate


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy needed for the SDP-island gate")
def test_g_positive_solve_flagged_invalid_by_gate():
    # The current g>0 solve is under-converged; the fail-closed gate must mark it
    # NOT validated (the spurious-solution guard the old qualitative test missed).
    ops = FockOps(2, 6)
    ans = MultiMonomialAnsatz(ops, degree=2)
    r = solve_two_matrix(ans, ops, 1.0, max_word_len=3, g_schedule=[0.5, 1.0],
                         steps=600, sdp_word_len=6)
    assert r["validated"] is False
    assert r["validation"]["sd_ok"] is False                 # residual not tight enough
    assert r["validation"]["island"][0] is not None          # SDP island was computed
    # confinement is still a true qualitative fact, just not a validated solution:
    O = [jnp.asarray(o) for o in r["operators"]]
    assert 0.0 < float(word_moment(O, (0, 0))) < 1.0


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy needed for the SDP-island gate")
def test_g_positive_solve_validated_at_degree3():
    # The fix for the degree-2 under-expressiveness: a degree-3 ansatz (cutoff at
    # the exactness bound floor((2+3)/2)*3 = 6) solves the truncated loop equations
    # to machine zero and lands the moment INSIDE the rigorous SDP island at g>0,
    # so the fail-closed gate now PASSES — the moment a degree-2 solve parked below
    # the lower bound. (Degree-2 floors sd_loss ~1e-3 at trM0^2~0.78 < lb here.)
    ops = FockOps(2, 6)
    ans = MultiMonomialAnsatz(ops, degree=3)
    r = solve_two_matrix(ans, ops, 0.5, max_word_len=2, g_schedule=[0.25, 0.5],
                         steps=2000, sdp_word_len=6)
    assert r["validated"] is True
    assert r["sd_loss"] < 1e-6                       # exact solution of the truncated loop eqs
    assert r["validation"]["in_island"] is True
    O = [jnp.asarray(o) for o in r["operators"]]
    assert float(word_moment(O, (0, 0))) > 0.80      # clearly above the degree-2 floor (~0.78)


def test_truncation_guard_is_degree_aware():
    # The guard scales the required cutoff with ansatz degree: a degree-3 letter
    # moves the Cuntz quanta count by +/-3, so the exact-moment cutoff is
    # floor((max_word_len+3)/2)*degree = floor(5/2)*3 = 6. Cutoff 5 must be
    # rejected even though 5 >= max_word_len+... (the OLD degree-blind guard,
    # need = max_word_len+3 = 5, would have wrongly accepted it).
    ops = FockOps(2, 5)
    ans = MultiMonomialAnsatz(ops, degree=3)
    with pytest.raises(ValueError):
        solve_two_matrix(ans, ops, 0.5, max_word_len=2, validate=False)
