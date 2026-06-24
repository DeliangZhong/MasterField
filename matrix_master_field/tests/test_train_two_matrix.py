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
