import os

import jax.numpy as jnp
import pytest

from matrix_master_field.ansatz import MultiMonomialAnsatz
from matrix_master_field.bootstrap_sdp import HAS_CVXPY
from matrix_master_field.fock_jax import FockOps, word_moment
from matrix_master_field.sparse_fock import SparseMonomialField
from matrix_master_field.train import solve_two_matrix, solve_two_matrix_sparse


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
def test_gate_rejects_low_max_word_len_truncation_artifact():
    # A degree-3 ansatz solves the max_word_len=2 loop equations to MACHINE ZERO,
    # but 2 is too few equations to pin the moment: at g=1 it lands tr M0²≈0.80,
    # which the loose L=6 island [0.62,1.0] admits yet the tight L=8 island
    # [0.63,0.73] rejects. So a low residual is NOT sufficient — checked at L=8 the
    # fail-closed gate correctly flags this truncation artifact as NOT validated.
    ops = FockOps(2, 6)
    ans = MultiMonomialAnsatz(ops, degree=3)
    r = solve_two_matrix(ans, ops, 1.0, max_word_len=2, g_schedule=[0.5, 1.0],
                         steps=2000, sdp_word_len=8)
    assert r["sd_loss"] < 1e-6                        # solves its (few) max_word_len=2 eqs exactly
    O = [jnp.asarray(o) for o in r["operators"]]
    assert float(word_moment(O, (0, 0))) > 0.75       # artifact sits high (~0.80)
    assert r["validation"]["in_island"] is False      # ...outside the tight L=8 island
    assert r["validated"] is False                     # fail-closed catches the artifact


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy needed for the SDP-island gate")
@pytest.mark.skipif(not os.environ.get("MMF_SLOW"),
                    reason="slow: dim-1023 max_word_len=3 solve (~7 min); set MMF_SLOW=1")
def test_max_word_len3_solve_validated_in_tight_island():
    # The genuine g>0 result: at max_word_len=3 (cutoff ⌊6/2⌋·3=9, dim 1023) the
    # degree-3 operator solve lands tr M0²≈0.69 at g=1, INSIDE the tight L=8 island
    # [0.63,0.73] — operator and bootstrap agree to ~1%. The fail-closed gate passes.
    ops = FockOps(2, 9)
    ans = MultiMonomialAnsatz(ops, degree=3)
    r = solve_two_matrix(ans, ops, 1.0, max_word_len=3, g_schedule=[0.5, 1.0],
                         steps=1500, sdp_word_len=8)
    assert r["sd_loss"] < 1e-8                         # exact solution of the loop equations
    assert r["validation"]["in_island"] is True
    assert r["validated"] is True
    O = [jnp.asarray(o) for o in r["operators"]]
    assert 0.63 < float(word_moment(O, (0, 0))) < 0.73


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy needed for the SDP-island gate")
def test_sparse_solve_g0_validated():
    # The sparse-Fock solver reproduces the exact g=0 free field and passes the gate.
    field = SparseMonomialField(2, cutoff=6, degree=3)
    r = solve_two_matrix_sparse(field, 0.0, max_word_len=2, g_schedule=[0.0],
                                steps=800, sdp_word_len=6)
    assert r["validated"] is True
    assert r["sd_loss"] < 1e-9
    assert abs(float(field.word_moment(r["params"], (0, 0))) - 1.0) < 1e-6


def test_sparse_solve_truncation_guard():
    # Same degree-aware cutoff guard as the dense path: degree-3 needs cutoff
    # ⌊(2+3)/2⌋·3 = 6, so cutoff 5 must be rejected.
    field = SparseMonomialField(2, cutoff=5, degree=3)
    with pytest.raises(ValueError):
        solve_two_matrix_sparse(field, 0.5, max_word_len=2, validate=False)


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy needed for the SDP-island gate")
def test_sparse_solve_accepts_warm_start():
    # init_params warm-start (the max_word_len-homotopy): seeding with the free
    # field params reproduces the default g=0 result and passes the gate.
    field = SparseMonomialField(2, cutoff=6, degree=3)
    r = solve_two_matrix_sparse(field, 0.0, max_word_len=2, g_schedule=[0.0],
                                steps=400, sdp_word_len=6,
                                init_params=field.params_for_free_field())
    assert r["validated"] is True
    assert abs(float(field.word_moment(r["params"], (0, 0))) - 1.0) < 1e-6


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
