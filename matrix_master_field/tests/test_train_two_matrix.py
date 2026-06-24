import jax.numpy as jnp

from matrix_master_field.ansatz import MultiMonomialAnsatz
from matrix_master_field.fock_jax import FockOps, word_moment
from matrix_master_field.train import solve_two_matrix


def _comm_sq(ops):
    # tr[M1,M2]^2 = 2(tr M1M2M1M2 - tr M1^2 M2^2)
    a = float(word_moment(ops, (0, 1, 0, 1)))
    b = float(word_moment(ops, (0, 0, 1, 1)))
    return 2.0 * (a - b)


def test_g0_two_matrix_recovers_free_field():
    ops = FockOps(2, 6)
    ans = MultiMonomialAnsatz(ops, degree=2)
    r = solve_two_matrix(ans, ops, 0.0, max_word_len=3, g_schedule=[0.0], steps=1500)
    assert r["sd_loss"] < 1e-9
    assert r["sym_loss"] < 1e-9
    O = [jnp.asarray(o) for o in r["operators"]]
    assert abs(float(word_moment(O, (0, 0))) - 1.0) < 1e-6      # tr M1^2 = 1
    assert abs(_comm_sq(O) - (-2.0)) < 1e-6                     # tr[M1,M2]^2 = -2 (free)


def test_g_positive_confines():
    # The mass + commutator interaction is confining: at g>0 the matrices shrink
    # (<tr M1^2> below the free value 1) and non-commutativity is reduced from
    # the free value tr[M1,M2]^2 = -2 toward 0. Qualitative, robust checks.
    ops = FockOps(2, 5)
    ans = MultiMonomialAnsatz(ops, degree=2)
    r = solve_two_matrix(ans, ops, 1.0, max_word_len=3, g_schedule=[0.5, 1.0], steps=1200)
    O = [jnp.asarray(o) for o in r["operators"]]
    assert r["sym_loss"] < 1e-3                       # symmetries stay enforced
    assert 0.0 < float(word_moment(O, (0, 0))) < 1.0  # confinement: <tr M1^2> < 1
    assert -2.0 < _comm_sq(O) < 0.0                   # non-commutativity reduced but present
