import jax
import numpy as np

from matrix_master_field.ansatz import MultiMonomialAnsatz
from matrix_master_field.fock_jax import FockOps
from matrix_master_field.loss import (
    cyclicity_loss,
    exchange_loss,
    two_matrix_test_words,
    z2_loss,
)


def test_free_field_satisfies_symmetries():
    ops = FockOps(2, 6)
    ans = MultiMonomialAnsatz(ops, degree=2)
    M = ans.build_operators(ans.params_for_free_field())
    words = two_matrix_test_words(4)
    # free semicircular vacuum is tracial, M1<->M2 symmetric, and Z2-even
    assert float(cyclicity_loss(M, words)) < 1e-9
    assert float(exchange_loss(M, words)) < 1e-9
    assert float(z2_loss(M, words)) < 1e-9


def test_cyclicity_detects_non_tracial_state():
    # A generic Hermitian-operator pair gives a NON-tracial vacuum state; the
    # loss must catch it (the spurious-state guard from the audit). Use random
    # O(1) symmetric operators, which are strongly non-tracial.
    ops = FockOps(2, 4)
    rng = np.random.default_rng(0)
    D = ops.D
    M0 = rng.normal(size=(D, D))
    M1 = rng.normal(size=(D, D))
    M = [jax.numpy.asarray(M0 + M0.T), jax.numpy.asarray(M1 + M1.T)]
    words = two_matrix_test_words(3)
    assert float(cyclicity_loss(M, words)) > 1e-3


def test_exchange_detects_asymmetric_state():
    ops = FockOps(2, 6)
    ans = MultiMonomialAnsatz(ops, degree=2)
    p = np.array(ans.params_for_free_field())
    p[0, ans._free_idx[0]] = 2.6  # scale M_0 only -> breaks M1<->M2
    M = ans.build_operators(jax.numpy.asarray(p))
    words = two_matrix_test_words(3)
    assert float(exchange_loss(M, words)) > 1e-3
