import jax
import numpy as np

from matrix_master_field.ansatz import MultiDenseHermitianAnsatz, MultiMonomialAnsatz
from matrix_master_field.fock_jax import FockOps, word_moment


def test_multi_monomial_operators_are_hermitian():
    ops = FockOps(n_matrices=2, max_length=4)
    ans = MultiMonomialAnsatz(ops, degree=2)
    params = ans.init_params(jax.random.PRNGKey(0))
    M = ans.build_operators(params)
    assert len(M) == 2
    for Mi in M:
        Mn = np.asarray(Mi)
        assert np.allclose(Mn, Mn.T, atol=1e-12)


def test_multi_monomial_free_field_reproduces_free_wick():
    ops = FockOps(n_matrices=2, max_length=4)
    ans = MultiMonomialAnsatz(ops, degree=2)
    M = ans.build_operators(ans.params_for_free_field())
    # M̂_i = â_i + â†_i (free semicirculars):
    #   τ(x1 x2 x1 x2) = 0 (alternating centered word),  τ(x1² x2²) = 1 (free factorization)
    assert np.isclose(float(word_moment(M, (0, 1, 0, 1))), 0.0, atol=1e-12)
    assert np.isclose(float(word_moment(M, (0, 0, 1, 1))), 1.0, atol=1e-12)
    # and each marginal is a unit semicircle: τ(x_i²)=1, τ(x_i⁴)=2
    assert np.isclose(float(word_moment(M, (0, 0))), 1.0, atol=1e-12)
    assert np.isclose(float(word_moment(M, (1, 1, 1, 1))), 2.0, atol=1e-12)


def test_multi_dense_operators_are_hermitian():
    ops = FockOps(n_matrices=2, max_length=4)
    ans = MultiDenseHermitianAnsatz(ops)
    M = ans.build_operators(ans.init_params(jax.random.PRNGKey(0)))
    assert len(M) == 2 and ans.n_params == 2 * ops.D * ops.D
    for Mi in M:
        Mn = np.asarray(Mi)
        assert np.allclose(Mn, Mn.T, atol=1e-12)


def test_multi_dense_free_field_reproduces_free_wick():
    # the maximal-flexibility ansatz, warm-started at the free field, must give
    # the same free-Wick moments as the monomial free field (same operators).
    ops = FockOps(n_matrices=2, max_length=4)
    ans = MultiDenseHermitianAnsatz(ops)
    M = ans.build_operators(ans.params_for_free_field())
    assert np.isclose(float(word_moment(M, (0, 1, 0, 1))), 0.0, atol=1e-12)
    assert np.isclose(float(word_moment(M, (0, 0, 1, 1))), 1.0, atol=1e-12)
    assert np.isclose(float(word_moment(M, (0, 0))), 1.0, atol=1e-12)
    assert np.isclose(float(word_moment(M, (1, 1, 1, 1))), 2.0, atol=1e-12)
