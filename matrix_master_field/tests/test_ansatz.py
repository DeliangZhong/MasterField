import jax
import numpy as np

from matrix_master_field.ansatz import DenseHermitianAnsatz, MonomialAnsatz
from matrix_master_field.fock_jax import FockOps, power_moments


def test_monomial_operators_are_hermitian():
    ops = FockOps(1, 6)
    ans = MonomialAnsatz(ops, degree=3)
    params = ans.init_params(jax.random.PRNGKey(0))
    for M in ans.build_operators(params):
        Mn = np.asarray(M)
        assert np.allclose(Mn, Mn.T.conj(), atol=1e-12)


def test_degree1_can_represent_free_field():
    ops = FockOps(1, 8)
    ans = MonomialAnsatz(ops, degree=1)
    M = ans.build_operators(ans.params_for_free_field())[0]
    m = np.asarray(power_moments(M, 4))
    # M = â + â† => semicircle/Catalan: m_2 = 1, m_4 = 2
    assert np.isclose(m[2], 1.0) and np.isclose(m[4], 2.0)


def test_dense_hermitian_operators_are_hermitian():
    ops = FockOps(1, 6)
    ans = DenseHermitianAnsatz(ops)
    params = ans.init_params(jax.random.PRNGKey(0))
    M = np.asarray(ans.build_operators(params)[0])
    assert np.allclose(M, M.T, atol=1e-12)
    assert ans.n_params == ops.D * ops.D
