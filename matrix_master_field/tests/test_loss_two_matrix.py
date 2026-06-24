import jax
import numpy as np

from matrix_master_field.ansatz import MultiMonomialAnsatz
from matrix_master_field.fock_jax import FockOps
from matrix_master_field.loss import two_matrix_sd_residual, two_matrix_test_words


def test_g0_free_field_satisfies_two_matrix_sd():
    # At g=0 the two matrices decouple into free semicirculars, which satisfy
    # the (commutator-free) Gaussian SD equations exactly.
    ops = FockOps(2, 6)
    ans = MultiMonomialAnsatz(ops, degree=2)
    M = ans.build_operators(ans.params_for_free_field())
    r = float(two_matrix_sd_residual(M, two_matrix_test_words(3), g=0.0))
    assert r < 1e-9, f"g=0 free-field residual {r:.2e}"


def test_residual_detects_perturbation():
    ops = FockOps(2, 6)
    ans = MultiMonomialAnsatz(ops, degree=2)
    p = np.array(ans.params_for_free_field())  # writable copy
    p[0, ans._free_idx[0]] = 2.4  # rescale M_0 -> breaks tr M_0^2 = 1
    M = ans.build_operators(jax.numpy.asarray(p))
    r = float(two_matrix_sd_residual(M, two_matrix_test_words(3), g=0.0))
    assert r > 1e-3, f"perturbed residual {r:.2e}"
