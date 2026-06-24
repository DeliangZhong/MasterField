import numpy as np

from matrix_master_field import one_matrix as om
from matrix_master_field.ansatz import MonomialAnsatz
from matrix_master_field.fock_jax import FockOps
from matrix_master_field.train import solve

# The engine optimizes a Hermitian operator from random init against the exact
# nonlinear loop equations (positivity automatic) and must recover the *known*
# one-matrix master field. We validate INTERIOR moments only: at finite cutoff K
# the loop equations leave the top few (truncation-edge) moments unconstrained,
# so they are free by construction and excluded from the check.


def test_engine_recovers_gaussian_by_optimization():
    ops = FockOps(1, 14)
    ans = MonomialAnsatz(ops, degree=3)
    res = solve(ans, [0.0, 1.0], ops, K=10, n_restarts=4, steps=3000, seed=0)
    target = om.gaussian_moments(10)
    err = np.max(np.abs(res["moments"][:7] - target[:7]))  # interior m_0..m_6
    assert err < 1e-3, f"interior maxerr {err:.2e}, sd_loss {res['sd_loss']:.2e}"


def test_engine_recovers_quartic_by_optimization():
    ops = FockOps(1, 16)
    ans = MonomialAnsatz(ops, degree=3)
    res = solve(ans, [0.0, 1.0, 0.0, 0.5], ops, K=12, n_restarts=4, steps=3000, seed=0)
    target = om.quartic_moments_from_sd(0.5, max_power=12)
    err = np.max(np.abs(res["moments"][:9] - target[:9]))  # interior m_0..m_8
    assert err < 1e-3, f"interior maxerr {err:.2e}, sd_loss {res['sd_loss']:.2e}"
