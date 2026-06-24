import numpy as np

from matrix_master_field import one_matrix as om
from matrix_master_field.amortized import AmortizedMonomial, train_amortized
from matrix_master_field.fock_jax import FockOps


def _quartic_vprime(g):
    return [0.0, 1.0, 0.0, g]


def test_amortized_learns_quartic_family():
    # One network M̂(g) trained unsupervised across 5 couplings. Accuracy scales
    # with capacity (probe: hidden=32 -> ~1.9e-2; hidden=96 -> ~1.7e-3 interior),
    # competitive with the dedicated per-coupling solve at sufficient width.
    from matrix_master_field.fock_jax import power_moments

    ops = FockOps(1, 16)
    model = AmortizedMonomial(ops, degree=3, hidden=96)
    train_g = [0.1, 0.3, 0.5, 0.7, 0.9]
    params, loss = train_amortized(model, _quartic_vprime, train_g, K=12, steps=16000, seed=0)

    # in-training coupling: one network reproduces the master field
    M = model.build_operators(params, 0.5)[0]
    Mn = np.asarray(M)
    assert np.allclose(Mn, Mn.T, atol=1e-12)  # Hermitian
    m_in = np.asarray(power_moments(M, 12))
    t_in = om.quartic_moments_from_sd(0.5, 12)
    err_in = np.max(np.abs(m_in[:9] - t_in[:9]))  # interior m_0..m_8
    assert err_in < 5e-3, f"in-training interior err {err_in:.2e}, loss {loss:.2e}"

    # held-out coupling (interpolated g=0.6): generalization
    m_out = np.asarray(power_moments(model.build_operators(params, 0.6)[0], 12))
    t_out = om.quartic_moments_from_sd(0.6, 12)
    err_out = np.max(np.abs(m_out[:9] - t_out[:9]))
    assert err_out < 5e-3, f"held-out interior err {err_out:.2e}"
