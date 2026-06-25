"""M5a — the QM bootstrap: fixed-E <x^2> island + certified energy lower bound."""
import jax

jax.config.update("jax_enable_x64", True)

import pytest  # noqa: E402

from matrix_master_field.bootstrap_sdp import (  # noqa: E402
    HAS_CVXPY,
    bootstrap_qm_anharmonic,
    has_trusted_solver,
    qm_anharmonic_feasibility,
)
from matrix_master_field.qm_fock import ground_state, moment  # noqa: E402

pytestmark = pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")


def test_feasibility_brackets_true_m2_at_exact_energy():
    g, K = 1.0, 6
    E0, omega = ground_state(80, g)
    m2_true = moment(omega, 2)
    m2_lo, m2_hi = qm_anharmonic_feasibility(g, K, E0)
    assert m2_lo is not None and m2_hi is not None
    assert m2_lo - 1e-4 <= m2_true <= m2_hi + 1e-4


def test_feasibility_infeasible_below_ground_energy():
    # E well below E0=1.39 has no positive-moment solution -> at least one edge None.
    m2_lo, m2_hi = qm_anharmonic_feasibility(1.0, 6, 0.5)
    assert m2_lo is None or m2_hi is None


def test_energy_lower_bound_brackets_and_tightens():
    g = 1.0
    E0 = ground_state(80, g)[0]  # exact-diag anchor (feasible at every K)
    E_lo_coarse = bootstrap_qm_anharmonic(g, 4, E0)
    E_lo_fine = bootstrap_qm_anharmonic(g, 6, E0)
    assert E_lo_coarse is not None and E_lo_fine is not None
    assert E_lo_coarse <= E0 + 1e-4  # valid lower bound
    assert E_lo_fine <= E0 + 1e-4
    assert E_lo_fine >= E_lo_coarse - 1e-4  # tightens (rises toward E0) with K
    assert E_lo_fine > 1.0  # non-trivial at K=6 (feasible(1.0) is False there)


@pytest.mark.skipif(not has_trusted_solver(), reason="certified edge needs CLARABEL/MOSEK")
def test_energy_lower_bound_certified():
    E0 = ground_state(80, 1.0)[0]
    E_lo, solver, status = bootstrap_qm_anharmonic(1.0, 6, E0, with_status=True)
    assert E_lo is not None
    assert solver in ("MOSEK", "CLARABEL") and status == "optimal"
