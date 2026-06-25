"""M5b — the single-matrix-QM SDP: certified lower bound on E/N^2 that brackets the
exact (free-fermion) value and does NOT collapse to the single particle."""
import jax

jax.config.update("jax_enable_x64", True)

import pytest  # noqa: E402

from matrix_master_field.bootstrap_sdp import (  # noqa: E402
    HAS_CVXPY,
    bootstrap_single_matrix_qm,
    has_trusted_solver,
)
from matrix_master_field.qm_collective import collective_master_field  # noqa: E402

pytestmark = pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")

# single-particle E0(g) (M5a) — the value a wrong (collapsing) bootstrap would return
SINGLE_PARTICLE = {0.0: 1.0, 0.5: 1.24185, 1.0: 1.39235}


def test_sdp_lower_bound_brackets_and_no_collapse():
    for g in (0.0, 0.5, 1.0):
        e_exact = collective_master_field(g)["energy"]
        lb = bootstrap_single_matrix_qm(g, L=4, maximize=False)
        assert lb is not None
        assert lb <= e_exact + 1e-4          # valid lower bound on the MATRIX E/N^2
        if g > 0:
            assert lb < SINGLE_PARTICLE[g] - 1e-3   # NOT the single-particle collapse


def test_sdp_g0_exact():
    lb = bootstrap_single_matrix_qm(0.0, L=4, maximize=False)
    assert abs(lb - 1.0) < 1e-4              # g=0 matrix E/N^2 = 1 exactly


@pytest.mark.skipif(not has_trusted_solver(), reason="certification needs CLARABEL/MOSEK")
def test_sdp_certified():
    lb, solver, status = bootstrap_single_matrix_qm(1.0, L=4, maximize=False, with_status=True)
    assert lb is not None
    assert solver in ("MOSEK", "CLARABEL") and status == "optimal"
