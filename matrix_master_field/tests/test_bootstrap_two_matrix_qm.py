"""M5c — certified SDP lower bound for two-matrix QM (HHK Eq 17)."""
import pytest

from matrix_master_field.bootstrap_sdp import (
    HAS_CVXPY,
    bootstrap_two_matrix_qm,
    has_trusted_solver,
)
from matrix_master_field.qm_master_field import gaussian_master_field

pytestmark = pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")


def test_lower_bound_anchor_lambda0():
    # λ=0 → E/N²=2m exactly; the SDP lower bound must not exceed it.
    lb = bootstrap_two_matrix_qm(1.0, 0.0, L=4, maximize=False)
    assert lb is not None
    assert lb <= 2.0 + 1e-4
    assert lb >= 2.0 - 0.5        # not a trivial ≥0 collapse


def test_lower_bound_brackets_gaussian():
    # E_lo (SDP) ≤ E/N² ≤ E_hi (Gaussian) for λ>0.
    for lam in (0.5, 1.0):
        lb = bootstrap_two_matrix_qm(1.0, lam, L=4, maximize=False)
        ub = gaussian_master_field(1.0, lam)["energy"]
        assert lb is not None
        assert lb <= ub + 1e-3


@pytest.mark.skipif(not has_trusted_solver(), reason="certification needs CLARABEL/MOSEK")
def test_sdp_certified():
    lb, solver, status = bootstrap_two_matrix_qm(1.0, 1.0, L=4, maximize=False, with_status=True)
    assert lb is not None
    assert solver in ("MOSEK", "CLARABEL") and status == "optimal"
