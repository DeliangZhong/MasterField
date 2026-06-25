"""M5b — the single-matrix-QM sandwich: certified SDP lower bound ≤ E/N² ≤ collective
variational upper bound, refereed by the exact free fermions."""
import os

import jax

jax.config.update("jax_enable_x64", True)

import pytest  # noqa: E402

from matrix_master_field.bootstrap_sdp import HAS_CVXPY, has_trusted_solver  # noqa: E402
from matrix_master_field.train import _sm_qm_gate, solve_single_matrix_qm  # noqa: E402


def test_sm_qm_gate_logic_pure():
    base = dict(E_lo=1.18, E_lo_cert=True, E_var=1.305, E_exact=1.302, e_tol=1e-3)
    _, ok = _sm_qm_gate(**base)
    assert ok is True
    _, ok = _sm_qm_gate(**{**base, "E_lo_cert": False})
    assert ok is False  # uncertified
    _, ok = _sm_qm_gate(**{**base, "E_lo": 1.40})
    assert ok is False  # lower bound above the truth
    _, ok = _sm_qm_gate(**{**base, "E_var": 1.20})
    assert ok is False  # variational upper bound below the truth


@pytest.mark.skipif(not (HAS_CVXPY and has_trusted_solver()),
                    reason="certified sandwich needs CLARABEL/MOSEK")
@pytest.mark.skipif(not os.environ.get("MMF_SLOW"),
                    reason="slow: collective variational + SDP; set MMF_SLOW=1")
def test_solve_single_matrix_qm_validated():
    r = solve_single_matrix_qm(1.0, L=4)
    assert r["validated"] is True
    v = r["validation"]
    assert v["E_lo"] <= r["E_exact"] <= r["E_var"] + 1e-3   # the squeeze
    assert v["certified"] is True
