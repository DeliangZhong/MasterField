"""M5c — the full two-matrix-QM sandwich + fail-closed gate."""
import os
import pytest

from matrix_master_field.bootstrap_sdp import HAS_CVXPY, has_trusted_solver
from matrix_master_field.train import solve_two_matrix_qm


def test_gate_rejects_bare_bracket_inclusion():
    # V6 + F4: bracket inclusion is necessary, NOT sufficient — every invariant must hold.
    from matrix_master_field.train import _tm_qm_gate

    good = dict(E_lo=1.0, E_lo_cert=True, E_hi=3.0, E_mf=2.0, mf_converged=True,
                phi_cond=1e3, sym_loss=1e-9, grad_norm=1e-5, e_tol=1e-2)
    val_ok, ok = _tm_qm_gate(**good)
    assert ok is True and val_ok["in_bracket"] is True

    # deliberately fail each gate input in turn → validated must flip False
    for key, bad in [("mf_converged", False), ("E_lo_cert", False),
                     ("phi_cond", 1e12), ("sym_loss", 1e-3), ("grad_norm", 1.0)]:
        _, ok_bad = _tm_qm_gate(**{**good, key: bad})
        assert ok_bad is False, key

    # inside the bracket but failing everything else → NOT validated
    val, ok_bare = _tm_qm_gate(E_lo=1.0, E_lo_cert=False, E_hi=3.0, E_mf=2.0,
                               mf_converged=False, phi_cond=1e12, sym_loss=1e-3,
                               grad_norm=1.0, e_tol=1e-2)
    assert ok_bare is False
    assert val["in_bracket"] is True        # inside the bracket… but rejected


@pytest.mark.skipif(not (HAS_CVXPY and has_trusted_solver()),
                    reason="certified sandwich needs CLARABEL/MOSEK")
@pytest.mark.skipif(not os.environ.get("MMF_SLOW"),
                    reason="slow: SDP + Cuntz–Fock optimization; set MMF_SLOW=1")
def test_solve_two_matrix_qm_validated_lambda0():
    r = solve_two_matrix_qm(1.0, 0.0, L=4)
    assert r["validated"] is True
    assert r["validation"]["E_lo"] <= 2.0 + 1e-2 <= r["E_hi"] + 1e-2   # the squeeze on 2m
