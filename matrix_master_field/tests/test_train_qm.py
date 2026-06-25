"""M5a — the certified sandwich: E_lo <= E0 <= E_var, validated against exact diag."""
import os

import jax

jax.config.update("jax_enable_x64", True)

import pytest  # noqa: E402

from matrix_master_field.bootstrap_sdp import HAS_CVXPY, has_trusted_solver  # noqa: E402
from matrix_master_field.train import _qm_gate, solve_qm_anharmonic  # noqa: E402


def test_qm_gate_logic_pure():
    base = dict(E_lo=1.30, E_lo_cert=True, m2_island=(0.28, 0.33), m2_island_cert=True,
                E_var=1.3924, E_exact=1.3924, m2_exact=0.3058, e_tol=1e-3, m2_tol=1e-3)
    _, ok = _qm_gate(**base)
    assert ok is True
    _, ok = _qm_gate(**{**base, "E_lo_cert": False})
    assert ok is False  # uncertified lower bound
    _, ok = _qm_gate(**{**base, "m2_island_cert": False})
    assert ok is False  # uncertified island
    _, ok = _qm_gate(**{**base, "E_lo": 1.50})
    assert ok is False  # lower bound above the truth
    _, ok = _qm_gate(**{**base, "E_var": 1.30})
    assert ok is False  # variational bound below the truth
    _, ok = _qm_gate(**{**base, "m2_island": (0.10, 0.20)})
    assert ok is False  # <x^2> outside the island


@pytest.mark.skipif(not (HAS_CVXPY and has_trusted_solver()),
                    reason="certified sandwich needs CLARABEL/MOSEK")
@pytest.mark.skipif(not os.environ.get("MMF_SLOW"),
                    reason="slow: bisection over SDP solves; set MMF_SLOW=1")
def test_solve_qm_validated_sandwich():
    r = solve_qm_anharmonic(1.0, K=24, K_sdp=6)
    assert r["validated"] is True
    v = r["validation"]
    assert v["E_lo"] <= r["E_exact"] <= r["E_var"] + 1e-3  # the squeeze
    assert v["certified"] is True
