# matrix_master_field/tests/test_m5c_loop_equations.py
"""M5c — verify QM stationarity + SU(N) Gauss law on the exact g=0 Gaussian moments (T5,T2)."""
import numpy as np

from matrix_master_field.tm_qm_relations import (
    g0_moment,
    gauss_terms,
    stationarity_terms,
)

LETTERS = (0, 1, 2, 3)  # X̃, Ỹ, P̃_X, P̃_Y


def _words_upto(L):
    out = [()]
    cur = [()]
    for _ in range(L):
        cur = [w + (c,) for w in cur for c in LETTERS]
        out += cur
    return out


def test_g0_moments_match_anchor():
    m = 1.0
    assert abs(g0_moment((0, 0), m) - 1.0 / (2.0 * m)) < 1e-12   # m[X̃²]=1/(2m)
    assert abs(g0_moment((2, 2), m) - m / 2.0) < 1e-12           # m[P̃_X²]=m/2
    assert abs(g0_moment((0, 2), m) - 0.5j) < 1e-12              # Gauss: m[X̃P̃_X]=i/2
    assert abs(g0_moment((0, 1), m)) < 1e-12                     # X̃Ỹ independent → 0


def test_stationarity_residual_zero_on_g0_moments():
    m = 1.0
    for w in _words_upto(3):
        terms = stationarity_terms(w)               # at λ=0 (commutator term dropped)
        resid = sum(c(m, 0.0) * g0_moment(ww, m) for c, ww in terms)
        assert abs(resid) < 1e-10, (w, resid)


def test_gauss_law_residual_zero_on_g0_moments():
    m = 1.0
    for O in _words_upto(2):
        resid = sum(c * g0_moment(ww, m) for c, ww in gauss_terms(O))
        assert abs(resid) < 1e-10, (O, resid)
