"""Milestone 4 (amortization) — one MLP (g,h) ↦ Kazakov–Zheng master field M̂(g,h)
across the coupling plane, generalizing to held-out couplings."""
import os

import jax
import pytest

from matrix_master_field.amortized import AmortizedKZ, train_amortized_kz
from matrix_master_field.bootstrap_sdp import (
    HAS_CVXPY,
    bootstrap_two_matrix_kz,
    has_trusted_solver,
)
from matrix_master_field.loss import kz_sd_residual_from_moment, two_matrix_test_words
from matrix_master_field.sparse_fock import SparseMonomialField


def test_amortized_kz_init_near_free_field():
    # The output bias is warm-started to the free field, so at init the net returns
    # ~the free field for ANY (g,h): tr A^2 ≈ 1 (a cold net would give ~0).
    field = SparseMonomialField(2, 6, 3)
    model = AmortizedKZ(field, hidden=32)
    theta = model.init_params(jax.random.PRNGKey(0))
    for (g, h) in [(0.0, 0.0), (0.6, 0.4)]:
        p = model.coeffs(theta, g, h)
        assert abs(float(field.word_moment(p, (0, 0))) - 1.0) < 0.3


def test_amortized_kz_trains_and_generalizes():
    # One MLP (g,h)->M̂(g,h) trained on a 4-point grid reaches low mean loop-equation
    # residual AND produces a low-residual solution at a HELD-OUT (g,h) — i.e. a
    # single network represents the master field across the coupling plane.
    field = SparseMonomialField(2, 6, 3)            # max_word_len=2, dim 127
    model = AmortizedKZ(field, hidden=64)
    grid = [(0.2, 0.2), (0.2, 0.5), (0.6, 0.2), (0.6, 0.5)]
    theta, final = train_amortized_kz(model, grid, max_word_len=2, steps=2500, lr=3e-3)
    assert final < 1e-3                              # learned the training grid

    words = two_matrix_test_words(2)
    p = model.coeffs(theta, 0.4, 0.35)               # held-out coupling (not in grid)
    res = float(kz_sd_residual_from_moment(lambda w: field.word_moment(p, w), words, 0.4, 0.35))
    assert res < 1e-2                                # generalizes to unseen couplings


@pytest.mark.skipif(not (HAS_CVXPY and has_trusted_solver()),
                    reason="certified island needs a trusted SDP solver (CLARABEL/MOSEK)")
@pytest.mark.skipif(not os.environ.get("MMF_SLOW"),
                    reason="slow: dim-1023 amortized training (~3-4 min); set MMF_SLOW=1")
def test_amortized_kz_held_out_in_certified_island():
    # The validated headline: at max_word_len=3 one net trained on a (g,h) grid
    # produces, at a HELD-OUT coupling, a master field INSIDE the certified KZ island.
    field = SparseMonomialField(2, 9, 3)             # max_word_len=3, dim 1023
    model = AmortizedKZ(field, hidden=96)
    grid = [(g, h) for g in (0.3, 0.6) for h in (0.1, 0.3, 0.5)]
    theta, final = train_amortized_kz(model, grid, max_word_len=3, steps=4000, lr=3e-3)
    assert final < 1e-5
    g, h = 0.45, 0.25                                # held-out
    m2 = float(field.word_moment(model.coeffs(theta, g, h), (0, 0)))
    lb, ls, lst = bootstrap_two_matrix_kz(g, h, max_word_len=8, target_word=(0, 0),
                                          maximize=False, with_status=True)
    ub, us, ust = bootstrap_two_matrix_kz(g, h, max_word_len=8, target_word=(0, 0),
                                          maximize=True, with_status=True)
    assert ls in ("MOSEK", "CLARABEL") and lst == "optimal"   # island is certified
    assert us in ("MOSEK", "CLARABEL") and ust == "optimal"
    assert lb - 1e-3 <= m2 <= ub + 1e-3                       # amortized field in-island
