"""Milestone 4 — Kazakov–Zheng "unsolvable" two-matrix model (arXiv:2108.04830 eq.6):
    S = N·tr[½(A²+B²) + (g/4)(A⁴+B⁴) − (h/2)[A,B]²],
force V'_a = M_a + g·M_a³ + h·(M_a M_b² + M_b² M_a − 2 M_b M_a M_b).
"""
import os

import jax
import numpy as np
import pytest

from matrix_master_field.bootstrap_sdp import (
    HAS_CVXPY,
    bootstrap_one_matrix,
    bootstrap_two_matrix,
    bootstrap_two_matrix_kz,
    has_trusted_solver,
)
from matrix_master_field.loss import kz_sd_residual_from_moment, two_matrix_test_words
from matrix_master_field.sparse_fock import SparseMonomialField
from matrix_master_field.train import solve_kz_sparse

_needs_cert = pytest.mark.skipif(
    not (HAS_CVXPY and has_trusted_solver()),
    reason="validated=True / certified island needs a trusted SDP solver (CLARABEL/MOSEK)",
)


def _exact_quartic_trM2(g):
    # one-matrix V=½M²+(g/4)M⁴, V'=M+gM³: one-cut band edge b=a² solves
    # b/4 + (3g/16)b² = 1 ⇒ 3g b² + 4b − 16 = 0; density (1/2π)(g x²+c)√(b−x²),
    # c=1+g b/2; ⟨tr M²⟩ = g b³/32 + c b²/16.
    b = (-4.0 + np.sqrt(16.0 + 192.0 * g)) / (6.0 * g)
    c = 1.0 + g * b / 2.0
    return g * b**3 / 32.0 + c * b**2 / 16.0


def test_kz_force_finite_difference():
    # The derived force V'_A = A + g A³ + h(AB²+B²A−2BAB) must equal ∂_A tr V:
    # for any Hermitian H, d/dε tr V(A+εH,B) = tr(V'_A · H).
    rng = np.random.default_rng(0)
    g, h, n, eps = 0.7, 0.4, 6, 1e-6

    def herm():
        M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        return (M + M.conj().T) / 2

    def trV(A, B):
        C = A @ B - B @ A
        return np.trace(0.5 * (A @ A + B @ B) + (g / 4) * (A @ A @ A @ A + B @ B @ B @ B)
                        - (h / 2) * (C @ C)).real

    def VpA(A, B):
        return A + g * (A @ A @ A) + h * (A @ B @ B + B @ B @ A - 2 * B @ A @ B)

    for _ in range(8):
        A, B, H = herm(), herm(), herm()
        fd = (trV(A + eps * H, B) - trV(A - eps * H, B)) / (2 * eps)
        assert abs(fd - np.trace(VpA(A, B) @ H).real) < 1e-5


def test_kz_residual_vanishes_at_free_field():
    # At g=h=0 the model is two decoupled free semicirculars, which satisfy the
    # (mass-only) loop equations exactly — the KZ residual must vanish there.
    field = SparseMonomialField(2, cutoff=6, degree=2)
    p = field.params_for_free_field()
    words = two_matrix_test_words(3)
    r = float(kz_sd_residual_from_moment(lambda w: field.word_moment(p, w), words, 0.0, 0.0))
    assert r < 1e-18


@_needs_cert
def test_kz_sdp_reduces_to_commutator_at_g0():
    # quartic g=0: KZ force = comm·(...) with comm=h; the commutator model has
    # comm=g_c²/2, so KZ(g=0,h) must equal bootstrap_two_matrix(g_c) with h=g_c²/2.
    h, g_c = 0.5, 1.0  # g_c²/2 = 0.5 = h
    for mx in (False, True):
        kz = bootstrap_two_matrix_kz(0.0, h, max_word_len=6, target_word=(0, 0), maximize=mx)
        cm = bootstrap_two_matrix(g_c, max_word_len=6, target_word=(0, 0), maximize=mx)
        assert abs(kz - cm) < 1e-6


@_needs_cert
def test_kz_sdp_h0_brackets_exact_quartic():
    # commutator h=0: the model decouples into two quartic one-matrix models, so the
    # KZ tr A² island must bracket the EXACT quartic ⟨tr M²⟩ (and the 1-matrix
    # bootstrap value too).
    g = 0.5
    lo = bootstrap_two_matrix_kz(g, 0.0, max_word_len=8, target_word=(0, 0), maximize=False)
    hi = bootstrap_two_matrix_kz(g, 0.0, max_word_len=8, target_word=(0, 0), maximize=True)
    exact = _exact_quartic_trM2(g)
    assert 0.62 < exact < 0.64                       # ≈ 0.6312
    assert lo - 1e-6 <= exact <= hi + 1e-6           # island brackets the exact value
    # also brackets the one-matrix bootstrap value
    om_hi = bootstrap_one_matrix([0.0, 1.0, 0.0, g], max_moment=8, target_moment=2, maximize=True)
    assert lo <= om_hi + 1e-6


@_needs_cert
def test_kz_solve_max_word_len2_artifact_rejected():
    # As in the commutator model, max_word_len=2 solves the (few) loop equations to
    # machine zero but lands OUTSIDE the tight L=8 island — a truncation artifact the
    # fail-closed gate must reject.
    field = SparseMonomialField(2, cutoff=6, degree=3)
    r = solve_kz_sparse(field, 0.5, 0.3, max_word_len=2, n_stages=4, steps=1500, sdp_word_len=8)
    assert r["sd_loss"] < 1e-6 and r["validation"]["sym_ok"] is True
    assert r["validation"]["in_island"] is False
    assert r["validated"] is False


@_needs_cert
@pytest.mark.skipif(not os.environ.get("MMF_SLOW"),
                    reason="slow: dim-1023 KZ max_word_len=3 solve (~1-2 min); set MMF_SLOW=1")
def test_kz_solve_validated_in_island():
    # The Milestone-4 result: the KZ operator master field at max_word_len=3 lands
    # INSIDE the certified KZ SDP island and passes the fail-closed gate.
    field = SparseMonomialField(2, cutoff=9, degree=3)
    r = solve_kz_sparse(field, 0.5, 0.3, max_word_len=3, n_stages=5, steps=1200, sdp_word_len=8)
    assert r["sd_loss"] < 1e-8
    assert r["validation"]["sym_ok"] is True
    assert r["validation"]["in_island"] is True
    assert r["validation"]["island_certified"] is True
    assert r["validated"] is True
    lb, ub = r["validation"]["island"]
    assert lb <= float(field.word_moment(r["params"], (0, 0))) <= ub
