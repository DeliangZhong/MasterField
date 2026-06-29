"""Tests for onematrix_alm — exact-factorization (Q = m mᵀ) master field via ALM.

The thesis: imposing the factorized loop equations EXACTLY (the non-linear constraint)
pins the unique large-N moment sequence, where the convex Q ⪰ m mᵀ relaxation leaves a
wide bound (Gaussian m₄ ∈ [2.0, 4.05], m₈⁺ unbounded). Both encodings — moments+ALM and
the operator (Jacobi) field — land on the same exact answer.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from matrix_master_field.one_matrix import gaussian_moments, quartic_moments_from_sd
from matrix_master_field.onematrix_alm import (
    _alm_one,
    alm_min_max,
    loop_residuals,
    method_a_jacobi,
)


def _catalan_even(K):
    return gaussian_moments(2 * K)[0 : 2 * K + 1 : 2]  # μ_0..μ_K = C_0..C_K


def test_loop_residuals_g0_catalan_zero():
    """Catalan moments satisfy the g=0 factorized loop equations exactly."""
    h = np.asarray(loop_residuals(jnp.asarray(_catalan_even(6)), 0.0))
    assert np.max(np.abs(h)) < 1e-10


def test_g0_alm_recovers_catalan():
    """g=0: the loop equations pin μ_1..μ_{K-1} = Catalan (1,2,5,14,...). The TOP
    moment μ_K is the free DOF at g=0 (its g-coupling vanishes), so it is left to the
    soft Hankel penalty and not asserted here."""
    K = 5
    mu, res = _alm_one(0.0, K, 1, +1.0, _catalan_even(K)[1:])
    cat = gaussian_moments(2 * K)
    assert res < 1e-5
    for p in range(1, K):  # loop-determined moments only (μ_K is free at g=0)
        assert abs(mu[p] - cat[2 * p]) < 1e-4


@pytest.mark.parametrize("g", [0.5, 1.0])
def test_exact_factorization_bracket(g):
    """g>0: the exact-factorization bracket CONTAINS the exact moment and is far
    tighter than the convex bound (~0.4 wide on m₂/m₄, unbounded on m₆)."""
    K = 5
    exact = quartic_moments_from_sd(g, 2 * K)
    for p in (1, 2):
        lo, hi, res, *_ = alm_min_max(g, K, p, restarts=2)
        assert lo - 2e-3 <= exact[2 * p] <= hi + 2e-3
        assert hi - lo < 0.05  # vs convex width ~0.4


def test_method_a_gaussian_is_semicircle():
    """Method A (Jacobi operator): g=0 → b_n = 1 (semicircle), Catalan to machine zero."""
    K = 6
    mu, res, b = method_a_jacobi(0.0, K)
    cat = gaussian_moments(2 * K)
    assert res < 1e-8
    assert np.allclose(b[:K], 1.0, atol=1e-4)
    for p in range(1, K + 1):
        assert abs(mu[p] - cat[2 * p]) < 1e-8


@pytest.mark.parametrize("g", [0.5, 1.0])
def test_method_a_matches_exact(g):
    """Method A (operator) recovers the exact quartic moments — same answer as method B
    (moments+ALM): the precision is the non-linear constraint, not the parametrization."""
    K = 6
    mu, res, b = method_a_jacobi(g, K)
    exact = quartic_moments_from_sd(g, 2 * K)
    for p in (1, 2, 3):
        assert abs(mu[p] - exact[2 * p]) / abs(exact[2 * p]) < 1e-2
