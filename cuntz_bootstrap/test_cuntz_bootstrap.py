"""Test suite for cuntz_bootstrap Phase 4 implementation.

Organisation:
- @pytest.mark.unit      — fast structural tests (Cuntz algebra, shapes, API)
- @pytest.mark.integration — physics validation (differentiability, optimizer step)
- @pytest.mark.slow      — full phase experiments
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from cuntz_bootstrap.fock import CuntzFockJAX


# -------------------------------------------------------------------------
# Task 2: JAX Cuntz-Fock space
# -------------------------------------------------------------------------


@pytest.mark.unit
def test_basis_dimension_formula():
    """dim = (n^{L+1} - 1)/(n - 1) for n > 1; dim = L + 1 for n = 1."""
    # n = 2D for D=2: n=4, L=3 -> (4^4 - 1)/3 = 85
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    assert f.dim == 85
    # n = 1, L = 6: dim = 7
    f1 = CuntzFockJAX(n_labels=1, L_trunc=6)
    assert f1.dim == 7


@pytest.mark.unit
def test_vacuum_basis_index_zero():
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    assert f.basis[0] == ()
    assert f.basis_to_idx[()] == 0


@pytest.mark.unit
def test_cuntz_relation_interior():
    """â_i â†_j = δ_{ij} on states of length < L (interior)."""
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    interior_idx = np.array(
        [k for k, w in enumerate(f.basis) if len(w) < f.L_trunc]
    )
    for i in range(4):
        for j in range(4):
            prod = np.asarray(f.a[i] @ f.adag[j])
            expected = float(i == j) * np.eye(f.dim)
            got_sub = prod[np.ix_(interior_idx, interior_idx)]
            exp_sub = expected[np.ix_(interior_idx, interior_idx)]
            err = np.max(np.abs(got_sub - exp_sub))
            assert err < 1e-12, f"(i={i}, j={j}) err={err}"


@pytest.mark.unit
def test_completeness_interior():
    """Σ_i â†_i â_i = I - |Ω⟩⟨Ω| on non-vacuum interior words (length < L)."""
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    sum_num = sum(f.adag[i] @ f.a[i] for i in range(4))
    sum_num = np.asarray(sum_num)
    P_Om = np.asarray(f.vacuum_projector())
    expected = np.eye(f.dim) - P_Om
    interior_idx = np.array(
        [k for k, w in enumerate(f.basis) if 0 < len(w) < f.L_trunc]
    )
    err = np.max(
        np.abs(
            sum_num[np.ix_(interior_idx, interior_idx)]
            - expected[np.ix_(interior_idx, interior_idx)]
        )
    )
    assert err < 1e-12


@pytest.mark.unit
def test_vacuum_vector_shape_and_value():
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    vac = f.vacuum_state()
    assert vac.shape == (f.dim,)
    assert float(jnp.real(vac[0])) == 1.0
    assert float(jnp.sum(jnp.abs(vac[1:]))) == 0.0


@pytest.mark.unit
def test_adag_creates_state():
    """â†_i |Ω⟩ = |i⟩."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    vac = f.vacuum_state()
    created = f.adag[0] @ vac
    # Expected: basis vector at index of word (0,)
    idx = f.basis_to_idx[(0,)]
    target = np.zeros(f.dim, dtype=np.complex128)
    target[idx] = 1.0
    err = float(jnp.max(jnp.abs(created - jnp.asarray(target))))
    assert err < 1e-12


@pytest.mark.unit
def test_a_annihilates_vacuum_to_zero():
    """â_i |Ω⟩ = 0."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    vac = f.vacuum_state()
    ann = f.a[0] @ vac
    err = float(jnp.max(jnp.abs(ann)))
    assert err < 1e-12


@pytest.mark.unit
def test_adag_truncation_boundary():
    """â†_i |w⟩ = 0 when len(w) = L_trunc (no room to add)."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    # Pick a length-2 word at the truncation boundary
    idx = f.basis_to_idx[(0, 0)]
    e = np.zeros(f.dim, dtype=np.complex128)
    e[idx] = 1.0
    e = jnp.asarray(e)
    result = f.adag[0] @ e
    err = float(jnp.max(jnp.abs(result)))
    assert err < 1e-12, f"Expected 0, got max entry {err}"
