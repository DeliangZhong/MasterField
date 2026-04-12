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


# -------------------------------------------------------------------------
# Task 3: Polynomial master operator
# -------------------------------------------------------------------------

from cuntz_bootstrap.master_operator import (  # noqa: E402
    assemble_master_operator,
    direction_to_label,
    init_master_operator_params,
)


@pytest.mark.unit
def test_direction_to_label_roundtrip():
    """+1 -> 0, -1 -> 1, +2 -> 2, -2 -> 3 for D = 2."""
    assert direction_to_label(1, D=2) == 0
    assert direction_to_label(-1, D=2) == 1
    assert direction_to_label(2, D=2) == 2
    assert direction_to_label(-2, D=2) == 3


@pytest.mark.unit
def test_direction_to_label_rejects_zero():
    with pytest.raises(ValueError):
        direction_to_label(0, D=2)


@pytest.mark.unit
def test_direction_to_label_rejects_out_of_range():
    with pytest.raises(ValueError):
        direction_to_label(3, D=2)


@pytest.mark.unit
def test_init_master_operator_params_shape():
    """Param count = 2*d_L - 1 complex per matrix."""
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    params = init_master_operator_params(n_matrices=2, fock=f, seed=0)
    assert len(params) == 2
    expected_n = 2 * f.dim - 1
    for p in params:
        assert p.shape == (expected_n,)
        assert p.dtype == jnp.complex128


@pytest.mark.unit
def test_assemble_master_operator_zero_gives_zero():
    """c = 0 → Û = 0 matrix."""
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    c = jnp.zeros(2 * f.dim - 1, dtype=jnp.complex128)
    U = assemble_master_operator(c, fock=f)
    assert float(jnp.max(jnp.abs(U))) < 1e-15


@pytest.mark.unit
def test_assemble_master_operator_identity_coefficient():
    """c[0] = 1, others = 0 (empty-word creation = identity) → Û = I."""
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    c = jnp.zeros(2 * f.dim - 1, dtype=jnp.complex128).at[0].set(1.0)
    U = assemble_master_operator(c, fock=f)
    err = float(jnp.max(jnp.abs(U - jnp.eye(f.dim, dtype=jnp.complex128))))
    assert err < 1e-12


@pytest.mark.unit
def test_assemble_master_operator_single_creation():
    """c[1] = 1 → Û = â†_0 (basis[1] = (0,) in canonical enumeration)."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    assert f.basis[1] == (0,), "Test assumes basis[1] = (0,)"
    c = jnp.zeros(2 * f.dim - 1, dtype=jnp.complex128).at[1].set(1.0)
    U = assemble_master_operator(c, fock=f)
    err = float(jnp.max(jnp.abs(U - f.adag[0])))
    assert err < 1e-12


@pytest.mark.unit
def test_assemble_master_operator_single_annihilation():
    """c[d] = 1 → Û = â_0 (first annihilation coefficient)."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    assert f.basis[1] == (0,), "Test assumes basis[1] = (0,)"
    c = jnp.zeros(2 * f.dim - 1, dtype=jnp.complex128).at[f.dim].set(1.0)
    U = assemble_master_operator(c, fock=f)
    err = float(jnp.max(jnp.abs(U - f.a[0])))
    assert err < 1e-12


@pytest.mark.unit
def test_assemble_master_operator_two_creation_words():
    """Linear combination: c[1] = 2, c[2] = 3 → Û = 2 â†_0 + 3 â†_1."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    assert f.basis[1] == (0,) and f.basis[2] == (1,)
    c = jnp.zeros(2 * f.dim - 1, dtype=jnp.complex128).at[1].set(2.0).at[2].set(3.0)
    U = assemble_master_operator(c, fock=f)
    expected = 2.0 * f.adag[0] + 3.0 * f.adag[1]
    err = float(jnp.max(jnp.abs(U - expected)))
    assert err < 1e-12


@pytest.mark.unit
def test_assemble_master_operator_length2_creation():
    """c for the word (0, 1) → Û = â†_0 @ â†_1 (applied to |Ω⟩ gives |0, 1⟩)."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    idx = f.basis_to_idx[(0, 1)]
    c = jnp.zeros(2 * f.dim - 1, dtype=jnp.complex128).at[idx].set(1.0)
    U = assemble_master_operator(c, fock=f)
    # Apply to vacuum, should give basis vector |0, 1⟩
    v_out = U @ f.vacuum_state()
    target = np.zeros(f.dim, dtype=np.complex128)
    target[f.basis_to_idx[(0, 1)]] = 1.0
    err = float(jnp.max(jnp.abs(v_out - jnp.asarray(target))))
    assert err < 1e-12


@pytest.mark.unit
def test_assemble_master_operator_rejects_wrong_shape():
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    with pytest.raises(ValueError):
        assemble_master_operator(jnp.zeros(10, dtype=jnp.complex128), fock=f)


# -------------------------------------------------------------------------
# Task 4: Unitarity loss
# -------------------------------------------------------------------------

from cuntz_bootstrap.unitarity import (  # noqa: E402
    unitarity_loss,
    unitarity_loss_from_params,
)


@pytest.mark.unit
def test_unitarity_loss_identity_zero():
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    I = jnp.eye(f.dim, dtype=jnp.complex128)
    loss = unitarity_loss([I, I])
    assert float(loss) < 1e-20


@pytest.mark.unit
def test_unitarity_loss_random_nonzero():
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    A = (
        jax.random.normal(k1, (f.dim, f.dim)) + 1j * jax.random.normal(k2, (f.dim, f.dim))
    ).astype(jnp.complex128)
    loss = unitarity_loss([A])
    assert float(loss) > 1.0


@pytest.mark.unit
def test_unitarity_loss_from_params_differentiable():
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    params = init_master_operator_params(n_matrices=2, fock=f, seed=0)
    loss, grads = jax.value_and_grad(unitarity_loss_from_params, argnums=0)(params, f)
    assert bool(jnp.isfinite(loss))
    for g in grads:
        assert bool(jnp.all(jnp.isfinite(g)))
        # gradient must be nonzero somewhere (loss not already at minimum)
        assert float(jnp.max(jnp.abs(g))) > 0.0


@pytest.mark.unit
def test_unitarity_loss_from_identity_params_zero():
    """c = (1, 0, ..., 0) for each matrix → Û = I → loss = 0."""
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    c0 = jnp.zeros(2 * f.dim - 1, dtype=jnp.complex128).at[0].set(1.0)
    params = [c0, c0]
    loss = unitarity_loss_from_params(params, f)
    assert float(loss) < 1e-20
